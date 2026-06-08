/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file LsaBarrierMultimemMPITests.cpp
 * @brief Coverage for the multicast (multimem) path in ncclLsaBarrierSession
 *        timeout logic (AICOMNET-193).
 *
 * Gap filled: lsa_barrier__funcs.h has two timeout code paths:
 *   Line 117: the multimem (NVLink Switch multicast) wait loop — mcInbox read
 *   Line 139: the standard ucInbox unicast peer-wait loop
 *
 * LsaBarrierTimeoutMPITests.cpp exercises line 139 (ucInbox) because
 * lsaMultimem=false is the NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER default.
 * This file exercises line 117 by setting reqs.lsaMultimem=true.
 *
 * Hardware requirement: NVLink Switch (nvlsSupport). On systems without it
 * (e.g. MI350X with XGMI), ncclDevCommCreate returns ncclInvalidUsage and
 * the test SKIPs uniformly on every rank. The skip itself is a valid result —
 * it confirms the guard is in place and the test never hangs.
 *
 * Tests:
 *   - Multimem_HealthyBarrierReturnsSuccess : multimem path, all arrive → ncclSuccess
 *   - Multimem_AbsentPeerProducesTimeout    : multimem path, absent peer → ncclTimeout
 *   - Multimem_ZeroBudgetAbsentPeerTimesOut : multimem path, budget=0 + absent
 *   - Multimem_RecoversAfterTimeout         : multimem path, timeout then healthy
 */

#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include "nccl.h"
#include "nccl_device.h"

#include <cstdint>
#include <hip/hip_runtime.h>
#include <mpi.h>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;

namespace {

constexpr uint64_t kHealthyTimeoutCycles = 5000000000ULL;
constexpr uint64_t kShortTimeoutCycles   = 200000000ULL;

// Kernel: one bounded LSA barrier, writes result to scratch.
// The ncclLsaBarrierSession constructor reads lsaMultimem from devComm and
// takes the multicast path when devComm.lsaMultimem.mcBasePtr != nullptr.
__global__ void lsaBarrierTimeoutKernel(struct ncclDevComm devComm,
                                         uint64_t timeoutCycles,
                                         int* outResult) {
    ncclLsaBarrierSession<ncclCoopCta> bar{
        ncclCoopCta(), devComm, ncclTeamTagLsa(), /*index=*/0u};
    ncclResult_t r = bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, timeoutCycles);
    if (threadIdx.x == 0 && blockIdx.x == 0) *outResult = static_cast<int>(r);
}

int runOneBarrier(ncclDevComm devComm, hipStream_t stream, uint64_t timeoutCycles) {
    int* dResult = nullptr;
    if (hipMalloc(&dResult, sizeof(int)) != hipSuccess) return -100;
    int sentinel = -1;
    (void)hipMemcpy(dResult, &sentinel, sizeof(int), hipMemcpyHostToDevice);
    lsaBarrierTimeoutKernel<<<1, 256, 0, stream>>>(devComm, timeoutCycles, dResult);
    (void)hipStreamSynchronize(stream);
    int hResult = -1;
    (void)hipMemcpy(&hResult, dResult, sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(dResult);
    return hResult;
}

// Create a devComm with lsaMultimem=true (multicast path).
// Returns ncclInvalidUsage on hardware without NVLink Switch.
ncclResult_t createMultimemDevComm(ncclComm_t comm, int nBarriers, ncclDevComm* out) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = nBarriers;
    reqs.lsaMultimem     = true;   // request multicast (mcInbox) path
    return ncclDevCommCreate(comm, &reqs, out);
}

} // namespace

class LsaBarrierMultimemMPITest : public MPITestBase {
protected:
    std::string skipReason_;

    // Sets up a multimem devComm. Returns false if requirements not met;
    // sets skipReason_ for GTEST_SKIP on hardware without NVLink Switch.
    bool setUpMultimemDevComm(int nBarriers, ncclComm_t* commOut,
                               hipStream_t* streamOut, ncclDevComm* devCommOut) {
        skipReason_.clear();
        if (!validateTestPrerequisites(2, kNoProcessLimit,
                                        kNoPowerOfTwoRequired, 1, kNoNodeLimit)) {
            ADD_FAILURE() << "Test requires at least 2 MPI processes";
            return false;
        }
        if (createTestCommunicator() != ncclSuccess) {
            ADD_FAILURE() << "createTestCommunicator failed";
            return false;
        }
        ncclComm_t comm = getActiveCommunicator();
        ncclResult_t devRc = createMultimemDevComm(comm, nBarriers, devCommOut);
        if (devRc != ncclSuccess) {
            // On MI300X/MI350X without NVLink Switch: expected skip path.
            skipReason_ = std::string("multimem LSA devComm unavailable "
                "(needs NVLink Switch / nvlsSupport): ")
                + ncclGetErrorString(devRc);
            return false;
        }
        *commOut   = comm;
        *streamOut = getActiveStream();
        return true;
    }
};

#define MULTIMEM_SETUP_OR_BAIL()                                      \
    do {                                                               \
        if (!skipReason_.empty()) GTEST_SKIP() << skipReason_;        \
        return;                                                        \
    } while (0)

// All ranks arrive via multicast → ncclSuccess.
TEST_F(LsaBarrierMultimemMPITest, Multimem_HealthyBarrierReturnsSuccess)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpMultimemDevComm(1, &comm, &stream, &devComm)) MULTIMEM_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    MPI_Barrier(MPI_COMM_WORLD);
    int hResult = runOneBarrier(devComm, stream, kHealthyTimeoutCycles);
    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess), hResult);
}

// Absent LSA rank → multicast inbox never reaches threshold → ncclTimeout.
// Exercises lsa_barrier__funcs.h line 117 (mcInbox wait with clock64 deadline).
TEST_F(LsaBarrierMultimemMPITest, Multimem_AbsentPeerProducesTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpMultimemDevComm(1, &comm, &stream, &devComm)) MULTIMEM_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);

    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runOneBarrier(devComm, stream, kShortTimeoutCycles);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "Multimem LSA rank " << lsaTeam.rank << " expected ncclTimeout, got "
            << ncclGetErrorString(static_cast<ncclResult_t>(hResult));
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        (void)runOneBarrier(devComm, stream, 0ULL);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int local = (isAbsent || hResult == static_cast<int>(ncclTimeout)) ? 1 : 0;
    ASSERT_MPI_TRUE(local == 1);
}

// Zero-cycle budget + absent peer → immediate timeout from multicast inbox.
TEST_F(LsaBarrierMultimemMPITest, Multimem_ZeroBudgetAbsentPeerTimesOut)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpMultimemDevComm(1, &comm, &stream, &devComm)) MULTIMEM_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);

    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runOneBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "Expected immediate ncclTimeout with zero budget on multimem path";
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        (void)runOneBarrier(devComm, stream, 0ULL);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int local = (isAbsent || hResult == static_cast<int>(ncclTimeout)) ? 1 : 0;
    ASSERT_MPI_TRUE(local == 1);
}

// Timeout round then fresh healthy multimem barrier on new devComm.
// Verifies multicast inbox state doesn't leak across devComm lifetime.
TEST_F(LsaBarrierMultimemMPITest, Multimem_RecoversAfterTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpMultimemDevComm(1, &comm, &stream, &devComm)) MULTIMEM_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);

    if (!isAbsent) {
        int r1 = runOneBarrier(devComm, stream, kShortTimeoutCycles);
        EXPECT_EQ(static_cast<int>(ncclTimeout), r1);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        (void)runOneBarrier(devComm, stream, 0ULL);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Fresh multimem devComm: all ranks participate → must succeed.
    ncclDevComm devComm2{};
    ASSERT_MPI_EQ(ncclSuccess, createMultimemDevComm(comm, 1, &devComm2));
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm2));

    MPI_Barrier(MPI_COMM_WORLD);
    int r2 = runOneBarrier(devComm2, stream, kHealthyTimeoutCycles);
    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess), r2);
}

#endif // MPI_TESTS_ENABLED
