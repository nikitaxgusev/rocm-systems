/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file LsaBarrierMultiNodeMPITests.cpp
 * @brief LSA barrier ncclTimeout tests for >2 nodes / >2 rail ranks (AICOMNET-193).
 *
 * Gap filled: LsaBarrierTimeoutMPITests.cpp runs with 2 MPI ranks on 1-2 nodes,
 * covering the single-hop (nRanks-1=1 peer) case. This file exercises the same
 * timeout paths with 4+ ranks across 4 nodes (1 rank/node), where each rank
 * waits for nRanks-1=3 peers. The multi-peer wait loop exercises more of the
 * rotating-peer pattern in syncInternal and verifies timeout fires correctly
 * when any one of the >1 expected peers is absent.
 *
 * Run with: --map-by ppr:1:node, 4 nodes minimum.
 * Skips cleanly (GTEST_SKIP) when symmetric memory is unavailable.
 *
 * Tests:
 *   - MultiNode_HealthyBarrierReturnsSuccess   : all 4 ranks arrive → ncclSuccess
 *   - MultiNode_AbsentPeerProducesTimeout      : last LSA rank absent → ncclTimeout
 *   - MultiNode_ZeroBudgetAbsentPeerTimesOut   : budget=0 + absent → immediate timeout
 *   - MultiNode_RecoversAfterTimeout           : timeout then fresh healthy barrier
 *   - MultiNode_BackToBackTimeouts             : 4 sequential stuck barriers
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

ncclResult_t createLsaDevComm(ncclComm_t comm, int nBarriers, ncclDevComm* out) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = nBarriers;
    return ncclDevCommCreate(comm, &reqs, out);
}

} // namespace

class LsaBarrierMultiNodeMPITest : public MPITestBase {
protected:
    std::string skipReason_;

    bool setUpLsaDevComm(int nBarriers, ncclComm_t* commOut,
                          hipStream_t* streamOut, ncclDevComm* devCommOut) {
        skipReason_.clear();
        // Require at least 4 ranks across at least 4 nodes (1 per node).
        if (!validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired,
                                        4, kNoNodeLimit)) {
            skipReason_ = "Test requires at least 4 MPI ranks on 4 separate nodes";
            return false;
        }
        if (createTestCommunicator() != ncclSuccess) {
            ADD_FAILURE() << "createTestCommunicator failed";
            return false;
        }
        ncclComm_t comm = getActiveCommunicator();
        ncclResult_t devRc = createLsaDevComm(comm, nBarriers, devCommOut);
        if (devRc != ncclSuccess) {
            skipReason_ = "LSA devComm unavailable (needs symmetric memory / kernel >= 6.8)";
            return false;
        }
        *commOut   = comm;
        *streamOut = getActiveStream();
        return true;
    }
};

#define LSA_MN_SETUP_OR_BAIL()                                      \
    do {                                                             \
        if (!skipReason_.empty()) GTEST_SKIP() << skipReason_;      \
        return;                                                      \
    } while (0)

// All 4 ranks arrive → success.
TEST_F(LsaBarrierMultiNodeMPITest, MultiNode_HealthyBarrierReturnsSuccess)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_MN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    MPI_Barrier(MPI_COMM_WORLD);
    int hResult = runOneBarrier(devComm, stream, kHealthyTimeoutCycles);
    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess), hResult);
}

// Last LSA rank absent → all present ranks time out.
TEST_F(LsaBarrierMultiNodeMPITest, MultiNode_AbsentPeerProducesTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_MN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);

    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runOneBarrier(devComm, stream, kShortTimeoutCycles);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "LSA rank " << lsaTeam.rank << "/" << lsaTeam.nRanks
            << " expected ncclTimeout, got "
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

// budget=0 + absent → immediate timeout.
TEST_F(LsaBarrierMultiNodeMPITest, MultiNode_ZeroBudgetAbsentPeerTimesOut)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_MN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);

    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runOneBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "Expected immediate ncclTimeout with zero budget";
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

// Timeout round then fresh healthy barrier on new devComm.
TEST_F(LsaBarrierMultiNodeMPITest, MultiNode_RecoversAfterTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_MN_SETUP_OR_BAIL();
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

    ncclDevComm devComm2{};
    ASSERT_MPI_EQ(ncclSuccess, createLsaDevComm(comm, 1, &devComm2));
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm2));

    MPI_Barrier(MPI_COMM_WORLD);
    int r2 = runOneBarrier(devComm2, stream, kHealthyTimeoutCycles);
    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess), r2);
}

// 4 sequential stuck barriers, each must return ncclTimeout.
TEST_F(LsaBarrierMultiNodeMPITest, MultiNode_BackToBackTimeouts)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_MN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);

    constexpr int kRounds = 4;
    int timeouts = 0;
    for (int i = 0; i < kRounds; ++i) {
        ncclDevComm dc{};
        MPI_Barrier(MPI_COMM_WORLD);
        if (createLsaDevComm(comm, 1, &dc) != ncclSuccess) break;
        if (!isAbsent) {
            int r = runOneBarrier(dc, stream, kShortTimeoutCycles);
            if (r == static_cast<int>(ncclTimeout)) ++timeouts;
            (void)hipStreamSynchronize(stream);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (isAbsent) {
            (void)runOneBarrier(dc, stream, 0ULL);
            (void)hipStreamSynchronize(stream);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        (void)ncclDevCommDestroy(comm, &dc);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (!isAbsent) EXPECT_EQ(kRounds, timeouts);
    MPI_Barrier(MPI_COMM_WORLD);

    int local = (isAbsent || timeouts == kRounds) ? 1 : 0;
    ASSERT_MPI_TRUE(local == 1);
}

#endif // MPI_TESTS_ENABLED
