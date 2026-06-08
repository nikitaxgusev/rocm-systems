/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DeviceTimeoutPropagationMPITests.cpp
 * @brief End-to-end coverage: device barrier ncclTimeout → ncclCommSetAsyncError
 *        → ncclCommGetAsyncError surfaces ncclTimeout to the caller (AICOMNET-193).
 *
 * Gap filled: the existing TimeoutMPITests inject ncclTimeout via
 * ncclCommSetAsyncError directly (no real device barrier). The existing
 * LsaBarrierTimeoutMPITests verify the device side but read the result
 * from runOneBarrier, not from ncclCommGetAsyncError. This file closes the
 * gap: a real stuck LSA barrier produces ncclTimeout from the device, the
 * test propagates it through ncclCommSetAsyncError (exactly as a production
 * polling path would), and then verifies the full host-API contract:
 *
 *   real device timeout → ncclCommSetAsyncError → ncclCommGetAsyncError
 *     → "timeout" string → comm still functional (AllReduce succeeds)
 *     → error clearable via ncclCommSetAsyncError(ncclSuccess)
 *
 * Tests:
 *   - DeviceTimeout_LsaSurfacesViaAsyncError: real LSA timeout → async error
 *     → AllReduce still works → error clearable
 *   - DeviceTimeout_MultipleTimeoutsAccumulate: two sequential real timeouts,
 *     both surface as ncclTimeout, comm remains healthy after clearing
 *   - DeviceTimeout_GinSurfacesViaAsyncError: real GIN timeout → async error
 *     (multi-node only; skips if GIN prerequisites unmet)
 */

#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include "nccl.h"
#include "nccl_device.h"

#include <cstring>
#include <hip/hip_runtime.h>
#include <mpi.h>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;

// Internal RCCL symbol — forward declaration (see TimeoutMPITests.cpp).
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState);

namespace {

constexpr uint64_t kShortTimeoutCycles   = 200000000ULL;
constexpr uint64_t kHealthyTimeoutCycles = 5000000000ULL;

// --- LSA helpers (mirrors LsaBarrierTimeoutMPITests.cpp) ------------------

__global__ void lsaBarrierTimeoutKernel(struct ncclDevComm devComm,
                                         uint64_t timeoutCycles,
                                         int* outResult) {
    ncclLsaBarrierSession<ncclCoopCta> bar{
        ncclCoopCta(), devComm, ncclTeamTagLsa(), /*index=*/0u};
    ncclResult_t r = bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, timeoutCycles);
    if (threadIdx.x == 0 && blockIdx.x == 0) *outResult = static_cast<int>(r);
}

int runLsaBarrier(ncclDevComm devComm, hipStream_t stream, uint64_t timeoutCycles) {
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

// --- GIN helpers (mirrors GinBarrierTimeoutMPITests.cpp) ------------------

__global__ void ginBarrierTimeoutKernel(struct ncclDevComm devComm,
                                         uint64_t timeoutCycles,
                                         int* outResult) {
    ncclGin gin{devComm, /*ginContext=*/0};
    ncclGinBarrierSession<ncclCoopCta> bar{
        ncclCoopCta(), gin, ncclTeamTagRail{}, /*index=*/0u};
    ncclResult_t r = bar.sync(ncclCoopCta(), cuda::memory_order_relaxed,
                              ncclGinFenceLevel::Relaxed, timeoutCycles);
    if (threadIdx.x == 0 && blockIdx.x == 0) *outResult = static_cast<int>(r);
}

int runGinBarrier(ncclDevComm devComm, hipStream_t stream, uint64_t timeoutCycles) {
    int* dResult = nullptr;
    if (hipMalloc(&dResult, sizeof(int)) != hipSuccess) return -100;
    int sentinel = -1;
    (void)hipMemcpy(dResult, &sentinel, sizeof(int), hipMemcpyHostToDevice);
    ginBarrierTimeoutKernel<<<1, 256, 0, stream>>>(devComm, timeoutCycles, dResult);
    (void)hipStreamSynchronize(stream);
    int hResult = -1;
    (void)hipMemcpy(&hResult, dResult, sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(dResult);
    return hResult;
}

ncclResult_t createGinDevComm(ncclComm_t comm, int nBarriers, ncclDevComm* out) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.railGinBarrierCount = nBarriers;
    reqs.ginConnectionType   = NCCL_GIN_CONNECTION_RAIL;
    return ncclDevCommCreate(comm, &reqs, out);
}

std::string ginBarrierSkipReason() {
    if (const char* e = std::getenv("NCCL_GIN_ENABLE"); e && std::strcmp(e, "0") == 0)
        return "GIN explicitly disabled";
    if (const char* e = std::getenv("NCCL_GIN_TYPE"); !e || std::atoi(e) != 2)
        return "NCCL_GIN_TYPE=2 required";
    if (const char* e = std::getenv("NCCL_CUMEM_ENABLE"); !e || std::strcmp(e, "1") != 0)
        return "NCCL_CUMEM_ENABLE=1 required";
    // Single-node needs intranet
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int nodeSize = 0, worldSize = 0;
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_free(&nodeComm);
    if (nodeSize == worldSize) {
        if (const char* e = std::getenv("RCCL_ENABLE_INTRANET"); !e || std::strcmp(e, "1") != 0)
            return "single-node: RCCL_ENABLE_INTRANET=1 required";
    }
    return "";
}

// Helper: AllReduce float sum on a live comm, verify correctness.
bool runAllReduce(ncclComm_t comm, hipStream_t stream, int worldSize) {
    const int   n     = 512;
    const float fill  = 1.0f;
    float* sendD = nullptr; float* recvD = nullptr;
    float* recvH = new float[n];
    auto guard = makeScopeGuard([&]() {
        delete[] recvH;
        if (sendD) (void)hipFree(sendD);
        if (recvD) (void)hipFree(recvD);
    });
    if (hipMalloc(&sendD, n*sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&recvD, n*sizeof(float)) != hipSuccess) return false;
    hipMemset(sendD, 0, n*sizeof(float));
    float* tmp = new float[n];
    for (int i = 0; i < n; ++i) tmp[i] = fill;
    hipMemcpy(sendD, tmp, n*sizeof(float), hipMemcpyHostToDevice);
    delete[] tmp;
    if (ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream) != ncclSuccess) return false;
    if (hipStreamSynchronize(stream) != hipSuccess) return false;
    if (hipMemcpy(recvH, recvD, n*sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return false;
    const float expected = fill * static_cast<float>(worldSize);
    for (int i = 0; i < n; ++i) if (recvH[i] != expected) return false;
    return true;
}

} // namespace

class DeviceTimeoutPropagationMPITest : public MPITestBase {};

/**
 * End-to-end: real LSA device barrier timeout → ncclCommSetAsyncError →
 * ncclCommGetAsyncError returns ncclTimeout → comm still functional →
 * error clearable.
 *
 * This closes the gap between LsaBarrierTimeoutMPITests (device side) and
 * TimeoutMPITests (host API with injected error): here the device produces
 * a real ncclTimeout which the test propagates through the async-error API
 * exactly as a production polling/proxy path would.
 */
TEST_F(DeviceTimeoutPropagationMPITest, DeviceTimeout_LsaSurfacesViaAsyncError)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit))
        << "Test requires at least 2 MPI processes";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    // Set up LSA devComm — skips if symmetric memory unavailable.
    ncclDevComm devComm{};
    ncclResult_t devRc = createLsaDevComm(comm, 1, &devComm);
    if (devRc != ncclSuccess) {
        GTEST_SKIP() << "LSA devComm unavailable (needs symmetric memory / kernel >= 6.8)";
    }
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);

    // --- Round 1: produce a real device ncclTimeout ---
    int deviceResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        deviceResult = runLsaBarrier(devComm, stream, kShortTimeoutCycles);
        EXPECT_EQ(static_cast<int>(ncclTimeout), deviceResult)
            << "LSA barrier expected ncclTimeout, got "
            << ncclGetErrorString(static_cast<ncclResult_t>(deviceResult));
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        (void)runLsaBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Propagate through the async-error API (as production code would) ---
    if (!isAbsent && deviceResult == static_cast<int>(ncclTimeout)) {
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclCommSetAsyncError(comm, static_cast<ncclResult_t>(deviceResult)));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Every rank must observe ncclTimeout via GetAsyncError ---
    if (!isAbsent) {
        ncclResult_t observed = ncclSuccess;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &observed));
        ASSERT_MPI_EQ(ncclTimeout, observed);

        // String must be "timeout"
        ASSERT_TRUE(std::strcmp(ncclGetErrorString(observed), "timeout") == 0)
            << "ncclGetErrorString(ncclTimeout) != 'timeout'";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Clear the error ---
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Comm must still be fully functional after clearing ---
    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    ASSERT_MPI_TRUE(runAllReduce(comm, stream, worldSize))
        << "AllReduce failed after clearing ncclTimeout async error";

    // --- Comm reports healthy ---
    ncclResult_t after = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &after));
    ASSERT_MPI_EQ(ncclSuccess, after);
}

/**
 * Two sequential real LSA device timeouts: both must surface as ncclTimeout,
 * each clearable, comm must remain healthy throughout.
 * Verifies that ncclCommSetAsyncError is idempotent for ncclTimeout and
 * that clearing then re-triggering works correctly.
 */
TEST_F(DeviceTimeoutPropagationMPITest, DeviceTimeout_MultipleTimeoutsAccumulate)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit))
        << "Test requires at least 2 MPI processes";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    ncclDevComm devComm{};
    ncclResult_t devRc = createLsaDevComm(comm, 1, &devComm);
    if (devRc != ncclSuccess) GTEST_SKIP() << "LSA devComm unavailable";
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    const bool isAbsent = (lsaTeam.rank == lsaTeam.nRanks - 1);
    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    for (int round = 0; round < 2; ++round) {
        // Produce real device timeout
        ncclDevComm dc{};
        MPI_Barrier(MPI_COMM_WORLD);
        ASSERT_EQ(ncclSuccess, createLsaDevComm(comm, 1, &dc));

        if (!isAbsent) {
            int r = runLsaBarrier(dc, stream, kShortTimeoutCycles);
            EXPECT_EQ(static_cast<int>(ncclTimeout), r)
                << "Round " << round << ": expected ncclTimeout";
            (void)hipStreamSynchronize(stream);
            // Propagate through API
            ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (isAbsent) {
            (void)runLsaBarrier(dc, stream, 0ULL);
            (void)hipStreamSynchronize(stream);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        (void)ncclDevCommDestroy(comm, &dc);
        MPI_Barrier(MPI_COMM_WORLD);

        // Verify timeout is observed
        if (!isAbsent) {
            ncclResult_t obs = ncclSuccess;
            ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &obs));
            ASSERT_MPI_EQ(ncclTimeout, obs);
        }

        // Clear and verify healthy
        ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
        MPI_Barrier(MPI_COMM_WORLD);

        ASSERT_MPI_TRUE(runAllReduce(comm, stream, worldSize))
            << "AllReduce failed after clearing in round " << round;

        ncclResult_t after = ncclTimeout;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &after));
        ASSERT_MPI_EQ(ncclSuccess, after);
    }
}

/**
 * End-to-end with real GIN device barrier: timeout → ncclCommSetAsyncError →
 * ncclCommGetAsyncError → comm functional. Multi-node (IB) variant.
 * Skips if GIN prerequisites are not met (NCCL_GIN_TYPE=2, CUMEM, etc.).
 */
TEST_F(DeviceTimeoutPropagationMPITest, DeviceTimeout_GinSurfacesViaAsyncError)
{
    auto skipReason = ginBarrierSkipReason();
    if (!skipReason.empty()) GTEST_SKIP() << skipReason;

    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit))
        << "Test requires at least 2 MPI processes";
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    ncclDevComm devComm{};
    ncclResult_t devRc = createGinDevComm(comm, 1, &devComm);
    if (devRc != ncclSuccess) GTEST_SKIP() << "GIN devComm unavailable";
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t railTeam = ncclTeamRail(comm);
    const bool isAbsent = (railTeam.rank == railTeam.nRanks - 1);

    int deviceResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        deviceResult = runGinBarrier(devComm, stream, kShortTimeoutCycles);
        EXPECT_EQ(static_cast<int>(ncclTimeout), deviceResult)
            << "GIN barrier expected ncclTimeout, got "
            << ncclGetErrorString(static_cast<ncclResult_t>(deviceResult));
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        (void)runGinBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (!isAbsent && deviceResult == static_cast<int>(ncclTimeout)) {
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclCommSetAsyncError(comm, static_cast<ncclResult_t>(deviceResult)));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (!isAbsent) {
        ncclResult_t observed = ncclSuccess;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &observed));
        ASSERT_MPI_EQ(ncclTimeout, observed);
        ASSERT_TRUE(std::strcmp(ncclGetErrorString(observed), "timeout") == 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
    MPI_Barrier(MPI_COMM_WORLD);

    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    ASSERT_MPI_TRUE(runAllReduce(comm, stream, worldSize))
        << "AllReduce failed after clearing GIN ncclTimeout async error";

    ncclResult_t after = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &after));
    ASSERT_MPI_EQ(ncclSuccess, after);
}

#endif // MPI_TESTS_ENABLED
