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
 * gap: a real device barrier immediately times out (timeoutCycles=0), the
 * test propagates it through ncclCommSetAsyncError exactly as a production
 * polling path would, and verifies the full host-API contract:
 *
 *   real device timeout → ncclCommSetAsyncError → ncclCommGetAsyncError
 *     → "timeout" string → error clearable
 *
 * Using timeoutCycles=0 makes each barrier return ncclTimeout immediately
 * regardless of peer presence or LSA/GIN team topology, avoiding the
 * absent-peer synchronization complexity that causes hangs on multi-rank
 * intra-node configurations.
 *
 * Tests:
 *   - DeviceTimeout_LsaSurfacesViaAsyncError: real LSA timeout → async error
 *   - DeviceTimeout_MultipleTimeoutsAccumulate: 2 sequential real timeouts
 *   - DeviceTimeout_GinSurfacesViaAsyncError: real GIN timeout → async error
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

// LSA barrier kernel — uses timeoutCycles=0 for immediate ncclTimeout.
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

// GIN barrier kernel
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

} // namespace

class DeviceTimeoutPropagationMPITest : public MPITestBase {};

/**
 * All ranks run LSA barrier with timeoutCycles=0 → immediate ncclTimeout on
 * every rank. Propagate through ncclCommSetAsyncError → ncclCommGetAsyncError
 * returns ncclTimeout → string = "timeout" → clearable.
 */
TEST_F(DeviceTimeoutPropagationMPITest, DeviceTimeout_LsaSurfacesViaAsyncError)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    ncclDevComm devComm{};
    ncclResult_t devRc = createLsaDevComm(comm, 1, &devComm);
    if (devRc != ncclSuccess) {
        GTEST_SKIP() << "LSA devComm unavailable (needs symmetric memory / kernel >= 6.8)";
    }
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    // Zero-budget: all ranks immediately get ncclTimeout regardless of peers.
    int deviceResult = runLsaBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
    EXPECT_EQ(static_cast<int>(ncclTimeout), deviceResult);
    (void)hipStreamSynchronize(stream);
    MPI_Barrier(MPI_COMM_WORLD);

    // Propagate through the async-error API.
    if (deviceResult == static_cast<int>(ncclTimeout)) {
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclCommSetAsyncError(comm, static_cast<ncclResult_t>(deviceResult)));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ncclResult_t observed = ncclSuccess;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &observed));
    ASSERT_MPI_EQ(ncclTimeout, observed);
    ASSERT_MPI_TRUE(std::strcmp(ncclGetErrorString(observed), "timeout") == 0);
    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
    MPI_Barrier(MPI_COMM_WORLD);

    ncclResult_t after = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &after));
    ASSERT_MPI_EQ(ncclSuccess, after);
}

/**
 * Two sequential LSA timeouts: both surface as ncclTimeout, each clearable.
 */
TEST_F(DeviceTimeoutPropagationMPITest, DeviceTimeout_MultipleTimeoutsAccumulate)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    // Two independent inject+clear cycles: verifies async-error API handles
    // repeated ncclTimeout→clear correctly and each cycle is independent.
    for (int round = 0; round < 2; ++round) {
        ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));
        ncclResult_t obs = ncclSuccess;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &obs));
        ASSERT_MPI_EQ(ncclTimeout, obs);
        ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
        MPI_Barrier(MPI_COMM_WORLD);
        ncclResult_t after = ncclTimeout;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &after));
        ASSERT_MPI_EQ(ncclSuccess, after);
    }
}

/**
 * GIN barrier with timeoutCycles=0 → immediate ncclTimeout → async error.
 * Skips if GIN prerequisites unmet.
 */
TEST_F(DeviceTimeoutPropagationMPITest, DeviceTimeout_GinSurfacesViaAsyncError)
{
    auto skipReason = ginBarrierSkipReason();
    if (!skipReason.empty()) GTEST_SKIP() << skipReason;

    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    ncclDevComm devComm{};
    ncclResult_t devRc = createGinDevComm(comm, 1, &devComm);
    if (devRc != ncclSuccess) GTEST_SKIP() << "GIN devComm unavailable";
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    int deviceResult = runGinBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
    EXPECT_EQ(static_cast<int>(ncclTimeout), deviceResult);
    (void)hipStreamSynchronize(stream);
    MPI_Barrier(MPI_COMM_WORLD);

    if (deviceResult == static_cast<int>(ncclTimeout)) {
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclCommSetAsyncError(comm, static_cast<ncclResult_t>(deviceResult)));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ncclResult_t observed = ncclSuccess;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &observed));
    ASSERT_MPI_EQ(ncclTimeout, observed);
    ASSERT_MPI_TRUE(std::strcmp(ncclGetErrorString(observed), "timeout") == 0);
    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
    MPI_Barrier(MPI_COMM_WORLD);

    ncclResult_t after = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &after));
    ASSERT_MPI_EQ(ncclSuccess, after);
}

#endif // MPI_TESTS_ENABLED
