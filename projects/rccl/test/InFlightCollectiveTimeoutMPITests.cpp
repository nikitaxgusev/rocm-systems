/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file InFlightCollectiveTimeoutMPITests.cpp
 * @brief Coverage for ncclTimeout lifecycle around real collective ops (AICOMNET-193).
 *
 * Gap filled: all existing tests either inject ncclTimeout with no real op
 * running, or use standalone device barrier kernels. This file verifies the
 * async-error API behaves correctly relative to real RCCL collectives:
 *
 *   1. ncclTimeout injected before/after an AllReduce does not corrupt results
 *   2. The error is observable via GetAsyncError between ops and clearable
 *   3. A subsequent AllReduce succeeds after clearing the error
 *   4. Multiple inject+clear cycles leave the comm clean
 *
 * Uses the standard blocking comm (createTestCommunicator) to avoid the
 * non-blocking comm init latency. GetAsyncError between ops on a blocking
 * comm is the correct usage pattern.
 *
 * Tests:
 *   - InFlight_AsyncErrorObservableDuringCollective
 *   - InFlight_TimeoutDoesNotCorruptCollectiveResult
 *   - InFlight_ClearableBeforeCompletion
 *   - InFlight_BlockingComm_TimeoutRoundTrip
 */

#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include "nccl.h"

#include <cstring>
#include <mpi.h>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;

// Internal RCCL symbol — forward declaration (see TimeoutMPITests.cpp).
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState);

namespace {

bool allocFill(float** ptr, int n, float val) {
    if (hipMalloc(ptr, n * sizeof(float)) != hipSuccess) return false;
    float* h = new float[n];
    for (int i = 0; i < n; ++i) h[i] = val;
    bool ok = (hipMemcpy(*ptr, h, n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    delete[] h;
    return ok;
}

bool checkResult(float* dBuf, int n, float expected) {
    float* h = new float[n];
    bool ok = (hipMemcpy(h, dBuf, n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
    if (ok) for (int i = 0; i < n; ++i) if (h[i] != expected) { ok = false; break; }
    delete[] h;
    return ok;
}

} // namespace

class InFlightCollectiveTimeoutMPITest : public MPITestBase {};

/**
 * Inject ncclTimeout after enqueuing an AllReduce, verify it is observable
 * via GetAsyncError after the stream completes, then clear and re-run.
 */
TEST_F(InFlightCollectiveTimeoutMPITest, InFlight_AsyncErrorObservableDuringCollective)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const int n = 64 * 1024;
    float* sendD = nullptr; float* recvD = nullptr;
    ASSERT_TRUE(allocFill(&sendD, n, 1.0f));
    ASSERT_TRUE(allocFill(&recvD, n, 0.0f));
    SCOPE_EXIT(if (sendD) (void)hipFree(sendD); if (recvD) (void)hipFree(recvD));

    // Enqueue AllReduce then immediately inject timeout.
    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));

    // Wait for op to complete.
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Verify error is visible.
    ncclResult_t observed = ncclSuccess;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &observed));
    ASSERT_MPI_EQ(ncclTimeout, observed);
    ASSERT_MPI_TRUE(std::strcmp(ncclGetErrorString(observed), "timeout") == 0);

    // AllReduce must have produced correct results.
    const float expected = 1.0f * static_cast<float>(worldSize);
    ASSERT_MPI_TRUE(checkResult(recvD, n, expected));

    // Clear and verify second AllReduce works.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));
    ASSERT_MPI_TRUE(checkResult(recvD, n, expected));

    ncclResult_t after = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &after));
    ASSERT_MPI_EQ(ncclSuccess, after);
}

/**
 * ncclTimeout is advisory — injecting it mid-flight must not corrupt
 * the AllReduce result. Data correctness is verified after stream sync.
 */
TEST_F(InFlightCollectiveTimeoutMPITest, InFlight_TimeoutDoesNotCorruptCollectiveResult)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const int n = 128 * 1024;
    const float fill = 2.0f;
    float* sendD = nullptr; float* recvD = nullptr;
    ASSERT_TRUE(allocFill(&sendD, n, fill));
    ASSERT_TRUE(allocFill(&recvD, n, 0.0f));
    SCOPE_EXIT(if (sendD) (void)hipFree(sendD); if (recvD) (void)hipFree(recvD));

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    const float expected = fill * static_cast<float>(worldSize);
    ASSERT_MPI_TRUE(checkResult(recvD, n, expected));

    ncclResult_t state = ncclSuccess;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &state));
    ASSERT_MPI_EQ(ncclTimeout, state);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
    MPI_Barrier(MPI_COMM_WORLD);

    ncclResult_t clean = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &clean));
    ASSERT_MPI_EQ(ncclSuccess, clean);
}

/**
 * Set then immediately clear before stream finishes: the clear must stick
 * and comm must be clean after completion.
 */
TEST_F(InFlightCollectiveTimeoutMPITest, InFlight_ClearableBeforeCompletion)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const int n = 256 * 1024;
    float* sendD = nullptr; float* recvD = nullptr;
    ASSERT_TRUE(allocFill(&sendD, n, 1.0f));
    ASSERT_TRUE(allocFill(&recvD, n, 0.0f));
    SCOPE_EXIT(if (sendD) (void)hipFree(sendD); if (recvD) (void)hipFree(recvD));

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));

    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ncclResult_t state = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &state));
    ASSERT_MPI_EQ(ncclSuccess, state);

    const float expected = 1.0f * static_cast<float>(worldSize);
    ASSERT_MPI_TRUE(checkResult(recvD, n, expected));
}

/**
 * Multiple set+get+clear cycles on a blocking comm between ops — each
 * cycle must be independent, comm healthy throughout.
 */
TEST_F(InFlightCollectiveTimeoutMPITest, InFlight_BlockingComm_TimeoutRoundTrip)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const int n = 1024;
    float* sendD = nullptr; float* recvD = nullptr;
    ASSERT_TRUE(allocFill(&sendD, n, 1.0f));
    ASSERT_TRUE(allocFill(&recvD, n, 0.0f));
    SCOPE_EXIT(if (sendD) (void)hipFree(sendD); if (recvD) (void)hipFree(recvD));

    for (int round = 0; round < 3; ++round) {
        ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));
        ncclResult_t obs = ncclSuccess;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &obs));
        ASSERT_MPI_EQ(ncclTimeout, obs);
        ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));

        ASSERT_MPI_EQ(ncclSuccess,
            ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

        const float expected = 1.0f * static_cast<float>(worldSize);
        ASSERT_MPI_TRUE(checkResult(recvD, n, expected));

        ncclResult_t after = ncclTimeout;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &after));
        ASSERT_MPI_EQ(ncclSuccess, after);

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

#endif // MPI_TESTS_ENABLED
