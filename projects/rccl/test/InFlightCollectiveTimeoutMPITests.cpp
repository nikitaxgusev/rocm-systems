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
 * non-blocking comm init latency. GetAsyncError between ops is the correct
 * usage pattern on blocking comms.
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

// Allocate n floats on device, fill with val.
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
 * Core test: inject ncclTimeout while AllReduce is in-flight on non-blocking
 * comm, verify it is observable via GetAsyncError, verify AllReduce still
 * produces correct results after it drains, verify clearable, verify
 * subsequent AllReduce works.
 */
TEST_F(InFlightCollectiveTimeoutMPITest, InFlight_AsyncErrorObservableDuringCollective)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const int n = 64 * 1024;
    float* sendD = nullptr; float* recvD = nullptr;
    ASSERT_TRUE(allocFill(&sendD, n, 1.0f));
    ASSERT_TRUE(allocFill(&recvD, n, 0.0f));
    SCOPE_EXIT(if (sendD) (void)hipFree(sendD); if (recvD) (void)hipFree(recvD));

    // Enqueue AllReduce on non-blocking comm — returns immediately.
    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));

    // Inject ncclTimeout while the op may still be in-flight.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));

    // GetAsyncError must return ncclTimeout (may need to yield briefly for
    // the store to propagate, but typically immediate).
    ncclResult_t observed = ncclSuccess;
    // Poll up to 1000 iterations to tolerate any memory-order delay.
    for (int i = 0; i < 1000; ++i) {
        ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &observed));
        if (observed == ncclTimeout) break;
        std::this_thread::yield();
    }
    ASSERT_MPI_EQ(ncclTimeout, observed);
    ASSERT_TRUE(std::strcmp(ncclGetErrorString(observed), "timeout") == 0);

    // Wait for the AllReduce to complete on GPU regardless of async error.
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    // AllReduce must have produced correct results despite async error.
    const float expected = 1.0f * static_cast<float>(worldSize);
    ASSERT_MPI_TRUE(checkResult(recvD, n, expected));

    // Clear the error.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
    MPI_Barrier(MPI_COMM_WORLD);

    // Subsequent AllReduce must succeed.
    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_MPI_TRUE(checkResult(recvD, n, expected));

    // Comm is clean.
    ncclResult_t after = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &after));
    ASSERT_MPI_EQ(ncclSuccess, after);
}

/**
 * Verify AllReduce result is correct even when ncclTimeout is injected
 * mid-flight: the async error is advisory, not a comm abort.
 * More explicit than the test above — checks correctness first, then
 * error state.
 */
TEST_F(InFlightCollectiveTimeoutMPITest, InFlight_TimeoutDoesNotCorruptCollectiveResult)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    int rank = 0, worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    ncclUniqueId id{};
    if (rank == 0) ASSERT_MPI_EQ(ncclSuccess, ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, createNonBlockingComm(worldSize, id, rank, &comm));

    const int n = 128 * 1024;
    const float fill = 2.0f;
    float* sendD = nullptr; float* recvD = nullptr;
    ASSERT_TRUE(allocFill(&sendD, n, fill));
    ASSERT_TRUE(allocFill(&recvD, n, 0.0f));
    SCOPE_EXIT(if (sendD) (void)hipFree(sendD); if (recvD) (void)hipFree(recvD));

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));

    // Inject timeout mid-flight.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));

    // Wait for GPU op to complete.
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Result must be correct: timeout is advisory, not a cancel.
    const float expected = fill * static_cast<float>(worldSize);
    ASSERT_MPI_TRUE(checkResult(recvD, n, expected));

    // Error is still set — verify then clear.
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
 * Setting ncclTimeout while an op is in-flight and then CLEARING it before
 * the op completes: the comm must report clean after the op finishes.
 * Tests that clearing is not a race that gets re-set by the completing op.
 */
TEST_F(InFlightCollectiveTimeoutMPITest, InFlight_ClearableBeforeCompletion)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();
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

    // Set then immediately clear — before op completes.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));

    // Wait for op.
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Comm must be clean — clearing before completion must stick.
    ncclResult_t state = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &state));
    ASSERT_MPI_EQ(ncclSuccess, state);

    // AllReduce result correctness.
    const float expected = 1.0f * static_cast<float>(worldSize);
    ASSERT_MPI_TRUE(checkResult(recvD, n, expected));
}

/**
 * Negative test: verify that ncclCommSetAsyncError does NOT reject ncclTimeout
 * on a blocking comm — the function itself is not comm-type-gated.
 * However, GetAsyncError on a blocking comm is only meaningful between ops,
 * not during — so we verify the round-trip without an active op.
 */
TEST_F(InFlightCollectiveTimeoutMPITest, InFlight_BlockingComm_TimeoutRoundTrip)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit,
                                          kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    // On blocking comm (default), set+get+clear must all work between ops.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));

    ncclResult_t observed = ncclSuccess;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &observed));
    ASSERT_MPI_EQ(ncclTimeout, observed);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));

    ncclResult_t clean = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &clean));
    ASSERT_MPI_EQ(ncclSuccess, clean);

    // A real collective still works after the error cycle.
    hipStream_t stream = getActiveStream();
    const int n = 1024;
    float* sendD = nullptr; float* recvD = nullptr;
    ASSERT_TRUE(allocFill(&sendD, n, 1.0f));
    ASSERT_TRUE(allocFill(&recvD, n, 0.0f));
    SCOPE_EXIT(if (sendD) (void)hipFree(sendD); if (recvD) (void)hipFree(recvD));

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    ASSERT_MPI_TRUE(checkResult(recvD, n, static_cast<float>(worldSize)));
}

#endif // MPI_TESTS_ENABLED
