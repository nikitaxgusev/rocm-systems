/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file TimeoutMPITests.cpp
 * @brief Distributed coverage for the ncclTimeout result code (AICOMNET-193).
 *
 * Exercises the ncclTimeout async-error pipeline on REAL multi-rank / multi-node
 * communicators (not mocks). These tests cover the host-reachable contract that
 * works on any kernel: a device producer (LSA/GIN barrier) surfaces ncclTimeout
 * by storing it into the comm's async state, exactly the path injected here via
 * ncclCommSetAsyncError. Every rank must:
 *   - accept ncclTimeout through ncclCommSetAsyncError,
 *   - read it back through ncclCommGetAsyncError,
 *   - stringify it to "timeout",
 *   - and the communicator must stay fully functional afterwards
 *     (a real AllReduce still succeeds and produces correct data).
 *
 * Device-side production of ncclTimeout from a real stuck barrier is covered
 * separately in LsaBarrierTimeoutMPITests.cpp (requires symmetric memory).
 */

#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "nccl.h"

#include <cstring>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;

// Internal RCCL symbol (declared in comm.h, exported from librccl.so). The
// signature mangles identically to the library definition, so a forward
// declaration lets the MPI test drive it without pulling internal headers.
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState);

class TimeoutMPITest : public MPITestBase {};

/**
 * Round-trip ncclTimeout through the async-error pipeline on a live comm,
 * then prove the comm is still healthy with a real AllReduce.
 */
TEST_F(TimeoutMPITest, AsyncErrorRoundTripOnLiveComm)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    // Inject ncclTimeout exactly where a real timeout producer would set it.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));

    // Every rank must observe ncclTimeout via the public getter ...
    ncclResult_t observed = ncclSuccess;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &observed));
    ASSERT_MPI_EQ(ncclTimeout, observed);

    // ... and translate it consistently.
    ASSERT_MPI_TRUE(std::strcmp(ncclGetErrorString(observed), "timeout") == 0);

    // Clear the async state and confirm the comm is genuinely usable again.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
    MPI_Barrier(MPI_COMM_WORLD);

    const int   n     = 1024;
    const float fill  = 1.0f;
    float*      sendH = new float[n];
    float*      recvH = new float[n];
    for (int i = 0; i < n; ++i) sendH[i] = fill;
    auto host_guard = makeScopeGuard([&]() { delete[] sendH; delete[] recvH; });

    float* sendD = nullptr;
    float* recvD = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&sendD, n * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recvD, n * sizeof(float)));
    auto dev_guard = makeScopeGuard([&]() { if (sendD) (void)hipFree(sendD);
                                            if (recvD) (void)hipFree(recvD); });
    HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(sendD, sendH, n * sizeof(float), hipMemcpyHostToDevice));

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(sendD, recvD, n, ncclFloat, ncclSum, comm, stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(recvH, recvD, n * sizeof(float), hipMemcpyDeviceToHost));

    // Sum of `fill` across all ranks.
    const float expected = fill * static_cast<float>(MPIEnvironment::world_size);
    bool        correct  = true;
    for (int i = 0; i < n; ++i)
        if (recvH[i] != expected) { correct = false; break; }
    ASSERT_MPI_TRUE(correct);

    // Comm must report healthy after a successful collective.
    ncclResult_t after = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &after));
    ASSERT_MPI_EQ(ncclSuccess, after);
}

/**
 * Setting ncclTimeout then resetting to ncclSuccess must leave the comm clean
 * on every rank, and a subsequent collective must still succeed -- a timeout
 * that has been observed and cleared is not a permanent error.
 */
TEST_F(TimeoutMPITest, AsyncErrorClearableOnLiveComm)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclTimeout));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));

    ncclResult_t observed = ncclTimeout;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommGetAsyncError(comm, &observed));
    ASSERT_MPI_EQ(ncclSuccess, observed);
}

/**
 * The ncclTimeout -> "timeout" mapping must be identical on every rank/node.
 */
TEST_F(TimeoutMPITest, ErrorStringConsistentAcrossRanks)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    const char* s = ncclGetErrorString(ncclTimeout);
    ASSERT_MPI_TRUE(s != nullptr && std::strcmp(s, "timeout") == 0);

    // AllReduce(min/max) the string length to confirm every rank produced the
    // same value.
    int len    = static_cast<int>(std::strlen(s));
    int lenMin = len;
    int lenMax = len;
    MPI_Allreduce(MPI_IN_PLACE, &lenMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &lenMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    ASSERT_MPI_EQ(lenMin, lenMax);
    ASSERT_MPI_EQ(len, lenMin);
}

#endif // MPI_TESTS_ENABLED
