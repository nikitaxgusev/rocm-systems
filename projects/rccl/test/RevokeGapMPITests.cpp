/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file RevokeGapMPITests.cpp
 *
 * Gap coverage between TorchComms nccl/ncclx tests and RCCL tests.
 * Focuses on three areas not covered by RevokeMPITests, RevokeStressMPITests,
 * RevokeShrinkDeepStressMPITests, or RevokeShrinkAdvancedMPITests:
 *
 *   1. Revoke rejects ALL collective types (not just AllReduce)
 *   2. All collective types work on a child comm after shrink
 *   3. Shrink to size 1 (single-rank communicator)
 */

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "nccl.h"

#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

class RevokeGapMPITest : public MPITestBase {};

static void computeSymmetricExclude(int worldRank, int worldSize,
                                    std::vector<int>& excludeList, bool& isExcluded)
{
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localSize, localRank;
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int numNodes = worldSize / localSize;
    isExcluded = (localRank == localSize - 1);
    excludeList.clear();
    for (int n = 0; n < numNodes; n++)
        excludeList.push_back((n + 1) * localSize - 1);
}

// ============================================================================
// 1. Revoke rejects ALL collective types, not just AllReduce.
//    TorchComms tests every op type after revoke; existing RCCL tests only
//    check ncclAllReduce. This covers: AllReduce, Broadcast, Reduce,
//    AllGather, ReduceScatter, Send, Recv, Gather, Scatter, AlltoAll.
// ============================================================================

TEST_F(RevokeGapMPITest, Revoke_RejectsAllCollectiveTypes)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;
    int         wsize  = MPIEnvironment::world_size;

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT));
    MPI_Barrier(MPI_COMM_WORLD);

    constexpr size_t kCount = 1024;
    size_t totalBufSize = kCount * wsize * sizeof(double);

    void* buf1 = nullptr;
    void* buf2 = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf1, totalBufSize));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf2, totalBufSize));
    auto g1 = makeScopeGuard([&]() { if(buf1) (void)hipFree(buf1); });
    auto g2 = makeScopeGuard([&]() { if(buf2) (void)hipFree(buf2); });

    int peer = (rank + 1) % wsize;

    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclAllReduce(buf1, buf2, kCount, ncclFloat, ncclSum, comm, stream));
    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclBroadcast(buf1, buf2, kCount, ncclFloat, 0, comm, stream));
    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclReduce(buf1, buf2, kCount, ncclFloat, ncclSum, 0, comm, stream));
    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclAllGather(buf1, buf2, kCount, ncclFloat, comm, stream));
    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclReduceScatter(buf1, buf2, kCount, ncclFloat, ncclSum, comm, stream));
    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclSend(buf1, kCount, ncclFloat, peer, comm, stream));
    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclRecv(buf2, kCount, ncclFloat, peer, comm, stream));
    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclGather(buf1, buf2, kCount, ncclFloat, 0, comm, stream));
    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclScatter(buf1, buf2, kCount, ncclFloat, 0, comm, stream));
    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclAlltoAll(buf1, buf2, kCount, ncclFloat, comm, stream));
}

// Stress variant: sweep 6 sizes across all collective types after revoke
TEST_F(RevokeGapMPITest, Revoke_RejectsAllCollectives_MultipleSizes)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;
    int         wsize  = MPIEnvironment::world_size;

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT));
    MPI_Barrier(MPI_COMM_WORLD);

    const size_t sizes[] = {1, 64, 1024, 65536, 1048576, 4194304};
    constexpr size_t kMaxCount = 4194304;
    size_t totalBufSize = kMaxCount * wsize * sizeof(float);

    void* buf1 = nullptr;
    void* buf2 = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf1, totalBufSize));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf2, totalBufSize));
    auto g1 = makeScopeGuard([&]() { if(buf1) (void)hipFree(buf1); });
    auto g2 = makeScopeGuard([&]() { if(buf2) (void)hipFree(buf2); });

    int peer = (rank + 1) % wsize;

    for (size_t count : sizes) {
        ASSERT_MPI_EQ(ncclInvalidUsage,
                      ncclAllReduce(buf1, buf2, count, ncclFloat, ncclSum, comm, stream));
        ASSERT_MPI_EQ(ncclInvalidUsage,
                      ncclBroadcast(buf1, buf2, count, ncclFloat, 0, comm, stream));
        ASSERT_MPI_EQ(ncclInvalidUsage,
                      ncclReduce(buf1, buf2, count, ncclFloat, ncclSum, 0, comm, stream));
        ASSERT_MPI_EQ(ncclInvalidUsage,
                      ncclAllGather(buf1, buf2, count, ncclFloat, comm, stream));
        ASSERT_MPI_EQ(ncclInvalidUsage,
                      ncclReduceScatter(buf1, buf2, count, ncclFloat, ncclSum, comm, stream));
        ASSERT_MPI_EQ(ncclInvalidUsage,
                      ncclSend(buf1, count, ncclFloat, peer, comm, stream));
        ASSERT_MPI_EQ(ncclInvalidUsage,
                      ncclRecv(buf2, count, ncclFloat, peer, comm, stream));
        ASSERT_MPI_EQ(ncclInvalidUsage,
                      ncclGather(buf1, buf2, count, ncclFloat, 0, comm, stream));
        ASSERT_MPI_EQ(ncclInvalidUsage,
                      ncclScatter(buf1, buf2, count, ncclFloat, 0, comm, stream));
        ASSERT_MPI_EQ(ncclInvalidUsage,
                      ncclAlltoAll(buf1, buf2, count, ncclFloat, comm, stream));
    }
}

// ============================================================================
// 2. All collective types work on child comm after revoke+shrink.
//    Existing tests only run AllReduce (sometimes Broadcast) on the child.
//    TorchComms test_shrink_multiple_collectives runs allreduce+broadcast.
//    This covers ALL collective types on the child, 5 iterations each.
// ============================================================================

TEST_F(RevokeGapMPITest, RevokeShrink_AllCollectiveTypes_OnChild)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool isExcluded;
    computeSymmetricExclude(rank, MPIEnvironment::world_size, excludeList, isExcluded);
    ncclComm_t child = NCCL_COMM_NULL;

    if (!isExcluded) {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (!isExcluded) {
        auto child_guard = makeCommAutoGuard(child);

        int childRank = -1, childSize = 0;
        ASSERT_EQ(ncclSuccess, ncclCommUserRank(child, &childRank));
        ASSERT_EQ(ncclSuccess, ncclCommCount(child, &childSize));

        constexpr size_t kCount = 1024 * 1024;
        size_t totalBufSize = kCount * childSize * sizeof(float);

        void* buf1 = nullptr;
        void* buf2 = nullptr;
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf1, totalBufSize));
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf2, totalBufSize));
        auto g1 = makeScopeGuard([&]() { if(buf1) (void)hipFree(buf1); });
        auto g2 = makeScopeGuard([&]() { if(buf2) (void)hipFree(buf2); });

        for (int iter = 0; iter < 5; iter++) {
            HIP_TEST_CHECK_GTEST_FAIL(hipMemset(buf1, 0, totalBufSize));
            HIP_TEST_CHECK_GTEST_FAIL(hipMemset(buf2, 0, totalBufSize));

            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(buf1, buf2, kCount, ncclFloat, ncclSum, child, stream));

            int root = iter % childSize;
            ASSERT_EQ(ncclSuccess,
                      ncclBroadcast(buf1, buf2, kCount, ncclFloat, root, child, stream));

            ASSERT_EQ(ncclSuccess,
                      ncclReduce(buf1, buf2, kCount, ncclFloat, ncclSum, 0, child, stream));

            ASSERT_EQ(ncclSuccess,
                      ncclAllGather(buf1, buf2, kCount, ncclFloat, child, stream));

            ASSERT_EQ(ncclSuccess,
                      ncclReduceScatter(buf1, buf2, kCount, ncclFloat, ncclSum, child, stream));

            ASSERT_EQ(ncclSuccess,
                      ncclGather(buf1, buf2, kCount, ncclFloat, 0, child, stream));

            ASSERT_EQ(ncclSuccess,
                      ncclScatter(buf1, buf2, kCount, ncclFloat, 0, child, stream));

            ASSERT_EQ(ncclSuccess,
                      ncclAlltoAll(buf1, buf2, kCount, ncclFloat, child, stream));

            // P2P ring
            int sendPeer = (childRank + 1) % childSize;
            int recvPeer = (childRank - 1 + childSize) % childSize;
            ASSERT_EQ(ncclSuccess, ncclGroupStart());
            ASSERT_EQ(ncclSuccess,
                      ncclSend(buf1, kCount, ncclFloat, sendPeer, child, stream));
            ASSERT_EQ(ncclSuccess,
                      ncclRecv(buf2, kCount, ncclFloat, recvPeer, child, stream));
            ASSERT_EQ(ncclSuccess, ncclGroupEnd());

            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================================
// 3. Shrink to size 1 (single-rank communicator).
//    Edge case: exclude all ranks except rank 0. Verify self-collectives work.
// ============================================================================

TEST_F(RevokeGapMPITest, RevokeShrink_ToSingleRank)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;
    int         wsize  = MPIEnvironment::world_size;

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    for (int r = 1; r < wsize; r++)
        excludeList.push_back(r);

    bool isExcluded = (rank != 0);
    ncclComm_t child = NCCL_COMM_NULL;

    if (!isExcluded) {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (!isExcluded) {
        auto child_guard = makeCommAutoGuard(child);

        int childRank = -1, childSize = 0;
        ASSERT_EQ(ncclSuccess, ncclCommUserRank(child, &childRank));
        ASSERT_EQ(ncclSuccess, ncclCommCount(child, &childSize));

        ASSERT_EQ(childSize, 1);
        ASSERT_EQ(childRank, 0);

        constexpr size_t kCount = 1024 * 1024;
        void* send_buf = nullptr;
        void* recv_buf = nullptr;
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
        auto sg = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
        auto rg = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

        // Self-allreduce (10 iterations at 1M floats)
        for (int iter = 0; iter < 10; iter++) {
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, child, stream));
            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }

        // Self-broadcast
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclBroadcast(send_buf, recv_buf, kCount, ncclFloat, 0, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

        // Self-reduce
        ASSERT_EQ(ncclSuccess,
                  ncclReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, 0, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

        // Self-allgather (count=1M, recvbuf needs 1M for 1 rank)
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(send_buf, recv_buf, kCount, ncclFloat, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

        // Self-reducescatter (count=1M for 1 rank)
        ASSERT_EQ(ncclSuccess,
                  ncclReduceScatter(send_buf, recv_buf, kCount, ncclFloat, ncclSum, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// ShrinkAbort variant of shrink-to-1
TEST_F(RevokeGapMPITest, ShrinkAbort_ToSingleRank)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;
    int         wsize  = MPIEnvironment::world_size;

    constexpr size_t kCount = 1024 * 1024;
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

    // In-flight allreduce before shrink
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, parent, stream));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    for (int r = 1; r < wsize; r++)
        excludeList.push_back(r);

    bool isExcluded = (rank != 0);
    ncclComm_t child = NCCL_COMM_NULL;

    if (!isExcluded) {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_ABORT));
        ASSERT_NE(child, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (!isExcluded) {
        auto child_guard = makeCommAutoGuard(child);

        int childSize = 0;
        ASSERT_EQ(ncclSuccess, ncclCommCount(child, &childSize));
        ASSERT_EQ(childSize, 1);

        for (int iter = 0; iter < 10; iter++) {
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, child, stream));
            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MPI_TESTS_ENABLED
