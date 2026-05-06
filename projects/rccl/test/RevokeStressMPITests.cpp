/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

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

class RevokeStressMPITest : public MPITestBase {};

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

/**
 * Stress: revoke then allreduce across 6 sizes (1K to 16M floats).
 * Verifies revoke correctly rejects collectives at all buffer sizes.
 */
TEST_F(RevokeStressMPITest, Revoke_RejectsCollectives_MultipleSizes)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT));
    MPI_Barrier(MPI_COMM_WORLD);

    const size_t sizes[] = {1024, 65536, 262144, 1048576, 4194304, 16777216};

    for (size_t count : sizes) {
        void* buf = nullptr;
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, count * sizeof(float)));
        auto guard = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, count));

        ncclResult_t result = ncclAllReduce(buf, buf, count, ncclFloat, ncclSum, comm, stream);
        ASSERT_MPI_EQ(ncclInvalidUsage, result);
    }
}

/**
 * Stress: revoke -> shrink -> 20 allreduces at 16M floats on child comm.
 * Exercises sustained throughput on a shrunk communicator.
 */
TEST_F(RevokeStressMPITest, RevokeShrink_LargeAllReduce_20Iterations)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;

    constexpr size_t kCount = 16 * 1024 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

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

        for (int iter = 0; iter < 20; iter++) {
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, child, stream));
            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Stress: revoke -> shrink -> sweep 6 sizes x 5 iterations of allreduce.
 * Covers small-to-large buffer transitions on child comm.
 */
TEST_F(RevokeStressMPITest, RevokeShrink_MultipleSizes_MultileIterations)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;

    constexpr size_t kMaxCount = 16 * 1024 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kMaxCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kMaxCount * sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

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

        const size_t sizes[] = {1024, 65536, 262144, 1048576, 4194304, 16777216};

        for (int iter = 0; iter < 5; iter++) {
            for (size_t count : sizes) {
                HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, count));
                HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, count));

                ASSERT_EQ(ncclSuccess,
                          ncclAllReduce(send_buf, recv_buf, count, ncclFloat, ncclSum, child, stream));
                HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Stress: NCCL_SHRINK_ABORT with 16M in-flight, then 15 allreduces on child.
 */
TEST_F(RevokeStressMPITest, ShrinkAbort_LargeInFlight_ManyCollectives)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;

    constexpr size_t kCount = 16 * 1024 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, parent, stream));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool isExcluded;
    computeSymmetricExclude(rank, MPIEnvironment::world_size, excludeList, isExcluded);
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

        for (int iter = 0; iter < 15; iter++) {
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, child, stream));
            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Stress: mixed collectives (allreduce, broadcast, reduce_scatter, allgather)
 * on child comm after revoke+shrink, 10 iterations, 4M elements.
 */
TEST_F(RevokeStressMPITest, RevokeShrink_MixedCollectives_Stress)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
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

        constexpr size_t kCount = 4 * 1024 * 1024;
        size_t totalBufSize = kCount * childSize * sizeof(float);

        void* buf1 = nullptr;
        void* buf2 = nullptr;
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf1, totalBufSize));
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf2, totalBufSize));
        auto g1 = makeScopeGuard([&]() { if(buf1) (void)hipFree(buf1); });
        auto g2 = makeScopeGuard([&]() { if(buf2) (void)hipFree(buf2); });

        for (int iter = 0; iter < 10; iter++) {
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf1, kCount * childSize));
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf2, kCount * childSize));

            // AllReduce
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(buf1, buf2, kCount, ncclFloat, ncclSum, child, stream));

            // Broadcast from root 0
            ASSERT_EQ(ncclSuccess,
                      ncclBroadcast(buf1, buf2, kCount, ncclFloat, 0, child, stream));

            // AllGather
            ASSERT_EQ(ncclSuccess,
                      ncclAllGather(buf1, buf2, kCount, ncclFloat, child, stream));

            // ReduceScatter
            ASSERT_EQ(ncclSuccess,
                      ncclReduceScatter(buf2, buf1, kCount, ncclFloat, ncclSum, child, stream));

            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Stress: P2P send/recv ring on child comm after revoke+shrink,
 * 10 iterations at 4M floats per message.
 */
TEST_F(RevokeStressMPITest, RevokeShrink_P2PRing_Stress)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
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

        if (childSize < 2) return;

        constexpr size_t kCount = 4 * 1024 * 1024;
        void* send_buf = nullptr;
        void* recv_buf = nullptr;
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
        auto sg = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
        auto rg = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

        int sendPeer = (childRank + 1) % childSize;
        int recvPeer = (childRank - 1 + childSize) % childSize;

        for (int iter = 0; iter < 10; iter++) {
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

            ASSERT_EQ(ncclSuccess, ncclGroupStart());
            ASSERT_EQ(ncclSuccess,
                      ncclSend(send_buf, kCount, ncclFloat, sendPeer, child, stream));
            ASSERT_EQ(ncclSuccess,
                      ncclRecv(recv_buf, kCount, ncclFloat, recvPeer, child, stream));
            ASSERT_EQ(ncclSuccess, ncclGroupEnd());

            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Stress: incomplete collective (rank np-1 skips), revoke, shrink,
 * then 10 allreduces at 16M on child. Tests recovery from stuck ops.
 */
TEST_F(RevokeStressMPITest, IncompleteCollective_RevokeShrink_ManyIterations)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;
    int         wsize  = MPIEnvironment::world_size;

    constexpr size_t kCount = 16 * 1024 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

    if (rank != wsize - 1) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, parent, stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    if (rank != wsize - 1) {
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool isExcluded;
    computeSymmetricExclude(rank, wsize, excludeList, isExcluded);
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

        for (int iter = 0; iter < 10; iter++) {
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, child, stream));
            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Stress: multiple datatypes (float, double, half, int32) on child after shrink.
 * 5 iterations at 4M elements each.
 */
TEST_F(RevokeStressMPITest, RevokeShrink_MultipleDataTypes)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));
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

        constexpr size_t kCount = 4 * 1024 * 1024;
        // Allocate max-size buffer (double is the largest type)
        void* buf1 = nullptr;
        void* buf2 = nullptr;
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf1, kCount * sizeof(double)));
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf2, kCount * sizeof(double)));
        auto g1 = makeScopeGuard([&]() { if(buf1) (void)hipFree(buf1); });
        auto g2 = makeScopeGuard([&]() { if(buf2) (void)hipFree(buf2); });

        struct DTypeConfig {
            ncclDataType_t dtype;
            size_t elemSize;
            const char* name;
        };
        DTypeConfig dtypes[] = {
            {ncclFloat,   sizeof(float),    "float"},
            {ncclDouble,  sizeof(double),   "double"},
            {ncclFloat16, sizeof(uint16_t), "float16"},
            {ncclInt32,   sizeof(int32_t),  "int32"},
        };

        for (auto& dt : dtypes) {
            for (int iter = 0; iter < 5; iter++) {
                HIP_TEST_CHECK_GTEST_FAIL(hipMemset(buf1, 0, kCount * dt.elemSize));
                HIP_TEST_CHECK_GTEST_FAIL(hipMemset(buf2, 0, kCount * dt.elemSize));

                ASSERT_EQ(ncclSuccess,
                          ncclAllReduce(buf1, buf2, kCount, dt.dtype, ncclSum, child, stream));
                HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Stress: 5 consecutive revoke -> destroy cycles on fresh communicators.
 * Verifies no resource leaks accumulate over repeated lifecycles.
 */
TEST_F(RevokeStressMPITest, RepeatedRevokeDestroy_5Cycles)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit));

    for (int cycle = 0; cycle < 5; cycle++) {
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

        ncclComm_t  comm   = getActiveCommunicator();
        hipStream_t stream = getActiveStream();

        constexpr size_t kCount = 4 * 1024 * 1024;
        void* buf = nullptr;
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
        auto guard = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

        ASSERT_MPI_EQ(ncclSuccess,
                      ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, comm, stream));

        ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

        MPI_Barrier(MPI_COMM_WORLD);

        ASSERT_MPI_EQ(ncclSuccess, ncclCommDestroy(comm));
        test_comm_ = nullptr;

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

#endif // MPI_TESTS_ENABLED
