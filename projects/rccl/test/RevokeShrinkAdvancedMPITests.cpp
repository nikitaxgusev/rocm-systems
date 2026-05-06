/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file RevokeShrinkAdvancedMPITests.cpp
 *
 * Advanced functional coverage for the SHRINK + REVOKE combination, focused
 * on the gaps that RevokeMPITests / RevokeStressMPITests / RevokeShrinkDeep
 * do not exercise.
 *
 * Tests (11):
 *   A. CascadingRevokeShrink_5Levels
 *   B. LongSoak_100Cycles_RevokeShrink
 *   C. HugeBuffer_1GiB_RevokeShrink_AllReduce
 *   D. AsymmetricShrinkAbort_InFlight
 *   E. CommSplit_ShrinkSplitChild_Grandchild
 *   F. SplitChildren_Independence_RevokeOneOtherSurvives
 *   G. MemoryLeak_30Cycles_VerifyDelta
 *   H. RegisterBuffer_RevokeShrink_RegisterOnChild
 *   I. ConcurrentRevoke_TwoComms_PerRankThreads
 *   J. CrossStream_RevokeFromAlternateStream
 *   K. MultiThreaded_RevokeShrinkLifecycle_PerComm
 *
 * Intentionally NOT covered here (rationale per item):
 *   - Revoke during in-progress ncclCommShrink: race-prone, behaviour is
 *     not specified by the RCCL/NCCL API contract; would need an external
 *     synchronisation primitive to be deterministic.
 *   - Mismatched flags across ranks (rank 0 SHRINK_DEFAULT, rank 1
 *     SHRINK_ABORT, etc.): undefined behaviour per the API contract,
 *     likely deadlocks or asserts.
 *   - Real network partition (drop a peer mid-collective): requires
 *     out-of-test control of the IB/Ionic transport, not feasible from a
 *     gtest.
 *   - Non-rank-0 revoke origination: ncclCommRevoke is an all-ranks call,
 *     so "originating rank" is not a meaningful axis at this layer.
 *
 * Multi-threaded tests (I, K) only call NCCL APIs from worker threads.
 * MPI calls are confined to the main thread (post-join) to avoid coupling
 * test correctness to the MPI threading mode.  MPI_Init_thread is invoked
 * with MPI_THREAD_MULTIPLE by MPIEnvironment, so this is conservative.
 */

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "nccl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

class RevokeShrinkAdvancedMPITest : public MPITestBase
{
};

// ---------------------------------------------------------------------------
// Local helpers (kept self-contained so this TU has no link-time deps on the
// other revoke test files).
// ---------------------------------------------------------------------------
namespace
{

// Drop the last rank on every node (e.g. ranks 7 and 15 on a 8x2 layout).
void computeLastRankPerNodeExclude(int worldRank, int worldSize,
                                   std::vector<int>& excludeList,
                                   bool& isExcluded)
{
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                        MPI_INFO_NULL, &nodeComm);
    int localSize = 0, localRank = 0;
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int numNodes = (localSize > 0) ? worldSize / localSize : 1;
    isExcluded = (localRank == localSize - 1);
    excludeList.clear();
    for(int n = 0; n < numNodes; n++)
        excludeList.push_back((n + 1) * localSize - 1);
}

// Drop the last rank of the FIRST HALF of the nodes only -> uneven layout.
void computeAsymmetricExclude(int worldRank, int worldSize,
                              std::vector<int>& excludeList, bool& isExcluded)
{
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                        MPI_INFO_NULL, &nodeComm);
    int localSize = 0;
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int nNodes         = (localSize > 0) ? (worldSize + localSize - 1) / localSize : 1;
    int nodesToExclude = std::max(1, nNodes / 2);

    excludeList.clear();
    isExcluded = false;
    for(int n = 0; n < nodesToExclude; n++)
    {
        int excludedRank = n * localSize + (localSize - 1);
        if(excludedRank < worldSize)
        {
            excludeList.push_back(excludedRank);
            if(worldRank == excludedRank)
                isExcluded = true;
        }
    }
}

// Generate a fresh nccl unique ID on rank 0 and broadcast.
ncclUniqueId broadcastUniqueId(int worldRank)
{
    ncclUniqueId id = {};
    if(worldRank == 0)
    {
        ncclResult_t r = ncclGetUniqueId(&id);
        (void)r;
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    return id;
}

} // namespace

// ===========================================================================
// A) Cascading revoke -> shrink for 5 generations.
//
// Pushes the chain depth past the 3-level mark covered by the deep-stress
// suite. Drops the last rank at each level: with 16 ranks the chain is
// 16 -> 15 -> 14 -> 13 -> 12 -> 11; with 8 ranks it is 8 -> 7 -> ... -> 3.
// ===========================================================================
TEST_F(RevokeShrinkAdvancedMPITest, CascadingRevokeShrink_5Levels)
{
    ASSERT_TRUE(validateTestPrerequisites(6, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit))
        << "Test requires at least 6 MPI processes (5 levels of last-rank exclusion)";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  current = getActiveCommunicator();
    hipStream_t stream  = getActiveStream();

    constexpr size_t kCount = 256 * 1024;
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, current, stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<NcclCommAutoGuard> childGuards;
    childGuards.reserve(5);

    bool stillIn = true;
    constexpr int kLevels = 5;
    for(int level = 0; level < kLevels; level++)
    {
        if(stillIn)
        {
            ASSERT_EQ(ncclSuccess, ncclCommRevoke(current, NCCL_REVOKE_DEFAULT));
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if(stillIn)
        {
            int myInCurrent = -1, currentSize = 0;
            ASSERT_EQ(ncclSuccess, ncclCommUserRank(current, &myInCurrent));
            ASSERT_EQ(ncclSuccess, ncclCommCount(current, &currentSize));
            ASSERT_GE(currentSize, 2)
                << "level=" << level << " comm too small to shrink further";

            std::vector<int> excludeList = {currentSize - 1};
            bool             droppedHere = (myInCurrent == currentSize - 1);

            if(!droppedHere)
            {
                ncclComm_t child = NCCL_COMM_NULL;
                ASSERT_EQ(ncclSuccess,
                          ncclCommShrink(current, excludeList.data(),
                                         excludeList.size(), &child, nullptr,
                                         NCCL_SHRINK_DEFAULT));
                ASSERT_NE(child, nullptr);
                childGuards.emplace_back(child);
                current = child;
            }
            else
            {
                stillIn = false;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if(stillIn)
        {
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, current, stream));
            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// ===========================================================================
// B) Long soak: 100 full revoke+shrink lifecycles with small buffers.
// ===========================================================================
TEST_F(RevokeShrinkAdvancedMPITest, LongSoak_100Cycles_RevokeShrink)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit));

    int worldRank = MPIEnvironment::world_rank;
    int worldSize = MPIEnvironment::world_size;

    constexpr size_t kCount = 32 * 1024; // 128 KiB float
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });

    constexpr int kCycles = 100;
    for(int cycle = 0; cycle < kCycles; cycle++)
    {
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ncclComm_t  parent = getActiveCommunicator();
        hipStream_t stream = getActiveStream();

        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, parent, stream));

        ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> excludeList;
        bool             isExcluded;
        computeLastRankPerNodeExclude(worldRank, worldSize, excludeList, isExcluded);

        ncclComm_t child = NCCL_COMM_NULL;
        if(!isExcluded)
        {
            ASSERT_EQ(ncclSuccess,
                      ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                     &child, nullptr, NCCL_SHRINK_DEFAULT));
            ASSERT_NE(child, nullptr);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if(!isExcluded)
        {
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, child, stream));
            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
            ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));
        }
        MPI_Barrier(MPI_COMM_WORLD);

        cleanupTest();
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// ===========================================================================
// C) Huge buffer: 1 GiB AllReduce on parent + revoke + shrink + 1 GiB
//    AllReduce on child. Skips at runtime if there is not enough free
//    GPU memory for two 1 GiB buffers + headroom.
// ===========================================================================
TEST_F(RevokeShrinkAdvancedMPITest, HugeBuffer_1GiB_RevokeShrink_AllReduce)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit));

    constexpr size_t kBytes = 1ULL << 30; // 1 GiB
    constexpr size_t kHeadroom = 256ULL << 20; // 256 MiB

    size_t freeMem = 0, totalMem = 0;
    HIP_TEST_CHECK_GTEST_FAIL(hipMemGetInfo(&freeMem, &totalMem));
    bool localFits = (freeMem >= 2 * kBytes + kHeadroom);
    int  localFitsI = localFits ? 1 : 0;
    int  globalFitsI = 0;
    MPI_Allreduce(&localFitsI, &globalFitsI, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(globalFitsI == 0)
    {
        GTEST_SKIP() << "Not enough free GPU memory for 1 GiB allreduce on every rank";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         worldRank = MPIEnvironment::world_rank;
    int         worldSize = MPIEnvironment::world_size;

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kBytes));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kBytes));
    auto sg = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto rg = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });
    HIP_TEST_CHECK_GTEST_FAIL(hipMemset(send_buf, 0, kBytes));
    HIP_TEST_CHECK_GTEST_FAIL(hipMemset(recv_buf, 0, kBytes));

    const size_t kCountFloats = kBytes / sizeof(float);

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCountFloats, ncclFloat, ncclSum,
                                parent, stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded;
    computeLastRankPerNodeExclude(worldRank, worldSize, excludeList, isExcluded);
    ncclComm_t child = NCCL_COMM_NULL;
    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        auto child_guard = makeCommAutoGuard(child);
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCountFloats, ncclFloat, ncclSum,
                                child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ===========================================================================
// D) Asymmetric exclusion combined with NCCL_SHRINK_ABORT and an in-flight
//    16 MiB AllReduce. The existing asymmetric test only exercises
//    NCCL_SHRINK_DEFAULT.
// ===========================================================================
TEST_F(RevokeShrinkAdvancedMPITest, AsymmetricShrinkAbort_InFlight)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          2, kNoNodeLimit))
        << "Test requires at least 4 MPI processes across 2 nodes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         worldRank = MPIEnvironment::world_rank;
    int         worldSize = MPIEnvironment::world_size;

    constexpr size_t kCount = 4 * 1024 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
    auto sg = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto rg = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

    // In-flight collective.
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum,
                                parent, stream));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded;
    computeAsymmetricExclude(worldRank, worldSize, excludeList, isExcluded);

    ncclComm_t child = NCCL_COMM_NULL;
    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_ABORT));
        ASSERT_NE(child, nullptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        auto child_guard = makeCommAutoGuard(child);
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum,
                                child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ===========================================================================
// E) Split parent into 2 halves, then shrink one of the split children
//    (drop its last rank) and run AllReduce on the resulting grandchild.
// ===========================================================================
TEST_F(RevokeShrinkAdvancedMPITest, CommSplit_ShrinkSplitChild_Grandchild)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         worldRank = MPIEnvironment::world_rank;
    int         worldSize = MPIEnvironment::world_size;

    int color = (worldRank < worldSize / 2) ? 0 : 1;
    int key   = worldRank;

    ncclComm_t splitChild = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommSplit(parent, color, key, &splitChild, nullptr));
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_NE(splitChild, nullptr);
    auto split_guard = makeCommAutoGuard(splitChild);

    constexpr size_t kCount = 256 * 1024;
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

    ASSERT_EQ(ncclSuccess,
              ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, splitChild, stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // Now shrink the split-child: drop its last rank.
    int splitSize = 0, splitRank = -1;
    ASSERT_EQ(ncclSuccess, ncclCommCount(splitChild, &splitSize));
    ASSERT_EQ(ncclSuccess, ncclCommUserRank(splitChild, &splitRank));
    ASSERT_GE(splitSize, 2);

    std::vector<int> excludeList = {splitSize - 1};
    bool             isExcluded  = (splitRank == splitSize - 1);

    ncclComm_t grandchild = NCCL_COMM_NULL;
    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(splitChild, excludeList.data(), excludeList.size(),
                                 &grandchild, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(grandchild, nullptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        auto grand_guard = makeCommAutoGuard(grandchild);
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, grandchild,
                                stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ===========================================================================
// F) Split into 2 colors, revoke the split-child of color 0, then verify
//    the split-child of color 1 still works (collective + shrink).
//    Validates that revoking one split-child does not affect its sibling.
// ===========================================================================
TEST_F(RevokeShrinkAdvancedMPITest, SplitChildren_Independence_RevokeOneOtherSurvives)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         worldRank = MPIEnvironment::world_rank;
    int         worldSize = MPIEnvironment::world_size;

    int color = (worldRank < worldSize / 2) ? 0 : 1;
    int key   = worldRank;

    ncclComm_t splitChild = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommSplit(parent, color, key, &splitChild, nullptr));
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_NE(splitChild, nullptr);
    auto split_guard = makeCommAutoGuard(splitChild);

    constexpr size_t kCount = 256 * 1024;
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

    // Color-0 ranks revoke their split-child. Color-1 ranks DO NOT touch theirs.
    if(color == 0)
    {
        ASSERT_EQ(ncclSuccess, ncclCommRevoke(splitChild, NCCL_REVOKE_DEFAULT));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(color == 0)
    {
        // Color-0 split-child is revoked: collectives must reject.
        ncclResult_t r = ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum,
                                       splitChild, stream);
        ASSERT_EQ(ncclInvalidUsage, r);
    }
    else
    {
        // Color-1 split-child is healthy: collective must succeed.
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, splitChild,
                                stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Color-1 ranks now shrink their (still-healthy) split-child.
    if(color == 1)
    {
        int splitSize = 0, splitRank = -1;
        ASSERT_EQ(ncclSuccess, ncclCommCount(splitChild, &splitSize));
        ASSERT_EQ(ncclSuccess, ncclCommUserRank(splitChild, &splitRank));
        ASSERT_GE(splitSize, 2);

        std::vector<int> excludeList = {splitSize - 1};
        bool             isExcluded  = (splitRank == splitSize - 1);
        ncclComm_t       grandchild  = NCCL_COMM_NULL;
        if(!isExcluded)
        {
            ASSERT_EQ(ncclSuccess,
                      ncclCommShrink(splitChild, excludeList.data(),
                                     excludeList.size(), &grandchild, nullptr,
                                     NCCL_SHRINK_DEFAULT));
            ASSERT_NE(grandchild, nullptr);
        }

        if(!isExcluded)
        {
            auto grand_guard = makeCommAutoGuard(grandchild);
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, grandchild,
                                    stream));
            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ===========================================================================
// G) Memory leak detector: warmup 5 cycles, snapshot free GPU mem, run 25
//    more revoke+shrink+destroy cycles, snapshot again.  Assert that the
//    delta is bounded.
//
//    The threshold is loose because RCCL (and HIP) cache pools internally;
//    a real leak shows up as continuous growth, not bounded jitter.
// ===========================================================================
TEST_F(RevokeShrinkAdvancedMPITest, MemoryLeak_30Cycles_VerifyDelta)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit));

    int worldRank = MPIEnvironment::world_rank;
    int worldSize = MPIEnvironment::world_size;

    constexpr size_t kCount       = 256 * 1024;
    constexpr int    kWarmup      = 5;
    constexpr int    kMeasure     = 25;
    constexpr size_t kAllowedDelta = 256ULL << 20; // 256 MiB - generous

    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });

    auto runOneCycle = [&]() {
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ncclComm_t  parent = getActiveCommunicator();
        hipStream_t stream = getActiveStream();

        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, parent, stream));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> excludeList;
        bool             isExcluded;
        computeLastRankPerNodeExclude(worldRank, worldSize, excludeList, isExcluded);

        ncclComm_t child = NCCL_COMM_NULL;
        if(!isExcluded)
        {
            ASSERT_EQ(ncclSuccess,
                      ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                     &child, nullptr, NCCL_SHRINK_DEFAULT));
            ASSERT_NE(child, nullptr);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if(!isExcluded)
        {
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, child, stream));
            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
            ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));
        }
        MPI_Barrier(MPI_COMM_WORLD);

        cleanupTest();
        MPI_Barrier(MPI_COMM_WORLD);
    };

    for(int i = 0; i < kWarmup; i++)
        runOneCycle();

    size_t freeBefore = 0, total = 0;
    HIP_TEST_CHECK_GTEST_FAIL(hipMemGetInfo(&freeBefore, &total));

    for(int i = 0; i < kMeasure; i++)
        runOneCycle();

    size_t freeAfter = 0;
    HIP_TEST_CHECK_GTEST_FAIL(hipMemGetInfo(&freeAfter, &total));

    long long delta = (long long)freeBefore - (long long)freeAfter;
    fprintf(stderr,
            "[rank %d] mem freeBefore=%zu MiB freeAfter=%zu MiB delta=%lld MiB\n",
            worldRank, freeBefore >> 20, freeAfter >> 20, delta >> 20);

    // delta > kAllowedDelta means free memory dropped by more than the bound,
    // i.e. some cycle resources were never released.
    ASSERT_LT(delta, (long long)kAllowedDelta)
        << "Possible leak: free GPU memory dropped by " << (delta >> 20)
        << " MiB across " << kMeasure << " revoke+shrink+destroy cycles";
}

// ===========================================================================
// H) ncclCommRegister + revoke + shrink + ncclCommRegister on child.
//    Validates the full registered-buffer lifecycle interacts cleanly with
//    revoke and shrink.
// ===========================================================================
TEST_F(RevokeShrinkAdvancedMPITest, RegisterBuffer_RevokeShrink_RegisterOnChild)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         worldRank = MPIEnvironment::world_rank;
    int         worldSize = MPIEnvironment::world_size;

    constexpr size_t kCount = 256 * 1024;
    constexpr size_t kBytes = kCount * sizeof(float);
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kBytes));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

    // Register on parent.  ncclCommRegister returns ncclSuccess with a NULL
    // handle on transports that do not support User Buffer Registration; in
    // that case the buffer is still usable, just unregistered.  We mirror the
    // pattern used by RegistrationMPITests.cpp.
    void* parentReg = nullptr;
    ASSERT_EQ(ncclSuccess, ncclCommRegister(parent, buf, kBytes, &parentReg));

    ASSERT_EQ(ncclSuccess,
              ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, parent, stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

    // Deregister BEFORE revoke (we don't want to interact with the
    // implementation-defined cleanup of registrations on a revoked comm).
    if(parentReg)
    {
        ASSERT_EQ(ncclSuccess, ncclCommDeregister(parent, parentReg));
        parentReg = nullptr;
    }

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded;
    computeLastRankPerNodeExclude(worldRank, worldSize, excludeList, isExcluded);

    ncclComm_t child = NCCL_COMM_NULL;
    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        auto child_guard = makeCommAutoGuard(child);

        // Register the same backing buffer on the child comm.
        void* childReg = nullptr;
        ASSERT_EQ(ncclSuccess, ncclCommRegister(child, buf, kBytes, &childReg));

        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

        if(childReg)
        {
            ASSERT_EQ(ncclSuccess, ncclCommDeregister(child, childReg));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ===========================================================================
// I) Concurrent revoke from per-rank threads: each rank owns commA and
//    commB; thread 0 revokes commA, thread 1 revokes commB, both at the
//    same time. Validates per-comm thread safety of revoke.
// ===========================================================================
TEST_F(RevokeShrinkAdvancedMPITest, ConcurrentRevoke_TwoComms_PerRankThreads)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit));

    int worldRank = MPIEnvironment::world_rank;
    int worldSize = MPIEnvironment::world_size;

    // commA via the fixture.
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t commA = getActiveCommunicator();
    hipStream_t streamA = getActiveStream();

    // commB built manually (different unique ID, same world).
    ncclUniqueId idB = broadcastUniqueId(worldRank);
    ncclComm_t commB = nullptr;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&commB, worldSize, idB, worldRank));
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_NE(commB, nullptr);
    auto commB_guard = makeCommAutoGuard(commB);

    hipStream_t streamB = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamCreate(&streamB));
    auto streamB_guard = makeStreamAutoGuard(streamB);

    constexpr size_t kCount = 256 * 1024;
    void* bufA = nullptr;
    void* bufB = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&bufA, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&bufB, kCount * sizeof(float)));
    auto ag = makeScopeGuard([&]() { if(bufA) (void)hipFree(bufA); });
    auto bg = makeScopeGuard([&]() { if(bufB) (void)hipFree(bufB); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(bufA, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(bufB, kCount));

    std::atomic<int> failsA{0};
    std::atomic<int> failsB{0};

    // The work each thread performs. No MPI calls inside.
    auto worker = [&](ncclComm_t c, hipStream_t s, void* b, std::atomic<int>& f) {
        if(ncclAllReduce(b, b, kCount, ncclFloat, ncclSum, c, s) != ncclSuccess)
            f++;
        // Don't sync first - revoke must be safe even if there are device-side
        // ops in flight on this stream.
        if(ncclCommRevoke(c, NCCL_REVOKE_DEFAULT) != ncclSuccess)
            f++;
        if(hipStreamSynchronize(s) != hipSuccess)
            f++;
    };

    std::thread tA(worker, commA, streamA, bufA, std::ref(failsA));
    std::thread tB(worker, commB, streamB, bufB, std::ref(failsB));
    tA.join();
    tB.join();

    ASSERT_EQ(0, failsA.load());
    ASSERT_EQ(0, failsB.load());

    MPI_Barrier(MPI_COMM_WORLD);

    // Both comms should now be revoked from every rank.
    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclAllReduce(bufA, bufA, kCount, ncclFloat, ncclSum, commA, streamA));
    ASSERT_MPI_EQ(ncclInvalidUsage,
                  ncclAllReduce(bufB, bufB, kCount, ncclFloat, ncclSum, commB, streamB));
}

// ===========================================================================
// J) Cross-stream revoke: collective on stream A is in flight while revoke
//    is issued from stream B's context.  Single-thread variant of (I).
// ===========================================================================
TEST_F(RevokeShrinkAdvancedMPITest, CrossStream_RevokeFromAlternateStream)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  comm    = getActiveCommunicator();
    hipStream_t streamA = getActiveStream();
    int         worldRank = MPIEnvironment::world_rank;
    int         worldSize = MPIEnvironment::world_size;

    hipStream_t streamB = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamCreate(&streamB));
    auto streamB_guard = makeStreamAutoGuard(streamB);

    constexpr size_t kCount = 4 * 1024 * 1024;
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

    // Enqueue a sizable AllReduce on streamA.
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, comm, streamA));

    // Revoke (a host-side call, not tied to either stream).
    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT));

    // Drain streamA.  If revoke aborted the in-flight collective, this just
    // returns; if revoke deferred the abort, the collective completes here.
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(streamA));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(streamB));

    MPI_Barrier(MPI_COMM_WORLD);

    // Shrink and verify the child comm works.
    std::vector<int> excludeList;
    bool             isExcluded;
    computeLastRankPerNodeExclude(worldRank, worldSize, excludeList, isExcluded);
    ncclComm_t child = NCCL_COMM_NULL;
    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(comm, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        auto child_guard = makeCommAutoGuard(child);
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, child, streamA));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(streamA));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ===========================================================================
// K) Multi-threaded full lifecycle: per-rank thread A drives commA through
//    AllReduce -> Revoke -> Shrink -> AllReduce(child) -> Destroy(child),
//    while thread B does the same on commB - concurrently.
// ===========================================================================
TEST_F(RevokeShrinkAdvancedMPITest, MultiThreaded_RevokeShrinkLifecycle_PerComm)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit));

    int worldRank = MPIEnvironment::world_rank;
    int worldSize = MPIEnvironment::world_size;

    // Build commA via the fixture, commB manually.  Each comm covers all ranks.
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  commA   = getActiveCommunicator();
    hipStream_t streamA = getActiveStream();

    ncclUniqueId idB = broadcastUniqueId(worldRank);
    ncclComm_t commB = nullptr;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&commB, worldSize, idB, worldRank));
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_NE(commB, nullptr);
    auto commB_guard = makeCommAutoGuard(commB);

    hipStream_t streamB = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamCreate(&streamB));
    auto streamB_guard = makeStreamAutoGuard(streamB);

    // Compute the (identical) exclusion list once - cross-rank consensus
    // happens here on the main thread, before workers diverge.
    std::vector<int> excludeList;
    bool             isExcluded;
    computeLastRankPerNodeExclude(worldRank, worldSize, excludeList, isExcluded);

    constexpr size_t kCount = 256 * 1024;
    void* bufA = nullptr;
    void* bufB = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&bufA, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&bufB, kCount * sizeof(float)));
    auto ag = makeScopeGuard([&]() { if(bufA) (void)hipFree(bufA); });
    auto bg = makeScopeGuard([&]() { if(bufB) (void)hipFree(bufB); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(bufA, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(bufB, kCount));

    std::atomic<int> failsA{0};
    std::atomic<int> failsB{0};

    auto lifecycle = [&](ncclComm_t comm, hipStream_t s, void* b,
                         std::atomic<int>& f) {
        if(ncclAllReduce(b, b, kCount, ncclFloat, ncclSum, comm, s) != ncclSuccess)
            f++;
        if(ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT) != ncclSuccess)
            f++;
        if(hipStreamSynchronize(s) != hipSuccess)
            f++;

        if(!isExcluded)
        {
            ncclComm_t child = NCCL_COMM_NULL;
            if(ncclCommShrink(comm, excludeList.data(), excludeList.size(),
                              &child, nullptr, NCCL_SHRINK_DEFAULT)
               != ncclSuccess)
            {
                f++;
                return;
            }
            if(child == NCCL_COMM_NULL)
            {
                f++;
                return;
            }
            if(ncclAllReduce(b, b, kCount, ncclFloat, ncclSum, child, s)
               != ncclSuccess)
                f++;
            if(hipStreamSynchronize(s) != hipSuccess)
                f++;
            if(ncclCommDestroy(child) != ncclSuccess)
                f++;
        }
    };

    std::thread tA(lifecycle, commA, streamA, bufA, std::ref(failsA));
    std::thread tB(lifecycle, commB, streamB, bufB, std::ref(failsB));
    tA.join();
    tB.join();

    ASSERT_EQ(0, failsA.load());
    ASSERT_EQ(0, failsB.load());

    MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MPI_TESTS_ENABLED
