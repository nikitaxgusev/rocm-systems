/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file RevokeShrinkDeepStressMPITests.cpp
 *
 * Deep functional stress for the SHRINK + REVOKE combination. These tests
 * complement RevokeStressMPITests.cpp (which focuses on size/iteration sweeps)
 * by exercising harder structural combinations:
 *
 *   - Cascading revoke -> shrink chains (several generations)
 *   - Per-comm isolation: revoking comm A must not affect comm B on the same world
 *   - Different exclusion patterns repeated on a freshly created parent
 *   - Long lifecycle loops to expose per-iteration leaks
 *   - Cascading SHRINK_ABORT under in-flight load
 *   - Mixed asymmetric -> symmetric topology cascade
 *   - Tight 50-iteration loop with small buffers
 *   - ncclCommSplit interaction: revoke parent, verify split children survive
 */

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "nccl.h"

#include <algorithm>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

class RevokeShrinkDeepStressMPITest : public MPITestBase
{
};

// ---------------------------------------------------------------------------
// Local helpers (kept self-contained to avoid coupling to other test TUs).
// ---------------------------------------------------------------------------

namespace
{

// Last-rank-on-each-node exclusion (same shape as computeSymmetricExclude in
// the sibling test files but implemented locally so this TU stays independent).
void computeLastRankPerNodeExclude(int worldRank, int worldSize,
                                   std::vector<int>& excludeList, bool& isExcluded)
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
    {
        excludeList.push_back((n + 1) * localSize - 1);
    }
}

// Asymmetric exclusion: drop the last rank of the FIRST HALF of nodes only,
// producing an uneven per-node rank count (e.g. 7+7+8+8 across 4 nodes).
void computeAsymmetricExclude(int worldRank, int worldSize,
                              std::vector<int>& excludeList, bool& isExcluded)
{
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                        MPI_INFO_NULL, &nodeComm);
    int localSize = 0;
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int nNodes = (localSize > 0) ? (worldSize + localSize - 1) / localSize : 1;
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

// Build an exclusion list inside a generic communicator's user-rank space.
// The pattern selector lets the caller cycle through structurally different
// shapes without ever excluding rank 0 (a few APIs care about rank 0 being
// present).
//
//   pattern == 0: drop the single last rank
//   pattern == 1: drop the single second rank  (first non-zero)
//   pattern == 2: drop the middle rank
//   pattern == 3: drop every odd rank
//   pattern == 4: drop every even rank except 0
void buildExclusionPatternForComm(int pattern, int commSize, int myRankInComm,
                                  std::vector<int>& excludeList,
                                  bool& isExcluded)
{
    excludeList.clear();
    isExcluded = false;
    if(commSize <= 1)
        return;

    auto add = [&](int r) {
        excludeList.push_back(r);
        if(r == myRankInComm)
            isExcluded = true;
    };

    switch(pattern % 5)
    {
        case 0:
            add(commSize - 1);
            break;
        case 1:
            add(1);
            break;
        case 2:
            add(commSize / 2);
            break;
        case 3:
            for(int r = 1; r < commSize; r += 2)
                add(r);
            break;
        case 4:
            for(int r = 2; r < commSize; r += 2)
                add(r);
            break;
    }
}

} // namespace

// ===========================================================================
// 1) Cascading revoke -> shrink chain across 3 generations.
//    parent --revoke+shrink--> childA --revoke+shrink--> childB --revoke+shrink--> childC
//    AllReduce at each generation to prove the resulting comm is functional.
// ===========================================================================
TEST_F(RevokeShrinkDeepStressMPITest, CascadingRevokeShrink_3Levels)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit))
        << "Test requires at least 4 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  current = getActiveCommunicator();
    hipStream_t stream  = getActiveStream();

    constexpr size_t kCount = 1024 * 1024;
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

    // Generation 0: AllReduce on the initial parent.
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, current, stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // Children created via shrink at each generation. Reverse-order destruction
    // (vector destructor pops back-to-front) ensures grandchildren are destroyed
    // before parents.
    std::vector<NcclCommAutoGuard> childGuards;
    childGuards.reserve(3);

    // Once we're excluded we stop participating in revoke/shrink, but we keep
    // hitting the world-wide MPI_Barriers so the remaining ranks stay in sync.
    bool stillIn = true;

    constexpr int kLevels = 3;
    for(int level = 0; level < kLevels; level++)
    {
        if(stillIn)
        {
            ASSERT_EQ(ncclSuccess, ncclCommRevoke(current, NCCL_REVOKE_DEFAULT));
        }
        MPI_Barrier(MPI_COMM_WORLD);

        ncclComm_t child = NCCL_COMM_NULL;
        if(stillIn)
        {
            int myInCurrent = -1, currentSize = 0;
            ASSERT_EQ(ncclSuccess, ncclCommUserRank(current, &myInCurrent));
            ASSERT_EQ(ncclSuccess, ncclCommCount(current, &currentSize));

            // Drop the last rank of the current comm at every level. With 16 ranks
            // this gives 16 -> 15 -> 14 -> 13. With 8 ranks: 8 -> 7 -> 6 -> 5.
            ASSERT_GE(currentSize, 2) << "level=" << level
                                       << " comm too small to shrink further";
            std::vector<int> excludeList = {currentSize - 1};
            bool             excludedHere = (myInCurrent == currentSize - 1);

            if(!excludedHere)
            {
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

    MPI_Barrier(MPI_COMM_WORLD);
}

// ===========================================================================
// 2) Multi-comm isolation: revoking comm A must not break comm B.
//    Both comms cover MPI_COMM_WORLD but are independent NCCL handles.
// ===========================================================================
TEST_F(RevokeShrinkDeepStressMPITest, MultiComm_RevokeIsolation_OtherShrinkAndCollective)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit))
        << "Test requires at least 4 MPI processes";

    // commA from the fixture.
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  commA  = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         worldRank = MPIEnvironment::world_rank;
    int         worldSize = MPIEnvironment::world_size;

    // commB built manually with a fresh unique ID.
    ncclUniqueId idB = {};
    if(worldRank == 0)
    {
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&idB));
    }
    MPI_Bcast(&idB, sizeof(idB), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t commB = nullptr;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&commB, worldSize, idB, worldRank));
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_NE(commB, nullptr);
    auto commB_guard = makeCommAutoGuard(commB);

    constexpr size_t kCount = 256 * 1024;
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

    // Sanity: both comms work before revoke.
    ASSERT_EQ(ncclSuccess,
              ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, commA, stream));
    ASSERT_EQ(ncclSuccess,
              ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, commB, stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // Revoke commA only.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(commA, NCCL_REVOKE_DEFAULT));
    MPI_Barrier(MPI_COMM_WORLD);

    // commA must reject collectives.
    {
        ncclResult_t r = ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, commA, stream);
        ASSERT_MPI_EQ(ncclInvalidUsage, r);
    }

    // commB must STILL work end-to-end: collective, then shrink, then collective.
    ASSERT_EQ(ncclSuccess,
              ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, commB, stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded;
    computeLastRankPerNodeExclude(worldRank, worldSize, excludeList, isExcluded);
    ncclComm_t commB_child = NCCL_COMM_NULL;
    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(commB, excludeList.data(), excludeList.size(),
                                 &commB_child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(commB_child, nullptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        auto child_guard = makeCommAutoGuard(commB_child);
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, commB_child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ===========================================================================
// 3) Five revoke -> shrink cycles, each cycle picking a different exclusion
//    pattern. Parent comm is recreated between cycles.
// ===========================================================================
TEST_F(RevokeShrinkDeepStressMPITest, AlternatingExclusionPatterns_RevokeShrink_5Cycles)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit))
        << "Test requires at least 4 MPI processes";

    int worldSize = MPIEnvironment::world_size;

    constexpr size_t kCount = 256 * 1024;
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });

    for(int cycle = 0; cycle < 5; cycle++)
    {
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ncclComm_t  parent = getActiveCommunicator();
        hipStream_t stream = getActiveStream();

        int myInParent = -1;
        ASSERT_EQ(ncclSuccess, ncclCommUserRank(parent, &myInParent));

        std::vector<int> excludeList;
        bool             isExcluded;
        buildExclusionPatternForComm(cycle, worldSize, myInParent,
                                     excludeList, isExcluded);

        // Make sure the pattern leaves at least 2 ranks behind so the shrink
        // produces a useful child.
        ASSERT_LT(static_cast<int>(excludeList.size()), worldSize - 1)
            << "cycle=" << cycle << " excludes too many ranks";

        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, parent, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

        ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));
        MPI_Barrier(MPI_COMM_WORLD);

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
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, child, stream));
            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Rebuild parent for next cycle. cleanupTest() destroys both the
        // (already revoked) parent and the test stream so the next iteration
        // gets a fresh fixture.
        cleanupTest();
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// ===========================================================================
// 4) 15 fast lifecycle cycles: create -> AllReduce -> revoke -> shrink ->
//    AllReduce(child) -> destroy(child) -> destroy(parent). Aimed at
//    flushing out per-iteration leaks in the revoke+shrink path.
// ===========================================================================
TEST_F(RevokeShrinkDeepStressMPITest, DeepLifecycle_15Cycles_RevokeShrink_AllReduce)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit));

    int worldRank = MPIEnvironment::world_rank;
    int worldSize = MPIEnvironment::world_size;

    constexpr size_t kCount = 1024 * 1024; // 4 MiB float
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });

    constexpr int kCycles = 15;
    for(int cycle = 0; cycle < kCycles; cycle++)
    {
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ncclComm_t  parent = getActiveCommunicator();
        hipStream_t stream = getActiveStream();

        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, parent, stream));
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
// 5) Cascading SHRINK_ABORT: in-flight on parent, revoke+abort to childA;
//    in-flight on childA, revoke+abort to childB; collective on childB.
//    Validates that the abort flag remains correct under repeated cascading.
// ===========================================================================
TEST_F(RevokeShrinkDeepStressMPITest, CascadingShrinkAbort_2Levels_InFlight)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit))
        << "Test requires at least 4 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         worldRank = MPIEnvironment::world_rank;
    int         worldSize = MPIEnvironment::world_size;

    constexpr size_t kCount = 8 * 1024 * 1024; // 32 MiB float - real in-flight load
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

    // ----- Generation 0: parent + in-flight + revoke -----
    ASSERT_EQ(ncclSuccess,
              ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, parent, stream));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> ex0;
    bool             out0;
    computeLastRankPerNodeExclude(worldRank, worldSize, ex0, out0);
    ncclComm_t childA = NCCL_COMM_NULL;
    if(!out0)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, ex0.data(), ex0.size(), &childA, nullptr,
                                 NCCL_SHRINK_ABORT));
        ASSERT_NE(childA, nullptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ----- Generation 1: childA + in-flight + revoke -----
    ncclComm_t childB = NCCL_COMM_NULL;
    NcclCommAutoGuard childA_guard(NCCL_COMM_NULL);
    NcclCommAutoGuard childB_guard(NCCL_COMM_NULL);
    if(!out0)
    {
        childA_guard.set(childA);

        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, childA, stream));
        ASSERT_EQ(ncclSuccess, ncclCommRevoke(childA, NCCL_REVOKE_DEFAULT));

        int childASize = 0, childARank = -1;
        ASSERT_EQ(ncclSuccess, ncclCommCount(childA, &childASize));
        ASSERT_EQ(ncclSuccess, ncclCommUserRank(childA, &childARank));
        ASSERT_GE(childASize, 2) << "childA too small after first shrink";

        std::vector<int> ex1 = {childASize - 1};
        bool             out1 = (childARank == childASize - 1);

        if(!out1)
        {
            ASSERT_EQ(ncclSuccess,
                      ncclCommShrink(childA, ex1.data(), ex1.size(), &childB, nullptr,
                                     NCCL_SHRINK_ABORT));
            ASSERT_NE(childB, nullptr);
            childB_guard.set(childB);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ----- Final collective on childB to prove cascade is healthy -----
    if(childB_guard.get() != NCCL_COMM_NULL)
    {
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum,
                                childB_guard.get(), stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ===========================================================================
// 6) Asymmetric -> symmetric topology cascade.
//    Asymmetric shrink (uneven per-node count) -> AllReduce -> revoke ->
//    symmetric shrink -> AllReduce. Exercises both topology paths in one test.
//    Requires >= 2 nodes for the asymmetric step to be meaningful.
// ===========================================================================
TEST_F(RevokeShrinkDeepStressMPITest, RevokeShrink_AsymmetricThenSymmetric_Cascade)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          2, kNoNodeLimit))
        << "Test requires at least 4 MPI processes across 2 nodes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         worldRank = MPIEnvironment::world_rank;
    int         worldSize = MPIEnvironment::world_size;

    constexpr size_t kCount = 1024 * 1024;
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

    // Generation 0: AllReduce on initial parent (full world).
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, parent, stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // ----- Asymmetric shrink (no revoke needed: shrink alone is allowed) -----
    std::vector<int> exAsym;
    bool             outAsym;
    computeAsymmetricExclude(worldRank, worldSize, exAsym, outAsym);
    ncclComm_t childAsym = NCCL_COMM_NULL;
    if(!outAsym)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, exAsym.data(), exAsym.size(), &childAsym,
                                 nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(childAsym, nullptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // AllReduce on the asymmetric child + revoke + symmetric shrink.
    ncclComm_t childSym = NCCL_COMM_NULL;
    NcclCommAutoGuard childAsym_guard(NCCL_COMM_NULL);
    NcclCommAutoGuard childSym_guard(NCCL_COMM_NULL);
    if(!outAsym)
    {
        childAsym_guard.set(childAsym);

        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, childAsym, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

        ASSERT_EQ(ncclSuccess, ncclCommRevoke(childAsym, NCCL_REVOKE_DEFAULT));

        int childAsymSize = 0, childAsymRank = -1;
        ASSERT_EQ(ncclSuccess, ncclCommCount(childAsym, &childAsymSize));
        ASSERT_EQ(ncclSuccess, ncclCommUserRank(childAsym, &childAsymRank));
        ASSERT_GE(childAsymSize, 2);

        // Drop just the last rank of childAsym for the symmetric step.
        std::vector<int> exSym = {childAsymSize - 1};
        bool             outSym = (childAsymRank == childAsymSize - 1);
        if(!outSym)
        {
            ASSERT_EQ(ncclSuccess,
                      ncclCommShrink(childAsym, exSym.data(), exSym.size(),
                                     &childSym, nullptr, NCCL_SHRINK_DEFAULT));
            ASSERT_NE(childSym, nullptr);
            childSym_guard.set(childSym);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(childSym_guard.get() != NCCL_COMM_NULL)
    {
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum,
                                childSym_guard.get(), stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ===========================================================================
// 7) Tight 50-iteration loop with small buffers to surface per-iteration
//    overhead and accumulated state in the revoke+shrink path.
// ===========================================================================
TEST_F(RevokeShrinkDeepStressMPITest, RevokeShrink_TightLoop_50Iterations_SmallBuffers)
{
    ASSERT_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit));

    int worldRank = MPIEnvironment::world_rank;
    int worldSize = MPIEnvironment::world_size;

    constexpr size_t kCount = 64 * 1024; // 256 KiB float - fast
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });

    constexpr int kIters = 50;
    for(int iter = 0; iter < kIters; iter++)
    {
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ncclComm_t  parent = getActiveCommunicator();
        hipStream_t stream = getActiveStream();

        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, parent, stream));
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
// 8) ncclCommSplit interaction. Split parent into two halves, then revoke
//    the parent. Both split-children must remain functional.
// ===========================================================================
TEST_F(RevokeShrinkDeepStressMPITest, CommSplit_RevokeParent_SplitChildSurvives)
{
    ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired,
                                          1, kNoNodeLimit))
        << "Test requires at least 4 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         worldRank = MPIEnvironment::world_rank;
    int         worldSize = MPIEnvironment::world_size;

    // Two-color split: lower half vs upper half.
    int color = (worldRank < worldSize / 2) ? 0 : 1;
    int key   = worldRank;

    ncclComm_t splitChild = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommSplit(parent, color, key, &splitChild, /*config*/ nullptr));
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_NE(splitChild, nullptr);
    auto splitChild_guard = makeCommAutoGuard(splitChild);
    MPI_Barrier(MPI_COMM_WORLD);

    constexpr size_t kCount = 256 * 1024;
    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, kCount * sizeof(float)));
    auto buf_g = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));

    // Sanity: split child works before revoke.
    ASSERT_EQ(ncclSuccess,
              ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, splitChild, stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // Revoke the parent only.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));
    MPI_Barrier(MPI_COMM_WORLD);

    // Parent must reject collectives.
    {
        ncclResult_t r = ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, parent, stream);
        ASSERT_MPI_EQ(ncclInvalidUsage, r);
    }

    // Both halves of the split child must still work end-to-end.
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
    ASSERT_EQ(ncclSuccess,
              ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, splitChild, stream));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // And the split child should itself accept revoke + shrink (proving it is
    // a fully independent communicator).
    int splitSize = 0, splitRank = -1;
    ASSERT_EQ(ncclSuccess, ncclCommCount(splitChild, &splitSize));
    ASSERT_EQ(ncclSuccess, ncclCommUserRank(splitChild, &splitRank));
    if(splitSize >= 2)
    {
        ASSERT_EQ(ncclSuccess, ncclCommRevoke(splitChild, NCCL_REVOKE_DEFAULT));
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> excludeList = {splitSize - 1};
        bool             isExcluded  = (splitRank == splitSize - 1);
        ncclComm_t       splitGrand  = NCCL_COMM_NULL;
        if(!isExcluded)
        {
            ASSERT_EQ(ncclSuccess,
                      ncclCommShrink(splitChild, excludeList.data(),
                                     excludeList.size(), &splitGrand, nullptr,
                                     NCCL_SHRINK_DEFAULT));
            ASSERT_NE(splitGrand, nullptr);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if(!isExcluded)
        {
            auto grand_guard = makeCommAutoGuard(splitGrand);
            HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(buf, kCount));
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum,
                                    splitGrand, stream));
            HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

#endif // MPI_TESTS_ENABLED
