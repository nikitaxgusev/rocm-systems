/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file LsaBarrierTimeoutMPITests.cpp
 * @brief Device-level coverage for the LSA barrier ncclTimeout producer
 *        (AICOMNET-193).
 *
 * The NCCL v2.30.4 sync added a timeout-bounded wait to the LSA barrier:
 *
 *   ncclResult_t ncclLsaBarrierSession::wait(Coop, memory_order, uint64_t timeoutCycles);
 *   ncclResult_t ncclLsaBarrierSession::sync(Coop, memory_order, uint64_t timeoutCycles);
 *
 * On expiry of the clock64() budget the wait returns ncclTimeout instead of
 * spinning forever. These tests launch REAL ncclLsaBarrierSession kernels on an
 * ncclDevComm and cover the happy path, the timeout producer, and stress /
 * corner cases:
 *
 *   - HealthyBarrierReturnsSuccess     : all peers arrive -> ncclSuccess.
 *   - AbsentPeerProducesTimeout        : one peer never arrives -> ncclTimeout.
 *   - ZeroBudgetAbsentPeerTimesOut     : timeoutCycles==0 + absent peer -> immediate ncclTimeout.
 *   - RepeatedHealthyBarriersSucceed   : 64 sequential barriers all succeed (no drift / false trip).
 *   - RecoversAfterTimeout             : timeout round, then a fresh healthy barrier succeeds.
 *   - BackToBackTimeouts               : repeated stuck barriers each yield ncclTimeout (no hang).
 *
 * Timing note: clock64() is a cycle-domain device clock whose frequency is
 * arch-dependent. The cases here are deliberately frequency-robust -- they rely
 * on "absent peer always expires" and "all peers arrive always succeeds with a
 * generous budget", never on a precise cycle-to-wallclock mapping.
 *
 * The LSA device barrier requires symmetric memory (cuMem / kernel >= 6.8).
 * Where unavailable, ncclDevCommCreate fails and the test SKIPS uniformly on
 * every rank (the ASSERT_MPI_* macros are collective, so the skip decision must
 * be identical across ranks).
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

// Generous budget: a real barrier resolves well within this, so the healthy
// cases never false-trip regardless of device clock frequency.
constexpr uint64_t kHealthyTimeoutCycles = 5000000000ULL;
// Short budget for absent-peer cases: the missing arrival guarantees expiry,
// so keep it brief to bound runtime.
constexpr uint64_t kShortTimeoutCycles   = 200000000ULL;

// One bounded barrier; writes the resulting ncclResult_t to device scratch.
__global__ void lsaBarrierTimeoutKernel(struct ncclDevComm devComm,
                                        uint64_t timeoutCycles,
                                        int* outResult) {
  ncclLsaBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), devComm, ncclTeamTagLsa(), /*index=*/0u};

  ncclResult_t r = bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, timeoutCycles);

  if (threadIdx.x == 0 && blockIdx.x == 0) *outResult = static_cast<int>(r);
}

// N sequential bounded barriers in one kernel; writes how many returned
// ncclSuccess. Stresses repeated healthy barriers (no drift / false trip across
// iterations that reuse the same barrier slot).
__global__ void lsaRepeatedBarrierKernel(struct ncclDevComm devComm,
                                         uint64_t timeoutCycles,
                                         int iterations,
                                         int* outSuccessCount) {
  int successes = 0;
  for (int i = 0; i < iterations; ++i) {
    ncclLsaBarrierSession<ncclCoopCta> bar{
        ncclCoopCta(), devComm, ncclTeamTagLsa(), /*index=*/0u};
    ncclResult_t r = bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, timeoutCycles);
    if (r == ncclSuccess) ++successes;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) *outSuccessCount = successes;
}

// Create an LSA devComm with `nBarriers` barrier slots; returns create status.
ncclResult_t createLsaDevComm(ncclComm_t comm, int nBarriers, ncclDevComm* outDevComm) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = nBarriers;
  return ncclDevCommCreate(comm, &reqs, outDevComm);
}

// Launch one bounded barrier on this rank and return the device-produced code.
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

} // namespace

class LsaBarrierTimeoutMPITest : public MPITestBase {
 protected:
  // When setUp* returns false, this holds a non-empty reason iff the test
  // should SKIP (vs. fail). GTEST_SKIP() must be issued from the test body
  // itself (it expands to a return), so the helper only records the reason.
  std::string skipReason_;

  // Validate ranks + create comm + create LSA devComm. Returns false when
  // prerequisites are unmet; on a skip-worthy condition skipReason_ is set.
  // On success, fills comm/stream/devComm; CALLER owns destroy (use SCOPE_EXIT).
  bool setUpLsaDevComm(int nBarriers, ncclComm_t* commOut, hipStream_t* streamOut,
                       ncclDevComm* devCommOut) {
    skipReason_.clear();
    if (!validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit)) {
      ADD_FAILURE() << "Test requires at least 2 MPI processes";
      return false;
    }
    if (createTestCommunicator() != ncclSuccess) {
      ADD_FAILURE() << "createTestCommunicator failed";
      return false;
    }
    ncclComm_t comm = getActiveCommunicator();
    ncclResult_t devRc = createLsaDevComm(comm, nBarriers, devCommOut);
    if (devRc != ncclSuccess) {
      skipReason_ = std::string("LSA barrier devComm requires symmetric memory "
                                "(cuMem, kernel >= 6.8); ncclDevCommCreate returned ")
                  + ncclGetErrorString(devRc);
      return false;
    }
    *commOut   = comm;
    *streamOut = getActiveStream();
    return true;
  }
};

// Issue GTEST_SKIP (if a skip reason was recorded) or just return, from the
// void test body where a return statement is valid.
#define LSA_SETUP_OR_BAIL()                                  \
    do {                                                     \
        if (!skipReason_.empty()) GTEST_SKIP() << skipReason_; \
        return;                                              \
    } while (0)

// All ranks arrive -> bounded sync completes within budget -> ncclSuccess.
// Guards against the timeout path firing on a progressing barrier.
TEST_F(LsaBarrierTimeoutMPITest, HealthyBarrierReturnsSuccess)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    MPI_Barrier(MPI_COMM_WORLD);
    int hResult = runOneBarrier(devComm, stream, kHealthyTimeoutCycles);

    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess), hResult);
}

// Rank (nRanks-1) never enters -> every waiting rank's deadline expires ->
// ncclTimeout. The genuine device-side production of ncclTimeout.
TEST_F(LsaBarrierTimeoutMPITest, AbsentPeerProducesTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    int rank = -1, nRanks = -1;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);
    const bool isAbsent = (rank == nRanks - 1);

    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runOneBarrier(devComm, stream, kShortTimeoutCycles);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "Rank " << rank << " expected ncclTimeout from a stuck LSA barrier, got "
            << ncclGetErrorString(static_cast<ncclResult_t>(hResult));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int local = (isAbsent || hResult == static_cast<int>(ncclTimeout)) ? 1 : 0;
    ASSERT_MPI_TRUE(local == 1);
}

// Corner case: a zero-cycle budget with an absent peer must time out
// immediately (the first deadline check fires before any arrival).
TEST_F(LsaBarrierTimeoutMPITest, ZeroBudgetAbsentPeerTimesOut)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    int rank = -1, nRanks = -1;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);
    const bool isAbsent = (rank == nRanks - 1);

    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runOneBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "Rank " << rank << " expected immediate ncclTimeout with zero budget, got "
            << ncclGetErrorString(static_cast<ncclResult_t>(hResult));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int local = (isAbsent || hResult == static_cast<int>(ncclTimeout)) ? 1 : 0;
    ASSERT_MPI_TRUE(local == 1);
}

// Stress: many sequential healthy barriers on the same slot must all succeed.
// Catches epoch/drift bugs where a later iteration would spuriously time out.
TEST_F(LsaBarrierTimeoutMPITest, RepeatedHealthyBarriersSucceed)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    constexpr int kIters = 64;
    int* dCount = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&dCount, sizeof(int)));
    SCOPE_EXIT(if (dCount) (void)hipFree(dCount));
    int zero = 0;
    HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(dCount, &zero, sizeof(int), hipMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    lsaRepeatedBarrierKernel<<<1, 256, 0, stream>>>(devComm, kHealthyTimeoutCycles, kIters, dCount);
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

    int hCount = -1;
    HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(&hCount, dCount, sizeof(int), hipMemcpyDeviceToHost));
    ASSERT_MPI_EQ(kIters, hCount);
}

// Recovery: a timeout round (absent peer), then a fresh healthy barrier must
// succeed -- a tripped timeout must not wedge subsequent device barriers. The
// absent rank skips the timeout round but rejoins for the healthy round.
TEST_F(LsaBarrierTimeoutMPITest, RecoversAfterTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    int rank = -1, nRanks = -1;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);
    const bool isAbsent = (rank == nRanks - 1);

    // Round 1: absent peer -> waiting ranks time out.
    if (!isAbsent) {
        int r1 = runOneBarrier(devComm, stream, kShortTimeoutCycles);
        EXPECT_EQ(static_cast<int>(ncclTimeout), r1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Round 2: a fresh devComm gives a clean barrier slot for all ranks. (A slot
    // abandoned mid-flight by half the ranks is not a supported reuse path; a
    // new devComm is how a real caller recovers from a barrier timeout.)
    ncclDevComm devComm2{};
    ASSERT_MPI_EQ(ncclSuccess, createLsaDevComm(comm, 1, &devComm2));
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm2));

    MPI_Barrier(MPI_COMM_WORLD);
    int r2 = runOneBarrier(devComm2, stream, kHealthyTimeoutCycles);
    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess), r2);
}

// Stress: repeated stuck barriers (absent peer) must each deterministically
// return ncclTimeout with no hang and no spurious success.
TEST_F(LsaBarrierTimeoutMPITest, BackToBackTimeouts)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpLsaDevComm(1, &comm, &stream, &devComm)) LSA_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    int rank = -1, nRanks = -1;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);
    const bool isAbsent = (rank == nRanks - 1);

    constexpr int kRounds = 4;
    int timeouts = 0;
    for (int i = 0; i < kRounds; ++i) {
        // Fresh devComm per round: the absent peer makes slot reuse unsafe, so
        // each stuck barrier gets its own clean slot. Both present and absent
        // ranks create/destroy so the collective calls stay matched.
        ncclDevComm dc{};
        if (createLsaDevComm(comm, 1, &dc) != ncclSuccess) break;
        if (!isAbsent) {
            int r = runOneBarrier(dc, stream, kShortTimeoutCycles);
            if (r == static_cast<int>(ncclTimeout)) ++timeouts;
        }
        (void)ncclDevCommDestroy(comm, &dc);
    }
    if (!isAbsent) EXPECT_EQ(kRounds, timeouts);
    MPI_Barrier(MPI_COMM_WORLD);

    int local = (isAbsent || timeouts == kRounds) ? 1 : 0;
    ASSERT_MPI_TRUE(local == 1);
}

#endif // MPI_TESTS_ENABLED
