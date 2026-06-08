/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file GinBarrierTimeoutMPITests.cpp
 * @brief Device-level coverage for the GIN barrier ncclTimeout producer
 *        (AICOMNET-193).
 *
 * The NCCL v2.30.4 sync added a timeout-bounded wait to the GIN barrier:
 *
 *   ncclResult_t ncclGinBarrierSession::sync(Coop, memory_order,
 *                                            ncclGinFenceLevel, uint64_t timeoutCycles);
 *
 * The timeout path replaces the unbounded waitSignal spin with a readSignal +
 * clock64() deadline loop that returns ncclTimeout on expiry. These tests launch
 * REAL ncclGinBarrierSession kernels on the rail GIN barrier and cover the happy
 * path, the timeout producer, and stress / corner cases:
 *
 *   - HealthyBarrierReturnsSuccess     : all peers arrive -> ncclSuccess.
 *   - AbsentPeerProducesTimeout        : one peer never arrives -> ncclTimeout.
 *   - ZeroBudgetAbsentPeerTimesOut     : timeoutCycles==0 + absent peer -> immediate ncclTimeout.
 *   - RepeatedHealthyBarriersSucceed   : 32 sequential barriers all succeed.
 *   - RecoversAfterTimeout             : timeout round, then a fresh healthy barrier succeeds.
 *   - BackToBackTimeouts               : repeated stuck barriers each yield ncclTimeout (no hang).
 *
 * The GIN barrier requires a GIN-capable build + symmetric memory + GIN proxy
 * connectivity (NCCL_GIN_TYPE=2, NCCL_CUMEM_ENABLE=1, intranet on single node).
 *
 * IMPORTANT: Run with exactly 1 MPI rank per node (--map-by ppr:1:node) so that
 * each MPI rank maps uniquely to one GIN rail rank. With >1 rank per node,
 * multiple MPI ranks share the same rail rank, so the absent-peer mechanism
 * cannot reliably starve a rail slot.
 * When any prerequisite is missing, ncclDevCommCreate fails or GIN is unbound,
 * and the test SKIPS uniformly on every rank (the ASSERT_MPI_* macros are
 * collective, so the skip must be identical across ranks).
 */

#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include "nccl.h"
#include "nccl_device.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <hip/hip_runtime.h>
#include <mpi.h>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;

namespace {

constexpr uint64_t kHealthyTimeoutCycles = 5000000000ULL;
constexpr uint64_t kShortTimeoutCycles   = 200000000ULL;

// --- GIN prerequisite probes (mirror GinDeviceMPITests.cpp) ----------------

std::string ginEnvDisabledReason() {
  if (const char* e = std::getenv("NCCL_GIN_ENABLE"); e && std::strcmp(e, "0") == 0)
    return "GIN explicitly disabled by environment (NCCL_GIN_ENABLE=0)";
  return "";
}

std::string ginTypeReason() {
  const char* ginType = std::getenv("NCCL_GIN_TYPE");
  if (!ginType)
    return "GIN type not set (required NCCL_GIN_TYPE=2)";
  if (std::atoi(ginType) != 2)
    return std::string("Invalid GIN type: ") + ginType + " (required NCCL_GIN_TYPE=2)";
  return "";
}

std::string cuMemReason() {
  const char* cumem = std::getenv("NCCL_CUMEM_ENABLE");
  if (!cumem || std::strcmp(cumem, "1") != 0)
    return "Symmetric memory required (NCCL_CUMEM_ENABLE=1)";
  return "";
}

// Single-node runs need intranet mode -- otherwise the topology pruner removes
// the NET node and GIN has no path to bind.
std::string intranetReason() {
  MPI_Comm nodeComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
  int nodeSize = 0, worldSize = 0;
  MPI_Comm_size(nodeComm, &nodeSize);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_free(&nodeComm);
  if (nodeSize != worldSize) return "";
  const char* intra = std::getenv("RCCL_ENABLE_INTRANET");
  if (!intra || std::strcmp(intra, "1") != 0)
    return "Intranet mode required for single-node run (RCCL_ENABLE_INTRANET=1)";
  return "";
}

// First failing prerequisite, or "" if all met.
std::string ginBarrierSkipReason() {
  for (auto check : {ginEnvDisabledReason, ginTypeReason, cuMemReason, intranetReason}) {
    if (auto reason = check(); !reason.empty()) return reason;
  }
  return "";
}

// One bounded rail-GIN barrier; writes the resulting ncclResult_t to scratch.
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

// N sequential bounded rail-GIN barriers; writes the ncclSuccess count.
__global__ void ginRepeatedBarrierKernel(struct ncclDevComm devComm,
                                         uint64_t timeoutCycles,
                                         int iterations,
                                         int* outSuccessCount) {
  ncclGin gin{devComm, /*ginContext=*/0};
  int successes = 0;
  for (int i = 0; i < iterations; ++i) {
    ncclGinBarrierSession<ncclCoopCta> bar{
        ncclCoopCta(), gin, ncclTeamTagRail{}, /*index=*/0u};
    ncclResult_t r = bar.sync(ncclCoopCta(), cuda::memory_order_relaxed,
                              ncclGinFenceLevel::Relaxed, timeoutCycles);
    if (r == ncclSuccess) ++successes;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) *outSuccessCount = successes;
}

// Create a GIN devComm with `nBarriers` rail-barrier slots; returns create status.
ncclResult_t createGinDevComm(ncclComm_t comm, int nBarriers, ncclDevComm* outDevComm) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.railGinBarrierCount = nBarriers;
  reqs.ginConnectionType   = NCCL_GIN_CONNECTION_RAIL;
  return ncclDevCommCreate(comm, &reqs, outDevComm);
}

// Launch one bounded barrier on this rank and return the device-produced code.
int runOneBarrier(ncclDevComm devComm, hipStream_t stream, uint64_t timeoutCycles) {
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

} // namespace

class GinBarrierTimeoutMPITest : public MPITestBase {
 protected:
  // When setUp* returns false, this holds a non-empty reason iff the test
  // should SKIP (vs. fail). GTEST_SKIP() must be issued from the test body
  // itself (it expands to a return), so the helper only records the reason.
  std::string skipReason_;

  // Validate GIN prereqs + ranks + create comm + create GIN devComm. Returns
  // false when prerequisites are unmet; on a skip-worthy condition skipReason_
  // is set. On success, fills comm/stream/devComm; CALLER owns destroy.
  bool setUpGinDevComm(int nBarriers, ncclComm_t* commOut, hipStream_t* streamOut,
                       ncclDevComm* devCommOut) {
    skipReason_.clear();
    if (auto reason = ginBarrierSkipReason(); !reason.empty()) {
      skipReason_ = reason;
      return false;
    }
    if (!validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit)) {
      ADD_FAILURE() << "Test requires at least 2 MPI processes";
      return false;
    }
    if (createTestCommunicator() != ncclSuccess) {
      ADD_FAILURE() << "createTestCommunicator failed";
      return false;
    }
    ncclComm_t comm = getActiveCommunicator();
    ncclResult_t devRc = createGinDevComm(comm, nBarriers, devCommOut);
    if (devRc != ncclSuccess) {
      skipReason_ = std::string("GIN barrier devComm unavailable; ncclDevCommCreate returned ")
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
#define GIN_SETUP_OR_BAIL()                                  \
    do {                                                     \
        if (!skipReason_.empty()) GTEST_SKIP() << skipReason_; \
        return;                                              \
    } while (0)

// All ranks arrive -> bounded sync resolves within budget -> ncclSuccess.
TEST_F(GinBarrierTimeoutMPITest, HealthyBarrierReturnsSuccess)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    MPI_Barrier(MPI_COMM_WORLD);
    int hResult = runOneBarrier(devComm, stream, kHealthyTimeoutCycles);

    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess), hResult);
}

// Rank (nRanks-1) never enters -> waiting ranks' deadlines expire -> ncclTimeout.
TEST_F(GinBarrierTimeoutMPITest, AbsentPeerProducesTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    // Use rail-team rank to identify the absent rank so the test works for any
    // number of MPI ranks per node (ppr:1:node, ppr:2:node, etc.).
    // With >1 rank/node, multiple MPI ranks share the same rail rank; using
    // MPI world rank nRanks-1 as "absent" would leave other rail-rank-1 peers
    // still signaling, causing spurious barrier success.
    ncclTeam_t railTeam = ncclTeamRail(comm);
    const bool isAbsent = (railTeam.rank == railTeam.nRanks - 1);

    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runOneBarrier(devComm, stream, kShortTimeoutCycles);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "Rail rank " << railTeam.rank << " expected ncclTimeout from a stuck GIN barrier, got "
            << ncclGetErrorString(static_cast<ncclResult_t>(hResult));
        (void)hipStreamSynchronize(stream);
    }
    // Present rank is done (timed out). Now the absent rank catches up its IB counter
    // by signaling the present rank's slot, keeping counters in sync for future tests.
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        (void)runOneBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int local = (isAbsent || hResult == static_cast<int>(ncclTimeout)) ? 1 : 0;
    ASSERT_MPI_TRUE(local == 1);
}

// Corner case: zero-cycle budget + absent peer must time out immediately.
TEST_F(GinBarrierTimeoutMPITest, ZeroBudgetAbsentPeerTimesOut)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t railTeam = ncclTeamRail(comm);
    const bool isAbsent = (railTeam.rank == railTeam.nRanks - 1);

    int hResult = static_cast<int>(ncclSuccess);
    if (!isAbsent) {
        hResult = runOneBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        EXPECT_EQ(static_cast<int>(ncclTimeout), hResult)
            << "Rail rank " << railTeam.rank << " expected immediate ncclTimeout with zero budget, got "
            << ncclGetErrorString(static_cast<ncclResult_t>(hResult));
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        (void)runOneBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int local = (isAbsent || hResult == static_cast<int>(ncclTimeout)) ? 1 : 0;
    ASSERT_MPI_TRUE(local == 1);
}

// Stress: many sequential healthy barriers must all succeed (no drift / false trip).
TEST_F(GinBarrierTimeoutMPITest, RepeatedHealthyBarriersSucceed)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    constexpr int kIters = 32;
    int* dCount = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&dCount, sizeof(int)));
    SCOPE_EXIT(if (dCount) (void)hipFree(dCount));
    int zero = 0;
    HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(dCount, &zero, sizeof(int), hipMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    ginRepeatedBarrierKernel<<<1, 256, 0, stream>>>(devComm, kHealthyTimeoutCycles, kIters, dCount);
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

    int hCount = -1;
    HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(&hCount, dCount, sizeof(int), hipMemcpyDeviceToHost));
    ASSERT_MPI_EQ(kIters, hCount);
}

// Recovery: a timeout round (absent peer), then a fresh healthy barrier on a new
// devComm must succeed -- a tripped timeout must not wedge subsequent barriers.
TEST_F(GinBarrierTimeoutMPITest, RecoversAfterTimeout)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t railTeam = ncclTeamRail(comm);
    const bool isAbsent = (railTeam.rank == railTeam.nRanks - 1);

    if (!isAbsent) {
        int r1 = runOneBarrier(devComm, stream, kShortTimeoutCycles);
        EXPECT_EQ(static_cast<int>(ncclTimeout), r1);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (isAbsent) {
        // Catch up IB counter after present rank's timeout so slot 0 is
        // clean for the next createGinDevComm in round 2.
        (void)runOneBarrier(devComm, stream, /*timeoutCycles=*/0ULL);
        (void)hipStreamSynchronize(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Round 2: fresh devComm — all ranks participate, barrier should succeed.
    ncclDevComm devComm2{};
    ASSERT_MPI_EQ(ncclSuccess, createGinDevComm(comm, 1, &devComm2));
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm2));

    MPI_Barrier(MPI_COMM_WORLD);
    int r2 = runOneBarrier(devComm2, stream, kHealthyTimeoutCycles);
    ASSERT_MPI_EQ(static_cast<int>(ncclSuccess), r2);
}

// Stress: repeated stuck barriers each return ncclTimeout with no hang.
// Works for any ppr:N:node configuration — uses rail-team rank, not MPI rank,
// to identify the absent peer. The absent rank skips runOneBarrier entirely;
// each round uses a fresh devComm slot so there is no IB counter carry-over.
//
// Protocol per round (mirroring RecoversAfterTimeout which is known-good):
//   1. MPI_Barrier  -- all ranks enter the round together
//   2. createGinDevComm -- collective IB context setup
//   3. present rank: runOneBarrier(kShort) -> ncclTimeout
//      absent rank:  runOneBarrier(0)      -> ncclTimeout immediately
//      Both ranks call runOneBarrier so both execute the signal + shadow-ptr
//      increment, keeping IB counters in sync across rounds.
//   4. hipStreamSynchronize -- drain GPU pipeline on all ranks
//   5. MPI_Barrier  -- confirm all kernels retired before IB teardown
//   6. ncclDevCommDestroy -- safe: no in-flight GIN proxy ops remain
//   7. MPI_Barrier  -- confirm teardown complete before next createGinDevComm
TEST_F(GinBarrierTimeoutMPITest, BackToBackTimeouts)
{
    ncclComm_t comm{}; hipStream_t stream{}; ncclDevComm devComm{};
    if (!setUpGinDevComm(1, &comm, &stream, &devComm)) GIN_SETUP_OR_BAIL();
    SCOPE_EXIT((void)ncclDevCommDestroy(comm, &devComm));

    ncclTeam_t railTeam = ncclTeamRail(comm);
    const bool isAbsent = (railTeam.rank == railTeam.nRanks - 1);

    constexpr int kRounds = 4;
    int timeouts = 0;
    for (int i = 0; i < kRounds; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);  // gate round entry
        ncclDevComm dc{};
        if (createGinDevComm(comm, 1, &dc) != ncclSuccess) break;
        if (!isAbsent) {
            int r = runOneBarrier(dc, stream, kShortTimeoutCycles);
            if (r == static_cast<int>(ncclTimeout)) ++timeouts;
            (void)hipStreamSynchronize(stream);
        }
        MPI_Barrier(MPI_COMM_WORLD);  // present rank: kernel retired
        if (isAbsent) {
            // Catch up IB counter after present rank's timeout.
            (void)runOneBarrier(dc, stream, /*timeoutCycles=*/0ULL);
            (void)hipStreamSynchronize(stream);
        }
        MPI_Barrier(MPI_COMM_WORLD);  // all done, safe to destroy
        (void)ncclDevCommDestroy(comm, &dc);
        MPI_Barrier(MPI_COMM_WORLD);  // teardown complete
    }
    if (!isAbsent) EXPECT_EQ(kRounds, timeouts);
    MPI_Barrier(MPI_COMM_WORLD);

    int local = (isAbsent || timeouts == kRounds) ? 1 : 0;
    ASSERT_MPI_TRUE(local == 1);
}

#endif // MPI_TESTS_ENABLED
