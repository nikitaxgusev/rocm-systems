/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file VerbsAcceptanceMPITests.cpp
 * @brief Hardware-agnostic RDMA/verbs acceptance (regression) tests.
 *
 * Each test targets a specific low-level NIC/driver code path that RCCL's
 * network transport depends on (RC/UD QPs, DMABUF read/write/atomic over GPU
 * memory, write-with-immediate ordering/visibility, relaxed-ordering MRs, QP
 * restore, and the HIP-allocation x IB-registration matrix). A test that flips
 * from PASS to FAIL after a firmware or driver update is an unambiguous
 * regression signal.
 *
 * The suite is MPI-first: MPI is both the execution model and the OOB channel
 * (MPI_Sendrecv replaces the TCP coordination of the standalone reference). It
 * is vendor-agnostic: it auto-selects the first active IB device and SKIPs
 * (coordinated across ranks) whenever a capability is unavailable, rather than
 * failing.
 *
 * Scaling model: ranks are paired lower-half <-> upper-half. With world size N,
 * rank r's peer is r +/- N/2; the lower half acts as receiver ("server"), the
 * upper half as sender ("client"). This yields cross-node pairs for multi-node
 * launches (e.g. 16 ranks @ 8/node -> 0..7 <-> 8..15) and intra-node loopback
 * pairs for single-node launches.
 */

#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#ifdef MPI_TESTS_ENABLED

#include <infiniband/verbs.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <hip/hip_runtime.h>

#include <arpa/inet.h>
#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

namespace
{

// ---- Test parameters -------------------------------------------------------
constexpr size_t   kMsgSize       = 4096;
constexpr int      kCqDepth       = 32;
constexpr int      kQpDepth       = 16;
constexpr uint32_t kImmDone       = 0xD011EU;
constexpr int      kPollTimeoutMs = 5000;
constexpr int      kMinRanks      = 2;

// OOB tags for MPI_Sendrecv exchanges.
constexpr int kTagQpInfo   = 7001;
constexpr int kTagRemoteBuf = 7002;
constexpr int kTagResult   = 7003;
constexpr int kTagFlag     = 7004;

// ---- HIP allocation modes (7) ---------------------------------------------
enum AllocMode
{
    ALLOC_DEVICE = 0,
    ALLOC_HOST_DEFAULT,
    ALLOC_HOST_WC,
    ALLOC_HOST_COHERENT,
    ALLOC_HOST_NONCOHERENT,
    ALLOC_FINEGRAIN,
    ALLOC_MANAGED,
    ALLOC_COUNT
};

// ---- IB registration modes (3) --------------------------------------------
enum RegMode
{
    REG_STD = 0,
    REG_DMABUF_V1,
    REG_DMABUF_V2_PCIE,
    REG_COUNT
};

const char* allocName(AllocMode m)
{
    static const char* names[ALLOC_COUNT] = {"hipMalloc",
                                             "hipHostMalloc_Default",
                                             "hipHostMalloc_WC",
                                             "hipHostMalloc_Coherent",
                                             "hipHostMalloc_NonCoherent",
                                             "hipExtMalloc_FineGrain",
                                             "hipMallocManaged"};
    return names[m];
}

const char* regName(RegMode m)
{
    static const char* names[REG_COUNT] = {"ibv_reg_mr", "dmabuf_v1", "dmabuf_v2_pcie"};
    return names[m];
}

// ---- Optional symbols resolved at runtime (graceful skip if absent) --------
typedef hsa_status_t (*fn_export_dmabuf_v1_t)(const void*, size_t, int*, uint64_t*);
typedef hsa_status_t (*fn_export_dmabuf_v2_t)(const void*, size_t, int*, uint64_t*, uint64_t);
typedef struct ibv_mr* (*fn_reg_dmabuf_mr_t)(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int);

fn_export_dmabuf_v1_t g_dmabuf_v1  = nullptr;
fn_export_dmabuf_v2_t g_dmabuf_v2  = nullptr;
fn_reg_dmabuf_mr_t    g_reg_dmabuf = nullptr;
bool                  g_symbols_resolved = false;

void resolveOptionalSymbols()
{
    if(g_symbols_resolved)
        return;
    g_symbols_resolved = true;

    void* hsa = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if(hsa)
    {
        g_dmabuf_v1 = reinterpret_cast<fn_export_dmabuf_v1_t>(
            dlsym(hsa, "hsa_amd_portable_export_dmabuf"));
        g_dmabuf_v2 = reinterpret_cast<fn_export_dmabuf_v2_t>(
            dlsym(hsa, "hsa_amd_portable_export_dmabuf_v2"));
    }
    void* ibv = dlopen("libibverbs.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if(ibv)
        g_reg_dmabuf = reinterpret_cast<fn_reg_dmabuf_mr_t>(dlsym(ibv, "ibv_reg_dmabuf_mr"));
}

// Wire-format structs exchanged over MPI OOB.
struct QpInfo
{
    uint32_t qpn;
    uint32_t psn;
    uint16_t lid;
    uint8_t  gid[16];
};

struct RemoteBuf
{
    uint64_t addr;
    uint32_t rkey;
    uint32_t pad;
};

struct RegResult
{
    struct ibv_mr* mr     = nullptr;
    int            dma_fd = -1;
};

// Reduce a local boolean "supported/pass" flag across all ranks so every rank
// makes the same skip/continue decision and no collective deadlocks.
bool allRanksAgree(bool localOk)
{
    int local  = localOk ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return global != 0;
}

} // namespace

// ===========================================================================
// GPU poll kernel for Atomic_Wakeup_GPU_Stream: spin until *flag >= target or
// the iteration budget is exhausted, then publish what was observed.
// ===========================================================================
__global__ void verbsWaitOnValueKernel(volatile uint32_t* flag,
                                        uint32_t           target,
                                        uint64_t           maxIters,
                                        uint32_t*          observed)
{
    uint64_t i = 0;
    uint32_t v = 0;
    for(; i < maxIters; ++i)
    {
        v = flag[0];
        if(v >= target)
            break;
    }
    *observed = v;
}

// ===========================================================================
// Fixture
// ===========================================================================
class VerbsAcceptanceMPITest : public MPITestBase
{
protected:
    struct ibv_context* ctx_     = nullptr;
    struct ibv_pd*      pd_      = nullptr;
    struct ibv_cq*      cq_      = nullptr;
    struct ibv_qp*      qp_      = nullptr;
    int                 ibPort_  = 1;
    int                 gidIdx_  = 3; // RoCE v2 default; override via NCCL_IB_GID_INDEX
    QpInfo              localQp_ = {};

    int worldRank_ = 0;
    int worldSize_ = 0;
    int peerRank_  = -1;
    bool isLowerHalf_ = false; // lower half == receiver/server role

    struct ibv_device_attr devAttr_ = {};

    void SetUp() override
    {
        MPITestBase::SetUp();
        worldRank_ = MPIEnvironment::world_rank;
        worldSize_ = MPIEnvironment::world_size;

        if(const char* g = std::getenv("NCCL_IB_GID_INDEX"))
        {
            int v = std::atoi(g);
            if(v >= 0)
                gidIdx_ = v;
        }

        // Pairing: lower half <-> upper half.
        isLowerHalf_ = (worldRank_ < worldSize_ / 2);
        peerRank_    = isLowerHalf_ ? worldRank_ + worldSize_ / 2 : worldRank_ - worldSize_ / 2;
    }

    void TearDown() override
    {
        (void)hipDeviceSynchronize();
        destroyQp();
        if(cq_)
        {
            ibv_destroy_cq(cq_);
            cq_ = nullptr;
        }
        if(pd_)
        {
            ibv_dealloc_pd(pd_);
            pd_ = nullptr;
        }
        if(ctx_)
        {
            ibv_close_device(ctx_);
            ctx_ = nullptr;
        }
        MPITestBase::TearDown();
    }

    // Even world size and >= kMinRanks; coordinated skip otherwise.
    bool requireEvenPairs()
    {
        if(!validateTestPrerequisites(kMinRanks))
            return false;
        if((worldSize_ % 2) != 0)
            return false;
        return true;
    }

    // Open the first IB device with an ACTIVE port. Vendor-agnostic. Allocates
    // pd_ and cq_. Returns false (locally) if nothing usable; callers reduce.
    bool openIbDevice()
    {
        int           nDevs = 0;
        ibv_device**  devs  = ibv_get_device_list(&nDevs);
        if(!devs || nDevs == 0)
            return false;
        SCOPE_EXIT(ibv_free_device_list(devs));

        for(int i = 0; i < nDevs && !ctx_; ++i)
        {
            ibv_context* c = ibv_open_device(devs[i]);
            if(!c)
                continue;
            ibv_port_attr pa{};
            if(ibv_query_port(c, ibPort_, &pa) == 0 && pa.state == IBV_PORT_ACTIVE)
            {
                ctx_ = c;
            }
            else
            {
                ibv_close_device(c);
            }
        }
        if(!ctx_)
            return false;

        if(ibv_query_device(ctx_, &devAttr_) != 0)
            return false;

        pd_ = ibv_alloc_pd(ctx_);
        if(!pd_)
            return false;
        cq_ = ibv_create_cq(ctx_, kCqDepth, nullptr, nullptr, 0);
        if(!cq_)
            return false;
        return true;
    }

    // Create a QP of the given type in INIT state, capture local QpInfo.
    bool createQp(enum ibv_qp_type type)
    {
        ibv_qp_init_attr qia{};
        qia.send_cq          = cq_;
        qia.recv_cq          = cq_;
        qia.qp_type          = type;
        qia.cap.max_send_wr  = kQpDepth;
        qia.cap.max_recv_wr  = kQpDepth;
        qia.cap.max_send_sge = 1;
        qia.cap.max_recv_sge = 1;
        qp_                  = ibv_create_qp(pd_, &qia);
        if(!qp_)
            return false;

        ibv_qp_attr qa{};
        qa.qp_state   = IBV_QPS_INIT;
        qa.pkey_index = 0;
        qa.port_num   = static_cast<uint8_t>(ibPort_);
        int flags     = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT;
        if(type == IBV_QPT_RC)
        {
            qa.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                                 | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
            flags |= IBV_QP_ACCESS_FLAGS;
        }
        else // UD
        {
            qa.qkey = 0x11111111U;
            flags |= IBV_QP_QKEY;
        }
        if(ibv_modify_qp(qp_, &qa, flags) != 0)
            return false;

        ibv_port_attr pa{};
        ibv_query_port(ctx_, ibPort_, &pa);
        localQp_.qpn = qp_->qp_num;
        localQp_.psn = static_cast<uint32_t>(lrand48() & 0xFFFFFF);
        localQp_.lid = pa.lid;
        ibv_gid gid{};
        ibv_query_gid(ctx_, ibPort_, gidIdx_, &gid);
        std::memcpy(localQp_.gid, gid.raw, 16);
        return true;
    }

    void destroyQp()
    {
        if(qp_)
        {
            ibv_destroy_qp(qp_);
            qp_ = nullptr;
        }
    }

    QpInfo exchangeQpInfo()
    {
        QpInfo remote{};
        MPI_Sendrecv(&localQp_, sizeof(QpInfo), MPI_BYTE, peerRank_, kTagQpInfo,
                     &remote, sizeof(QpInfo), MPI_BYTE, peerRank_, kTagQpInfo,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return remote;
    }

    // Transition an RC QP INIT->RTR->RTS given the remote QpInfo.
    bool rcToRtrRts(const QpInfo& remote)
    {
        ibv_qp_attr qa{};
        qa.qp_state           = IBV_QPS_RTR;
        qa.path_mtu           = IBV_MTU_1024;
        qa.dest_qp_num        = remote.qpn;
        qa.rq_psn             = remote.psn;
        qa.max_dest_rd_atomic = 1;
        qa.min_rnr_timer      = 12;
        qa.ah_attr.is_global  = 1;
        qa.ah_attr.dlid       = remote.lid;
        qa.ah_attr.port_num   = static_cast<uint8_t>(ibPort_);
        std::memcpy(qa.ah_attr.grh.dgid.raw, remote.gid, 16);
        qa.ah_attr.grh.sgid_index = static_cast<uint8_t>(gidIdx_);
        qa.ah_attr.grh.hop_limit  = 64;
        if(ibv_modify_qp(qp_, &qa,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN
                             | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)
           != 0)
            return false;

        std::memset(&qa, 0, sizeof(qa));
        qa.qp_state      = IBV_QPS_RTS;
        qa.sq_psn        = localQp_.psn;
        qa.timeout       = 14;
        qa.retry_cnt     = 7;
        qa.rnr_retry     = 7;
        qa.max_rd_atomic = 1;
        if(ibv_modify_qp(qp_, &qa,
                         IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT
                             | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC)
           != 0)
            return false;
        return true;
    }

    // Bring a UD QP to RTS (no remote address needed for state machine).
    bool udToRts()
    {
        ibv_qp_attr qa{};
        qa.qp_state = IBV_QPS_RTR;
        if(ibv_modify_qp(qp_, &qa, IBV_QP_STATE) != 0)
            return false;
        std::memset(&qa, 0, sizeof(qa));
        qa.qp_state = IBV_QPS_RTS;
        qa.sq_psn   = localQp_.psn;
        if(ibv_modify_qp(qp_, &qa, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0)
            return false;
        return true;
    }

    // Allocate a buffer using the given HIP allocation mode.
    void* allocBuf(AllocMode mode, size_t size)
    {
        void*      ptr = nullptr;
        hipError_t e   = hipSuccess;
        switch(mode)
        {
            case ALLOC_DEVICE: e = hipMalloc(&ptr, size); break;
            case ALLOC_HOST_DEFAULT: e = hipHostMalloc(&ptr, size, hipHostMallocDefault); break;
            case ALLOC_HOST_WC: e = hipHostMalloc(&ptr, size, hipHostMallocWriteCombined); break;
            case ALLOC_HOST_COHERENT: e = hipHostMalloc(&ptr, size, hipHostMallocCoherent); break;
            case ALLOC_HOST_NONCOHERENT:
                e = hipHostMalloc(&ptr, size, hipHostMallocNonCoherent);
                break;
            case ALLOC_FINEGRAIN:
                e = hipExtMallocWithFlags(&ptr, size, hipDeviceMallocFinegrained);
                break;
            case ALLOC_MANAGED: e = hipMallocManaged(&ptr, size, hipMemAttachGlobal); break;
            default: return nullptr;
        }
        return (e == hipSuccess) ? ptr : nullptr;
    }

    void freeBuf(AllocMode mode, void* ptr)
    {
        if(!ptr)
            return;
        switch(mode)
        {
            case ALLOC_DEVICE:
            case ALLOC_FINEGRAIN:
            case ALLOC_MANAGED: (void)hipFree(ptr); break;
            default: (void)hipHostFree(ptr); break;
        }
    }

    RegResult regBuf(void* ptr, size_t size, RegMode mode, int access)
    {
        RegResult r;
        if(mode == REG_STD)
        {
            r.mr = ibv_reg_mr(pd_, ptr, size, access);
            return r;
        }
        if(!g_reg_dmabuf)
            return r;
        uint64_t     offset = 0;
        hsa_status_t hs;
        if(mode == REG_DMABUF_V1)
        {
            if(!g_dmabuf_v1)
                return r;
            hs = g_dmabuf_v1(ptr, size, &r.dma_fd, &offset);
        }
        else
        {
            if(!g_dmabuf_v2)
                return r;
            hs = g_dmabuf_v2(ptr, size, &r.dma_fd, &offset, HSA_AMD_DMABUF_MAPPING_TYPE_PCIE);
        }
        if(hs != HSA_STATUS_SUCCESS)
        {
            r.dma_fd = -1;
            return r;
        }
        r.mr = g_reg_dmabuf(pd_, offset, size, reinterpret_cast<uint64_t>(ptr), r.dma_fd, access);
        if(!r.mr)
        {
            close(r.dma_fd);
            r.dma_fd = -1;
        }
        return r;
    }

    void deregBuf(RegResult& r)
    {
        if(r.mr)
            ibv_dereg_mr(r.mr);
        if(r.dma_fd >= 0)
            close(r.dma_fd);
        r = {};
    }

    void postRecv(struct ibv_mr* mr, void* buf, size_t len, uint64_t wrId)
    {
        ibv_sge sge{};
        sge.addr   = reinterpret_cast<uint64_t>(buf);
        sge.length = static_cast<uint32_t>(len);
        sge.lkey   = mr->lkey;
        ibv_recv_wr wr{};
        wr.wr_id   = wrId;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        ibv_recv_wr* bad = nullptr;
        ibv_post_recv(qp_, &wr, &bad);
    }

    // Poll the CQ until a completion with wr_id arrives, or timeout. On a
    // non-success completion, fills wc and returns false.
    bool pollCq(uint64_t wrId, ibv_wc& wc, int timeoutMs)
    {
        for(int elapsed = 0; elapsed < timeoutMs; ++elapsed)
        {
            int n = ibv_poll_cq(cq_, 1, &wc);
            if(n < 0)
                return false;
            if(n > 0)
            {
                if(wc.status != IBV_WC_SUCCESS)
                    return false;
                if(wc.wr_id == wrId)
                    return true;
            }
            usleep(1000);
        }
        return false;
    }

    void fillPatternGpu(void* ptr, size_t size, uint8_t seed)
    {
        std::vector<uint8_t> tmp(size);
        for(size_t i = 0; i < size; ++i)
            tmp[i] = static_cast<uint8_t>((seed + i) % 256);
        (void)hipMemcpy(ptr, tmp.data(), size, hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
    }

    bool verifyPatternGpu(const void* ptr, size_t size, uint8_t seed)
    {
        std::vector<uint8_t> tmp(size);
        (void)hipMemcpy(tmp.data(), ptr, size, hipMemcpyDeviceToHost);
        (void)hipDeviceSynchronize();
        for(size_t i = 0; i < size; ++i)
            if(tmp[i] != static_cast<uint8_t>((seed + i) % 256))
                return false;
        return true;
    }

    // Post a single RDMA Write (optionally with immediate) from local buf to a
    // remote address, signaled, and wait for local completion.
    bool doRdmaWrite(void*          localBuf,
                     struct ibv_mr* localMr,
                     uint64_t       remoteAddr,
                     uint32_t       remoteRkey,
                     size_t         len,
                     bool           withImm,
                     uint32_t       immValue,
                     uint64_t       wrId)
    {
        ibv_sge sge{};
        sge.addr   = reinterpret_cast<uint64_t>(localBuf);
        sge.length = static_cast<uint32_t>(len);
        sge.lkey   = localMr->lkey;
        ibv_send_wr wr{};
        wr.wr_id              = wrId;
        wr.opcode            = withImm ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
        wr.send_flags        = IBV_SEND_SIGNALED;
        wr.sg_list           = &sge;
        wr.num_sge           = 1;
        if(withImm)
            wr.imm_data = htonl(immValue);
        wr.wr.rdma.remote_addr = remoteAddr;
        wr.wr.rdma.rkey        = remoteRkey;
        ibv_send_wr* bad       = nullptr;
        if(ibv_post_send(qp_, &wr, &bad) != 0)
            return false;
        ibv_wc wc{};
        return pollCq(wrId, wc, kPollTimeoutMs);
    }
};

// ===========================================================================
// Test 1: UD_RC_Support -- baseline. Can the NIC create both RC and UD QPs and
// bring each to RTS?
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, UD_RC_Support)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";

    resolveOptionalSymbols();
    bool ok = openIbDevice();
    if(!allRanksAgree(ok))
        GTEST_SKIP() << "No active IB device on all ranks";

    // RC -- reduce the create result before any OOB exchange so paired ranks
    // never block in MPI_Sendrecv when one side fails to create a QP.
    bool rc = createQp(IBV_QPT_RC);
    if(!allRanksAgree(rc))
    {
        destroyQp();
        GTEST_SKIP() << "RC QP creation not supported on all ranks";
    }
    QpInfo remote = exchangeQpInfo();
    rc            = rcToRtrRts(remote);
    destroyQp();
    ASSERT_MPI_TRUE(rc);

    // UD
    bool ud = createQp(IBV_QPT_UD);
    if(!allRanksAgree(ud))
    {
        destroyQp();
        GTEST_SKIP() << "UD QP not supported on all ranks";
    }
    (void)exchangeQpInfo(); // keep OOB symmetric across ranks
    ud = udToRts();
    destroyQp();
    ASSERT_MPI_TRUE(ud);
}

// ===========================================================================
// Test 2/3: RDMA Write / Read over DMABUF-registered GPU memory. Each runs once
// per available dmabuf version. Lower half = receiver/target; upper = initiator.
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, RdmaWrite_DMABUF)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";
    if(!allRanksAgree(g_reg_dmabuf != nullptr && (g_dmabuf_v1 || g_dmabuf_v2)))
        GTEST_SKIP() << "DMABUF registration not available";

    const int      access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                       | IBV_ACCESS_REMOTE_READ;
    bool ranAny = false;

    for(RegMode reg : {REG_DMABUF_V1, REG_DMABUF_V2_PCIE})
    {
        bool haveSym = (reg == REG_DMABUF_V1) ? (g_dmabuf_v1 != nullptr)
                                              : (g_dmabuf_v2 != nullptr);
        if(!allRanksAgree(haveSym))
            continue;

        ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
        QpInfo remote = exchangeQpInfo();
        ASSERT_MPI_TRUE(rcToRtrRts(remote));

        void* buf = allocBuf(ALLOC_DEVICE, kMsgSize);
        ASSERT_MPI_TRUE(buf != nullptr);
        SCOPE_EXIT(freeBuf(ALLOC_DEVICE, buf));
        RegResult rr = regBuf(buf, kMsgSize, reg, access);
        if(!allRanksAgree(rr.mr != nullptr))
        {
            deregBuf(rr);
            destroyQp();
            continue;
        }
        SCOPE_EXIT(deregBuf(rr));

        const uint8_t seed = static_cast<uint8_t>(0xA0 + static_cast<int>(reg));
        bool          pass = false;
        if(isLowerHalf_)
        {
            // Target: clear buffer, advertise it, wait for the data to land.
            (void)hipMemset(buf, 0, kMsgSize);
            (void)hipDeviceSynchronize();
            RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
            MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
            uint8_t go = 0;
            MPI_Recv(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pass = verifyPatternGpu(buf, kMsgSize, seed);
            uint8_t res = pass ? 1 : 0;
            MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
        }
        else
        {
            RemoteBuf rb{};
            MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            fillPatternGpu(buf, kMsgSize, seed);
            bool wrote = doRdmaWrite(buf, rr.mr, rb.addr, rb.rkey, kMsgSize, false, 0, 1);
            uint8_t go = 1;
            MPI_Send(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD);
            uint8_t res = 0;
            MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pass = wrote && (res == 1);
        }
        deregBuf(rr);
        destroyQp();
        ranAny = true;
        ASSERT_MPI_TRUE(pass);
    }

    if(!allRanksAgree(ranAny))
        GTEST_SKIP() << "No DMABUF version exercised";
}

TEST_F(VerbsAcceptanceMPITest, RdmaRead_DMABUF)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";
    if(!allRanksAgree(g_reg_dmabuf != nullptr && (g_dmabuf_v1 || g_dmabuf_v2)))
        GTEST_SKIP() << "DMABUF registration not available";

    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    bool      ranAny = false;

    for(RegMode reg : {REG_DMABUF_V1, REG_DMABUF_V2_PCIE})
    {
        bool haveSym = (reg == REG_DMABUF_V1) ? (g_dmabuf_v1 != nullptr) : (g_dmabuf_v2 != nullptr);
        if(!allRanksAgree(haveSym))
            continue;

        ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
        QpInfo remote = exchangeQpInfo();
        ASSERT_MPI_TRUE(rcToRtrRts(remote));

        void* buf = allocBuf(ALLOC_DEVICE, kMsgSize);
        ASSERT_MPI_TRUE(buf != nullptr);
        SCOPE_EXIT(freeBuf(ALLOC_DEVICE, buf));
        RegResult rr = regBuf(buf, kMsgSize, reg, access);
        if(!allRanksAgree(rr.mr != nullptr))
        {
            deregBuf(rr);
            destroyQp();
            continue;
        }

        const uint8_t seed = static_cast<uint8_t>(0xB0 + static_cast<int>(reg));
        bool          pass = false;
        if(isLowerHalf_)
        {
            // Target holds the source data; initiator reads it.
            fillPatternGpu(buf, kMsgSize, seed);
            RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
            MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
            uint8_t res = 0;
            MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pass = (res == 1);
        }
        else
        {
            RemoteBuf rb{};
            MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            (void)hipMemset(buf, 0, kMsgSize);
            (void)hipDeviceSynchronize();
            ibv_sge sge{};
            sge.addr   = reinterpret_cast<uint64_t>(buf);
            sge.length = static_cast<uint32_t>(kMsgSize);
            sge.lkey   = rr.mr->lkey;
            ibv_send_wr wr{};
            wr.wr_id               = 2;
            wr.opcode              = IBV_WR_RDMA_READ;
            wr.send_flags          = IBV_SEND_SIGNALED;
            wr.sg_list             = &sge;
            wr.num_sge             = 1;
            wr.wr.rdma.remote_addr = rb.addr;
            wr.wr.rdma.rkey        = rb.rkey;
            ibv_send_wr* bad       = nullptr;
            bool         ok        = (ibv_post_send(qp_, &wr, &bad) == 0);
            if(ok)
            {
                ibv_wc wc{};
                ok = pollCq(2, wc, kPollTimeoutMs);
            }
            pass        = ok && verifyPatternGpu(buf, kMsgSize, seed);
            uint8_t res = pass ? 1 : 0;
            MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
        }
        deregBuf(rr);
        destroyQp();
        ranAny = true;
        ASSERT_MPI_TRUE(pass);
    }

    if(!allRanksAgree(ranAny))
        GTEST_SKIP() << "No DMABUF version exercised";
}

// ===========================================================================
// Test 4: Atomic_DMABUF -- Fetch-and-Add over GPU memory via DMABUF.
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, Atomic_DMABUF)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";
    if(!allRanksAgree(devAttr_.atomic_cap != IBV_ATOMIC_NONE))
        GTEST_SKIP() << "NIC reports no atomic capability";
    if(!allRanksAgree(g_reg_dmabuf != nullptr && (g_dmabuf_v1 || g_dmabuf_v2)))
        GTEST_SKIP() << "DMABUF registration not available";

    RegMode reg = g_dmabuf_v2 ? REG_DMABUF_V2_PCIE : REG_DMABUF_V1;
    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
    QpInfo remote = exchangeQpInfo();
    ASSERT_MPI_TRUE(rcToRtrRts(remote));

    // 8-byte aligned atomic target.
    void* buf = allocBuf(ALLOC_DEVICE, kMsgSize);
    ASSERT_MPI_TRUE(buf != nullptr);
    SCOPE_EXIT(freeBuf(ALLOC_DEVICE, buf));
    RegResult rr = regBuf(buf, kMsgSize, reg, access);
    if(!allRanksAgree(rr.mr != nullptr))
    {
        deregBuf(rr);
        GTEST_SKIP() << "DMABUF atomic MR registration failed";
    }
    SCOPE_EXIT(deregBuf(rr));

    const uint64_t kInit = 0x1000;
    const uint64_t kAdd  = 0x25;
    bool           pass  = false;

    if(isLowerHalf_)
    {
        uint64_t init = kInit;
        (void)hipMemcpy(buf, &init, sizeof(init), hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
        RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
        MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
        uint8_t go = 0;
        MPI_Recv(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        uint64_t observed = 0;
        (void)hipMemcpy(&observed, buf, sizeof(observed), hipMemcpyDeviceToHost);
        (void)hipDeviceSynchronize();
        pass        = (observed == kInit + kAdd);
        uint8_t res = pass ? 1 : 0;
        MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
    }
    else
    {
        RemoteBuf rb{};
        MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        // Local result landing zone for the fetched value; needs its own MR.
        void* res_buf = allocBuf(ALLOC_DEVICE, sizeof(uint64_t));
        SCOPE_EXIT(freeBuf(ALLOC_DEVICE, res_buf));
        RegResult resReg = res_buf ? regBuf(res_buf, sizeof(uint64_t), reg, access) : RegResult{};
        bool      ok     = (res_buf != nullptr) && (resReg.mr != nullptr);
        SCOPE_EXIT(deregBuf(resReg));

        if(ok)
        {
            ibv_sge sge{};
            sge.addr   = reinterpret_cast<uint64_t>(res_buf);
            sge.length = sizeof(uint64_t);
            sge.lkey   = resReg.mr->lkey;
            ibv_send_wr wr{};
            wr.wr_id                  = 3;
            wr.opcode                 = IBV_WR_ATOMIC_FETCH_AND_ADD;
            wr.send_flags             = IBV_SEND_SIGNALED;
            wr.sg_list                = &sge;
            wr.num_sge                = 1;
            wr.wr.atomic.remote_addr  = rb.addr;
            wr.wr.atomic.rkey         = rb.rkey;
            wr.wr.atomic.compare_add  = kAdd;
            ibv_send_wr* bad          = nullptr;
            ok                        = (ibv_post_send(qp_, &wr, &bad) == 0);
            if(ok)
            {
                ibv_wc wc{};
                ok = pollCq(3, wc, kPollTimeoutMs);
            }
        }
        uint8_t go = 1;
        MPI_Send(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD);
        uint8_t res = 0;
        MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pass = ok && (res == 1);
    }

    deregBuf(rr);
    destroyQp();
    ASSERT_MPI_TRUE(pass);
}

// ===========================================================================
// Test 5: WriteWithImmediate -- imm_data arrives and ordering is preserved
// (payload visible before the imm completion is observed).
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, WriteWithImmediate)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";

    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
    QpInfo remote = exchangeQpInfo();
    ASSERT_MPI_TRUE(rcToRtrRts(remote));

    void* buf = allocBuf(ALLOC_DEVICE, kMsgSize);
    ASSERT_MPI_TRUE(buf != nullptr);
    SCOPE_EXIT(freeBuf(ALLOC_DEVICE, buf));
    RegResult rr = regBuf(buf, kMsgSize, REG_STD, access);
    ASSERT_MPI_TRUE(rr.mr != nullptr);
    SCOPE_EXIT(deregBuf(rr));

    const uint8_t seed = 0x5A;
    bool          pass = false;

    if(isLowerHalf_)
    {
        (void)hipMemset(buf, 0, kMsgSize);
        (void)hipDeviceSynchronize();
        postRecv(rr.mr, buf, kMsgSize, 100);
        RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
        MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
        ibv_wc wc{};
        bool   got = pollCq(100, wc, kPollTimeoutMs);
        bool   immOk = got && wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM
                   && (wc.wc_flags & IBV_WC_WITH_IMM) && ntohl(wc.imm_data) == kImmDone;
        bool dataOk = verifyPatternGpu(buf, kMsgSize, seed);
        pass        = immOk && dataOk;
        uint8_t res = pass ? 1 : 0;
        MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
    }
    else
    {
        RemoteBuf rb{};
        MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        fillPatternGpu(buf, kMsgSize, seed);
        bool wrote = doRdmaWrite(buf, rr.mr, rb.addr, rb.rkey, kMsgSize, true, kImmDone, 4);
        uint8_t res = 0;
        MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pass = wrote && (res == 1);
    }

    deregBuf(rr);
    destroyQp();
    ASSERT_MPI_TRUE(pass);
}

// ===========================================================================
// Test 6: RC_QP_Restore -- after forcing the QP to error, transition it back
// (Err -> Reset -> Init -> RTR -> RTS) and resume data transfer.
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, RC_QP_Restore)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";

    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
    QpInfo remote = exchangeQpInfo();
    ASSERT_MPI_TRUE(rcToRtrRts(remote));

    // Force QP to error state, then drive it back to RESET and re-arm.
    bool restored = false;
    {
        ibv_qp_attr qa{};
        qa.qp_state = IBV_QPS_ERR;
        (void)ibv_modify_qp(qp_, &qa, IBV_QP_STATE);

        std::memset(&qa, 0, sizeof(qa));
        qa.qp_state = IBV_QPS_RESET;
        bool toReset = (ibv_modify_qp(qp_, &qa, IBV_QP_STATE) == 0);

        // Re-INIT
        bool toInit = false;
        if(toReset)
        {
            std::memset(&qa, 0, sizeof(qa));
            qa.qp_state        = IBV_QPS_INIT;
            qa.pkey_index      = 0;
            qa.port_num        = static_cast<uint8_t>(ibPort_);
            qa.qp_access_flags = access;
            toInit = (ibv_modify_qp(qp_, &qa,
                                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT
                                        | IBV_QP_ACCESS_FLAGS)
                      == 0);
        }

        // New PSN, re-exchange, re-arm RTR/RTS.
        if(toInit)
        {
            localQp_.psn  = static_cast<uint32_t>(lrand48() & 0xFFFFFF);
            QpInfo remote2 = exchangeQpInfo();
            restored       = rcToRtrRts(remote2);
        }
        else
        {
            (void)exchangeQpInfo(); // keep OOB symmetric
        }
    }
    if(!allRanksAgree(restored))
        GTEST_SKIP() << "QP restore path not supported on all ranks";

    // Verify the restored QP can move data.
    void* buf = allocBuf(ALLOC_DEVICE, kMsgSize);
    ASSERT_MPI_TRUE(buf != nullptr);
    SCOPE_EXIT(freeBuf(ALLOC_DEVICE, buf));
    RegResult rr = regBuf(buf, kMsgSize, REG_STD, access);
    ASSERT_MPI_TRUE(rr.mr != nullptr);
    SCOPE_EXIT(deregBuf(rr));

    const uint8_t seed = 0x6C;
    bool          pass = false;
    if(isLowerHalf_)
    {
        (void)hipMemset(buf, 0, kMsgSize);
        (void)hipDeviceSynchronize();
        RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
        MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
        uint8_t go = 0;
        MPI_Recv(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pass        = verifyPatternGpu(buf, kMsgSize, seed);
        uint8_t res = pass ? 1 : 0;
        MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
    }
    else
    {
        RemoteBuf rb{};
        MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        fillPatternGpu(buf, kMsgSize, seed);
        bool    wrote = doRdmaWrite(buf, rr.mr, rb.addr, rb.rkey, kMsgSize, false, 0, 5);
        uint8_t go    = 1;
        MPI_Send(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD);
        uint8_t res = 0;
        MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pass = wrote && (res == 1);
    }

    deregBuf(rr);
    destroyQp();
    ASSERT_MPI_TRUE(pass);
}

// ===========================================================================
// Test 7: HipAlloc_x_IbvReg -- parametrized 7 alloc modes x 3 reg modes = 21
// combinations, each an RDMA Write+Immediate over RC. Each combo is a separate
// result line.
// ===========================================================================
class VerbsHipAllocRegTest
    : public VerbsAcceptanceMPITest
    , public ::testing::WithParamInterface<std::tuple<AllocMode, RegMode>>
{
};

TEST_P(VerbsHipAllocRegTest, WriteImmRoundTrip)
{
    const AllocMode amode = std::get<0>(GetParam());
    const RegMode   rmode = std::get<1>(GetParam());

    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";

    if(rmode != REG_STD)
    {
        bool haveSym = (rmode == REG_DMABUF_V1) ? (g_dmabuf_v1 != nullptr)
                                                : (g_dmabuf_v2 != nullptr);
        if(!allRanksAgree(g_reg_dmabuf != nullptr && haveSym))
            GTEST_SKIP() << "DMABUF registration (" << regName(rmode) << ") not available";
    }

    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    // Allocation may legitimately be unsupported on some hardware -> skip.
    void* buf = allocBuf(amode, kMsgSize);
    if(!allRanksAgree(buf != nullptr))
    {
        if(buf)
            freeBuf(amode, buf);
        GTEST_SKIP() << "Allocation (" << allocName(amode) << ") not available on all ranks";
    }
    SCOPE_EXIT(freeBuf(amode, buf));

    RegResult rr = regBuf(buf, kMsgSize, rmode, access);
    if(!allRanksAgree(rr.mr != nullptr))
    {
        deregBuf(rr);
        GTEST_SKIP() << "Registration (" << allocName(amode) << " x " << regName(rmode)
                     << ") not supported on all ranks";
    }
    SCOPE_EXIT(deregBuf(rr));

    ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
    QpInfo remote = exchangeQpInfo();
    ASSERT_MPI_TRUE(rcToRtrRts(remote));

    const uint8_t seed = static_cast<uint8_t>(amode * REG_COUNT + rmode);
    bool          pass = false;

    if(isLowerHalf_)
    {
        (void)hipMemset(buf, 0, kMsgSize);
        (void)hipDeviceSynchronize();
        postRecv(rr.mr, buf, kMsgSize, 200);
        RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
        MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
        ibv_wc wc{};
        bool   got = pollCq(200, wc, kPollTimeoutMs);
        pass = got && wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM && verifyPatternGpu(buf, kMsgSize, seed);
        uint8_t res = pass ? 1 : 0;
        MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
    }
    else
    {
        RemoteBuf rb{};
        MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        fillPatternGpu(buf, kMsgSize, seed);
        bool    wrote = doRdmaWrite(buf, rr.mr, rb.addr, rb.rkey, kMsgSize, true, kImmDone, 6);
        uint8_t res   = 0;
        MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pass = wrote && (res == 1);
    }

    deregBuf(rr);
    destroyQp();
    ASSERT_MPI_TRUE(pass);
}

INSTANTIATE_TEST_SUITE_P(
    AllocXReg,
    VerbsHipAllocRegTest,
    ::testing::Combine(::testing::Values(ALLOC_DEVICE,
                                         ALLOC_HOST_DEFAULT,
                                         ALLOC_HOST_WC,
                                         ALLOC_HOST_COHERENT,
                                         ALLOC_HOST_NONCOHERENT,
                                         ALLOC_FINEGRAIN,
                                         ALLOC_MANAGED),
                       ::testing::Values(REG_STD, REG_DMABUF_V1, REG_DMABUF_V2_PCIE)),
    [](const ::testing::TestParamInfo<std::tuple<AllocMode, RegMode>>& info) {
        return std::string(allocName(std::get<0>(info.param))) + "__"
               + regName(std::get<1>(info.param));
    });

// ===========================================================================
// Test 8: Atomic_Wakeup_GPU_Stream -- a GPU kernel polls a value; a remote NIC
// atomic-adds to it; verify the kernel observes the update without a CPU-side
// barrier.
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, Atomic_Wakeup_GPU_Stream)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";
    if(!allRanksAgree(devAttr_.atomic_cap != IBV_ATOMIC_NONE))
        GTEST_SKIP() << "NIC reports no atomic capability";

    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
    QpInfo remote = exchangeQpInfo();
    ASSERT_MPI_TRUE(rcToRtrRts(remote));

    // Atomic target (fine-grained device memory so the GPU kernel sees NIC
    // writes coherently).
    void* flag = allocBuf(ALLOC_FINEGRAIN, sizeof(uint64_t));
    if(!allRanksAgree(flag != nullptr))
    {
        if(flag)
            freeBuf(ALLOC_FINEGRAIN, flag);
        GTEST_SKIP() << "Fine-grained allocation not available";
    }
    SCOPE_EXIT(freeBuf(ALLOC_FINEGRAIN, flag));
    RegResult rr = regBuf(flag, sizeof(uint64_t), REG_STD, access);
    ASSERT_MPI_TRUE(rr.mr != nullptr);
    SCOPE_EXIT(deregBuf(rr));

    const uint64_t kAdd = 7;
    bool           pass = false;

    if(isLowerHalf_)
    {
        // Receiver: launch a polling kernel that waits for the value to reach kAdd.
        uint64_t zero = 0;
        (void)hipMemcpy(flag, &zero, sizeof(zero), hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();

        uint32_t* observed = nullptr;
        bool      setup    = (hipMalloc(&observed, sizeof(uint32_t)) == hipSuccess);
        SCOPE_EXIT(if(observed)(void) hipFree(observed));
        if(setup)
            (void)hipMemset(observed, 0, sizeof(uint32_t));

        hipStream_t stream = nullptr;
        if(setup)
            setup = (hipStreamCreate(&stream) == hipSuccess);
        SCOPE_EXIT(if(stream)(void) hipStreamDestroy(stream));

        if(setup)
        {
            // Kernel polls the low 32 bits of the atomic flag.
            verbsWaitOnValueKernel<<<1, 1, 0, stream>>>(
                reinterpret_cast<volatile uint32_t*>(flag), static_cast<uint32_t>(kAdd),
                5000000000ULL, observed);
        }

        // Advertise the target and let the sender fire the atomic.
        RemoteBuf rb{reinterpret_cast<uint64_t>(flag), rr.mr->rkey, 0};
        MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);

        bool     kdone   = setup && (hipStreamSynchronize(stream) == hipSuccess);
        uint32_t obsHost = 0;
        if(setup)
        {
            (void)hipMemcpy(&obsHost, observed, sizeof(obsHost), hipMemcpyDeviceToHost);
            (void)hipDeviceSynchronize();
        }
        pass = kdone && (obsHost >= static_cast<uint32_t>(kAdd));

        uint8_t res = pass ? 1 : 0;
        MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
    }
    else
    {
        RemoteBuf rb{};
        MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        void* res_buf = allocBuf(ALLOC_DEVICE, sizeof(uint64_t));
        SCOPE_EXIT(freeBuf(ALLOC_DEVICE, res_buf));
        RegResult resReg = res_buf ? regBuf(res_buf, sizeof(uint64_t), REG_STD, access) : RegResult{};
        bool      ok     = (res_buf != nullptr) && (resReg.mr != nullptr);
        SCOPE_EXIT(deregBuf(resReg));

        if(ok)
        {
            ibv_sge sge{};
            sge.addr   = reinterpret_cast<uint64_t>(res_buf);
            sge.length = sizeof(uint64_t);
            sge.lkey   = resReg.mr->lkey;
            ibv_send_wr wr{};
            wr.wr_id                 = 7;
            wr.opcode                = IBV_WR_ATOMIC_FETCH_AND_ADD;
            wr.send_flags            = IBV_SEND_SIGNALED;
            wr.sg_list               = &sge;
            wr.num_sge               = 1;
            wr.wr.atomic.remote_addr = rb.addr;
            wr.wr.atomic.rkey        = rb.rkey;
            wr.wr.atomic.compare_add = kAdd;
            ibv_send_wr* bad         = nullptr;
            ok                       = (ibv_post_send(qp_, &wr, &bad) == 0);
            if(ok)
            {
                ibv_wc wc{};
                ok = pollCq(7, wc, kPollTimeoutMs);
            }
        }
        uint8_t res = 0;
        MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pass = ok && (res == 1);
    }

    deregBuf(rr);
    destroyQp();
    ASSERT_MPI_TRUE(pass);
}

// ===========================================================================
// Test 9: Write_Then_Atomic_Visibility -- data written via RDMA Write must be
// visible in GPU memory by the time a subsequent atomic increment is observed.
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, Write_Then_Atomic_Visibility)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";
    if(!allRanksAgree(devAttr_.atomic_cap != IBV_ATOMIC_NONE))
        GTEST_SKIP() << "NIC reports no atomic capability";

    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
    QpInfo remote = exchangeQpInfo();
    ASSERT_MPI_TRUE(rcToRtrRts(remote));

    // Layout: [0 .. kMsgSize) payload, [kMsgSize .. +8) counter (8B aligned).
    const size_t totalSize   = kMsgSize + 64;
    const size_t counterOff  = kMsgSize;
    void*        buf         = allocBuf(ALLOC_DEVICE, totalSize);
    ASSERT_MPI_TRUE(buf != nullptr);
    SCOPE_EXIT(freeBuf(ALLOC_DEVICE, buf));
    RegResult rr = regBuf(buf, totalSize, REG_STD, access);
    ASSERT_MPI_TRUE(rr.mr != nullptr);
    SCOPE_EXIT(deregBuf(rr));

    const uint8_t seed = 0x9D;
    bool          pass = false;

    if(isLowerHalf_)
    {
        (void)hipMemset(buf, 0, totalSize);
        (void)hipDeviceSynchronize();
        RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
        MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);

        // Spin (host) until the counter becomes non-zero, then check the payload.
        uint64_t counter = 0;
        for(int i = 0; i < kPollTimeoutMs; ++i)
        {
            (void)hipMemcpy(&counter, static_cast<uint8_t*>(buf) + counterOff, sizeof(counter),
                            hipMemcpyDeviceToHost);
            (void)hipDeviceSynchronize();
            if(counter != 0)
                break;
            usleep(1000);
        }
        bool dataOk = (counter != 0) && verifyPatternGpu(buf, kMsgSize, seed);
        pass        = dataOk;
        uint8_t res = pass ? 1 : 0;
        MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
    }
    else
    {
        RemoteBuf rb{};
        MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        fillPatternGpu(buf, kMsgSize, seed);

        // 1) RDMA Write the payload.
        bool ok = doRdmaWrite(buf, rr.mr, rb.addr, rb.rkey, kMsgSize, false, 0, 8);

        // 2) Atomic increment of the remote counter.
        void* res_buf = allocBuf(ALLOC_DEVICE, sizeof(uint64_t));
        if(ok && res_buf)
        {
            RegResult resReg = regBuf(res_buf, sizeof(uint64_t), REG_STD, access);
            if(resReg.mr)
            {
                ibv_sge sge{};
                sge.addr   = reinterpret_cast<uint64_t>(res_buf);
                sge.length = sizeof(uint64_t);
                sge.lkey   = resReg.mr->lkey;
                ibv_send_wr wr{};
                wr.wr_id                 = 9;
                wr.opcode                = IBV_WR_ATOMIC_FETCH_AND_ADD;
                wr.send_flags            = IBV_SEND_SIGNALED;
                wr.sg_list               = &sge;
                wr.num_sge               = 1;
                wr.wr.atomic.remote_addr = rb.addr + counterOff;
                wr.wr.atomic.rkey        = rb.rkey;
                wr.wr.atomic.compare_add = 1;
                ibv_send_wr* bad         = nullptr;
                ok                       = (ibv_post_send(qp_, &wr, &bad) == 0);
                if(ok)
                {
                    ibv_wc wc{};
                    ok = pollCq(9, wc, kPollTimeoutMs);
                }
            }
            else
            {
                ok = false;
            }
            deregBuf(resReg);
        }
        else
        {
            ok = false;
        }
        if(res_buf)
            freeBuf(ALLOC_DEVICE, res_buf);

        uint8_t res = 0;
        MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pass = ok && (res == 1);
    }

    deregBuf(rr);
    destroyQp();
    ASSERT_MPI_TRUE(pass);
}

// ===========================================================================
// Test 10: WriteWithImm_Data_Visibility -- on observing the imm completion, the
// payload must already be fully visible in GPU HBM.
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, WriteWithImm_Data_Visibility)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";

    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
    QpInfo remote = exchangeQpInfo();
    ASSERT_MPI_TRUE(rcToRtrRts(remote));

    void* buf = allocBuf(ALLOC_DEVICE, kMsgSize);
    ASSERT_MPI_TRUE(buf != nullptr);
    SCOPE_EXIT(freeBuf(ALLOC_DEVICE, buf));
    RegResult rr = regBuf(buf, kMsgSize, REG_STD, access);
    ASSERT_MPI_TRUE(rr.mr != nullptr);
    SCOPE_EXIT(deregBuf(rr));

    const uint8_t seed = 0xAE;
    bool          pass = false;

    if(isLowerHalf_)
    {
        (void)hipMemset(buf, 0, kMsgSize);
        (void)hipDeviceSynchronize();
        postRecv(rr.mr, buf, kMsgSize, 110);
        RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
        MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
        ibv_wc wc{};
        bool   got = pollCq(110, wc, kPollTimeoutMs);
        // The instant the imm completion is observed, the payload must be visible.
        bool immOk  = got && wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM;
        bool dataOk = immOk && verifyPatternGpu(buf, kMsgSize, seed);
        pass        = dataOk;
        uint8_t res = pass ? 1 : 0;
        MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
    }
    else
    {
        RemoteBuf rb{};
        MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        fillPatternGpu(buf, kMsgSize, seed);
        bool    wrote = doRdmaWrite(buf, rr.mr, rb.addr, rb.rkey, kMsgSize, true, kImmDone, 10);
        uint8_t res   = 0;
        MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pass = wrote && (res == 1);
    }

    deregBuf(rr);
    destroyQp();
    ASSERT_MPI_TRUE(pass);
}

// ===========================================================================
// Test 11: RelaxedOrdering_MR -- register an MR with IBV_ACCESS_RELAXED_ORDERING
// and confirm RDMA completes correctly (no silent corruption).
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, RelaxedOrdering_MR)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";

#ifdef IBV_ACCESS_RELAXED_ORDERING
    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
                       | IBV_ACCESS_RELAXED_ORDERING;

    ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
    QpInfo remote = exchangeQpInfo();
    ASSERT_MPI_TRUE(rcToRtrRts(remote));

    void* buf = allocBuf(ALLOC_DEVICE, kMsgSize);
    ASSERT_MPI_TRUE(buf != nullptr);
    SCOPE_EXIT(freeBuf(ALLOC_DEVICE, buf));

    // Relaxed-ordering registration may be rejected -> coordinated skip.
    RegResult rr = regBuf(buf, kMsgSize, REG_STD, access);
    if(!allRanksAgree(rr.mr != nullptr))
    {
        deregBuf(rr);
        destroyQp();
        GTEST_SKIP() << "Relaxed-ordering MR registration not supported on all ranks";
    }
    SCOPE_EXIT(deregBuf(rr));

    const uint8_t seed = 0xC3;
    bool          pass = false;
    if(isLowerHalf_)
    {
        (void)hipMemset(buf, 0, kMsgSize);
        (void)hipDeviceSynchronize();
        RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
        MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
        uint8_t go = 0;
        MPI_Recv(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pass        = verifyPatternGpu(buf, kMsgSize, seed);
        uint8_t res = pass ? 1 : 0;
        MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
    }
    else
    {
        RemoteBuf rb{};
        MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        fillPatternGpu(buf, kMsgSize, seed);
        bool    wrote = doRdmaWrite(buf, rr.mr, rb.addr, rb.rkey, kMsgSize, false, 0, 11);
        uint8_t go    = 1;
        MPI_Send(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD);
        uint8_t res = 0;
        MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pass = wrote && (res == 1);
    }

    deregBuf(rr);
    destroyQp();
    ASSERT_MPI_TRUE(pass);
#else
    GTEST_SKIP() << "IBV_ACCESS_RELAXED_ORDERING not defined in this rdma-core";
#endif
}

#endif // MPI_TESTS_ENABLED
