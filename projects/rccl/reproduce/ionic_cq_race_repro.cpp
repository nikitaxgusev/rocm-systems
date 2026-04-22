/*
 * ionic_cq_race_repro.cpp
 *
 * Standalone reproducer for:
 *   libionic.so segfault at +0x8ab6 (non-canonical pointer dereference in
 *   ionic ibv_poll_cq CQ entry ring) observed on TW AINIC cluster.
 *
 * ROOT CAUSE:
 *   RCCL's ncclIbCloseSend/Recv calls the sequence (with NO lock held):
 *     1. ibv_destroy_qp(qp)   -- transitions QP→ERROR, deposits FLUSH_ERR
 *                                 completions into the shared CQ ring, but
 *                                 frees the ionic QP object immediately.
 *     2. ibv_destroy_cq(cq)   -- frees the ionic CQ ring buffer.
 *
 *   Concurrently, the progress thread calls ibv_poll_cq(cq, ...) with no
 *   lock held. Inside ionic's poll_cq:
 *     mov 0x158(%rdi,%rax,1),%rax   ; load QP ptr from CQ entry ring[idx]
 *     mov (%rax),%rsi               ; SIGSEGV — freed QP object (0x72f500007942)
 *
 *   The ionic CQ entry ring (at +0x158 in the ionic CQ struct, stride=192 B)
 *   maps completion slots → QP objects. After ibv_destroy_qp, the slot still
 *   holds the freed QP pointer. A concurrent ibv_poll_cq dereferences it.
 *
 * WHAT THIS REPRO DOES:
 *   Thread A (poll):  tight loop calling ibv_poll_cq — stays inside ionic's
 *                     poll_cq implementation as long as possible.
 *   Thread B (main):  posts recv WRs on N QPs sharing one CQ, then calls
 *                     ibv_modify_qp(ERR) to flush them, then ibv_destroy_qp
 *                     (frees QP, leaves FLUSH_ERR completions in CQ ring),
 *                     then ibv_destroy_cq — reproducing the RCCL sequence.
 *
 * EXPECTED OUTPUT (with buggy libionic):
 *   [poll] received SIGBUS/SIGSEGV — crash inside ibv_poll_cq
 *   Failing at address: 0x...  (non-canonical pointer from freed QP slot)
 *
 * BUILD:
 *   g++ -O0 -g -o ionic_cq_race_repro ionic_cq_race_repro.cpp -libverbs -lpthread
 *
 * RUN:
 *   RDMAV_FORK_SAFE=1 ./ionic_cq_race_repro [device]   # default: rdma0
 *   ./ionic_cq_race_repro rdma1   # pick any ionic HCA
 *
 * To stress-test (run until crash):
 *   for i in $(seq 1 200); do ./ionic_cq_race_repro; done
 */

#include <infiniband/verbs.h>
#include <pthread.h>
#include <signal.h>
#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

using std::atomic;
using std::atomic_load;
using std::atomic_store;
using std::atomic_fetch_add;

// ── tunables ────────────────────────────────────────────────────────────────
// Number of QPs sharing the single CQ — more QPs = more CQ ring slots in use
// = wider race window. Matches NCCL_IB_QPS_PER_CONNECTION * nChannels in RCCL.
#define N_QPS        16
#define CQ_DEPTH     1024
#define BUF_SIZE     65536
#define RECV_WRS     32   // per-QP recv WRs posted before destroy
// ────────────────────────────────────────────────────────────────────────────

static struct ibv_cq     *g_cq;
static atomic<int>        g_poll_running{0};  // poll thread has started
static atomic<int>        g_stop_poll{0};     // signal poll thread to exit
static atomic<long>       g_poll_calls{0};    // total ibv_poll_cq calls made
static volatile int       g_crashed = 0;      // set by SIGSEGV handler

// ── SIGSEGV handler — confirm we hit the ionic crash ──────────────────────
static void sigsegv_handler(int sig, siginfo_t *si, void *) {
    g_crashed = 1;
    // Use write() — async-signal-safe
    char msg[256];
    int  len = snprintf(msg, sizeof(msg),
        "\n[CRASH] Signal %d at address %p inside ibv_poll_cq\n"
        "  This is the ionic CQ ring use-after-free (libionic.so +0x8ab6).\n"
        "  Poll calls before crash: %ld\n",
        sig, si->si_addr, atomic_load(&g_poll_calls));
    (void)write(STDERR_FILENO, msg, (size_t)len);
    // Exit with distinctive code so the wrapper script can detect it
    _exit(11);
}

// ── Poll thread — mirrors RCCL's progress thread calling ncclIbTest ────────
static void *poll_thread_fn(void *) {
    struct ibv_wc wcs[32];
    atomic_store(&g_poll_running, 1);

    while (!atomic_load(&g_stop_poll)) {
        // ibv_poll_cq with no lock — exact same pattern as RCCL ncclIbTest
        int n = ibv_poll_cq(g_cq, 32, wcs);
        atomic_fetch_add(&g_poll_calls, 1);

        if (n < 0) {
            fprintf(stderr, "[poll] ibv_poll_cq returned %d (errno=%d) — CQ destroyed\n",
                    n, errno);
            break;
        }
        // Don't check wc->status — we want to stay in the tight loop
    }

    fprintf(stderr, "[poll] thread exiting after %ld poll_cq calls\n",
            atomic_load(&g_poll_calls));
    return nullptr;
}

// ── Move a QP through INIT → RTR → RTS for RC loopback ──────────────────
// (needed to post send WRs; recv WRs only need INIT)
static int qp_to_init(struct ibv_qp *qp, int port) {
    struct ibv_qp_attr attr = {};
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = 0;
    attr.port_num        = (uint8_t)port;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                         | IBV_ACCESS_REMOTE_READ;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

static int qp_to_error(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_ERR;
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE);
}

// ────────────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    const char *dev_name = (argc > 1) ? argv[1] : "rdma0";

    // ── install SIGSEGV/SIGBUS handler ─────────────────────────────────────
    struct sigaction sa = {};
    sa.sa_sigaction = sigsegv_handler;
    sa.sa_flags     = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS,  &sa, nullptr);

    // ── open device ────────────────────────────────────────────────────────
    int num_devices;
    struct ibv_device **devs = ibv_get_device_list(&num_devices);
    if (!devs) { perror("ibv_get_device_list"); return 1; }

    struct ibv_device *dev = nullptr;
    for (int i = 0; i < num_devices; i++) {
        if (strcmp(ibv_get_device_name(devs[i]), dev_name) == 0) {
            dev = devs[i]; break;
        }
    }
    if (!dev) {
        fprintf(stderr, "Device '%s' not found. Available:\n", dev_name);
        for (int i = 0; i < num_devices; i++)
            fprintf(stderr, "  %s\n", ibv_get_device_name(devs[i]));
        ibv_free_device_list(devs);
        return 1;
    }

    struct ibv_context *ctx = ibv_open_device(dev);
    if (!ctx) { perror("ibv_open_device"); return 1; }
    fprintf(stderr, "[main] opened device: %s\n", dev_name);

    // ── PD ─────────────────────────────────────────────────────────────────
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd) { perror("ibv_alloc_pd"); return 1; }

    // ── Single shared CQ — all N_QPS will post completions here ───────────
    // Mirrors RCCL: one ibv_cq per ncclIbNetCommDevBase, shared by all QPs.
    g_cq = ibv_create_cq(ctx, CQ_DEPTH, nullptr, nullptr, 0);
    if (!g_cq) { perror("ibv_create_cq"); return 1; }
    fprintf(stderr, "[main] shared CQ created (depth=%d)\n", CQ_DEPTH);

    // ── Register a memory region for recv WRs ─────────────────────────────
    char *buf = (char *)calloc(1, BUF_SIZE * N_QPS);
    if (!buf) { perror("calloc"); return 1; }
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, (size_t)BUF_SIZE * N_QPS,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) { perror("ibv_reg_mr"); return 1; }

    // ── Create N_QPS RC QPs, all sharing g_cq ─────────────────────────────
    struct ibv_qp *qps[N_QPS];
    for (int i = 0; i < N_QPS; i++) {
        struct ibv_qp_init_attr attr = {};
        attr.send_cq   = g_cq;
        attr.recv_cq   = g_cq;
        attr.qp_type   = IBV_QPT_RC;
        attr.cap.max_send_wr  = RECV_WRS;
        attr.cap.max_recv_wr  = RECV_WRS;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        qps[i] = ibv_create_qp(pd, &attr);
        if (!qps[i]) {
            fprintf(stderr, "ibv_create_qp[%d] failed (errno=%d)\n", i, errno);
            return 1;
        }
    }
    fprintf(stderr, "[main] created %d QPs on shared CQ\n", N_QPS);

    // ── Move all QPs to INIT ───────────────────────────────────────────────
    for (int i = 0; i < N_QPS; i++) {
        if (qp_to_init(qps[i], 1) != 0) {
            fprintf(stderr, "qp_to_init[%d] failed (errno=%d)\n", i, errno);
            return 1;
        }
    }

    // ── Post RECV_WRS recv WRs on each QP ─────────────────────────────────
    // INIT state allows recv posts (verbs spec §11.2.3).
    // When we force QP→ERROR these will all flush as IBV_WC_WR_FLUSH_ERR,
    // depositing completions in the CQ ring that point back to each QP.
    for (int i = 0; i < N_QPS; i++) {
        for (int j = 0; j < RECV_WRS; j++) {
            struct ibv_sge sge = {};
            sge.addr   = (uint64_t)(buf + (size_t)(i * BUF_SIZE + j * 64));
            sge.length = 64;
            sge.lkey   = mr->lkey;

            struct ibv_recv_wr wr = {};
            struct ibv_recv_wr *bad;
            wr.wr_id   = (uint64_t)(i * RECV_WRS + j);
            wr.sg_list = &sge;
            wr.num_sge = 1;

            if (ibv_post_recv(qps[i], &wr, &bad) != 0) {
                fprintf(stderr, "ibv_post_recv[qp=%d wr=%d] failed (errno=%d)\n",
                        i, j, errno);
                return 1;
            }
        }
    }
    fprintf(stderr, "[main] posted %d recv WRs per QP (%d total)\n",
            RECV_WRS, RECV_WRS * N_QPS);

    // ── Start poll thread ──────────────────────────────────────────────────
    pthread_t ptid;
    pthread_create(&ptid, nullptr, poll_thread_fn, nullptr);
    while (!atomic_load(&g_poll_running)) usleep(100);
    fprintf(stderr, "[main] poll thread running\n");

    // Brief spin to let poll thread settle into ibv_poll_cq
    usleep(2000);

    // ──────────────────────────────────────────────────────────────────────
    // THE RACE — mirrors ncclIbCloseSend without a lock:
    //
    // Step 1: ibv_modify_qp → ERROR
    //   Flushes all posted recv WRs as IBV_WC_WR_FLUSH_ERR completions
    //   into the CQ ring. Each CQ ring slot stores a back-pointer to the QP.
    //
    // Step 2: ibv_destroy_qp
    //   Frees the ionic QP object immediately.
    //   The FLUSH_ERR completions are still in the CQ ring, pointing to the
    //   now-freed QP.
    //
    // Step 3: ibv_destroy_cq  (RCCL: ncclIbDestroyBase → wrap_ibv_destroy_cq)
    //   Frees the ionic CQ ring buffer itself.
    //
    // If poll thread is inside ionic's ibv_poll_cq during step 2 or 3:
    //   - During step 2: poll_cq processes a FLUSH_ERR completion, loads the
    //     freed QP pointer from the CQ ring slot (offset +0x158, stride 192 B)
    //     → dereferences it → SIGSEGV at libionic.so+0x8ab6
    //   - During step 3: poll_cq accesses the freed CQ ring buffer → SIGSEGV
    // ──────────────────────────────────────────────────────────────────────

    fprintf(stderr, "[main] flushing QPs to ERROR...\n");
    for (int i = 0; i < N_QPS; i++) {
        // Force all outstanding recv WRs to flush as errors into the CQ ring
        if (qp_to_error(qps[i]) != 0)
            fprintf(stderr, "[main] qp_to_error[%d] failed (errno=%d)\n", i, errno);
    }

    // Tiny delay — let flush completions propagate into the CQ ring
    // but NOT long enough for poll thread to drain them all
    usleep(500);

    fprintf(stderr, "[main] destroying QPs (race window: QP freed, CQ ring still dirty)...\n");
    for (int i = 0; i < N_QPS; i++) {
        // ibv_destroy_qp frees the ionic QP struct.
        // The CQ ring still has FLUSH_ERR completions with a pointer to this QP.
        // Next ibv_poll_cq will dereference the freed pointer → crash.
        if (ibv_destroy_qp(qps[i]) != 0)
            fprintf(stderr, "[main] ibv_destroy_qp[%d] failed (errno=%d)\n", i, errno);
    }

    fprintf(stderr, "[main] destroying CQ (second race: CQ ring freed mid-poll)...\n");
    // If poll thread is inside ionic poll_cq right now → crash
    if (ibv_destroy_cq(g_cq) != 0)
        fprintf(stderr, "[main] ibv_destroy_cq failed (errno=%d)\n", errno);
    g_cq = nullptr;

    // Tell poll thread to exit (if it hasn't crashed yet)
    atomic_store(&g_stop_poll, 1);

    pthread_join(ptid, nullptr);

    // cleanup
    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(devs);
    free(buf);

    if (g_crashed) {
        fprintf(stderr, "[main] REPRODUCED: ionic CQ use-after-free confirmed\n");
        return 11;
    }
    fprintf(stderr, "[main] No crash this iteration (race not hit — retry)\n");
    return 0;
}
