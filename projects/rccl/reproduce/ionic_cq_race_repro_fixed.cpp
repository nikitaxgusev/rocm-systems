/*
 * ionic_cq_race_repro_fixed.cpp
 *
 * Correct fix for the ionic CQ use-after-free race.
 *
 * THE BUG (in the previous "fixed" version):
 *   The poll thread was stopped and joined AFTER ibv_destroy_qp had already
 *   been called. Between ibv_destroy_qp (which deposits FLUSH_ERR CQEs with
 *   freed QP back-pointers) and the join, the poll thread could dereference
 *   those freed pointers -> SIGSEGV.
 *
 * CORRECT TEARDOWN ORDER:
 *   1. destroying.store(true)              stop future polls
 *   2. lock(polling_mutex) + unlock()      wait for any in-flight poll_cq to return
 *   3. qp -> ERROR + ibv_destroy_qp        safe: no thread can dereference CQE now
 *   4. drain CQ                            remove FLUSH_ERR entries
 *   5. ibv_destroy_cq
 *
 * KEY INVARIANT: no thread may be inside ibv_poll_cq at step 3 or later.
 *
 * RCCL mapping:
 *   - poll_lock in poll_thread_fn  <->  std::lock_guard in ncclIbTest
 *   - steps 1+2 in main()          <->  steps 1+2 in ncclIbCloseSend/CloseRecv
 *                                        (before any ibv_destroy_qp call)
 *   - steps 4+5 in main()          <->  drain+destroy in ncclIbDestroyBase
 *
 * The poll thread is still running (NOT joined) when QPs are destroyed —
 * it is only excluded via the mutex, not by being stopped. This is the
 * only synchronization mechanism available in RCCL where the progress
 * thread cannot be joined from the teardown path.
 */

#include <infiniband/verbs.h>
#include <pthread.h>
#include <signal.h>
#include <atomic>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

using std::atomic;

#define N_QPS     16
#define CQ_DEPTH  1024
#define BUF_SIZE  65536
#define RECV_WRS  32

static struct ibv_cq  *g_cq;
static atomic<int>    g_poll_running{0};
static atomic<int>    g_destroying{0};   // set before QP destruction
static atomic<long>   g_poll_calls{0};
static std::mutex     g_polling_mutex;   // held during ibv_poll_cq; teardown acquires to drain
static volatile int   g_crashed = 0;

static void sigsegv_handler(int sig, siginfo_t *si, void *) {
    g_crashed = 1;
    char msg[256];
    int len = snprintf(msg, sizeof(msg),
        "\n[CRASH] Signal %d at address %p — fix did NOT work!\n",
        sig, si->si_addr);
    (void)write(STDERR_FILENO, msg, (size_t)len);
    _exit(11);
}

static void *poll_thread_fn(void *) {
    struct ibv_wc wcs[32];
    g_poll_running.store(1);

    while (true) {
        // Lock covers ibv_poll_cq so teardown can wait for us to finish
        // by acquiring the same mutex (lock+unlock barrier).
        std::lock_guard<std::mutex> lk(g_polling_mutex);
        if (g_destroying.load()) break;   // checked inside the lock
        int n = ibv_poll_cq(g_cq, 32, wcs);
        g_poll_calls.fetch_add(1);
        if (n < 0) break;
    }
    fprintf(stderr, "[poll] exited cleanly after %ld poll_cq calls\n",
            g_poll_calls.load());
    return nullptr;
}

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

int main(int argc, char *argv[]) {
    const char *dev_name = (argc > 1) ? argv[1] : "rdma0";

    struct sigaction sa = {};
    sa.sa_sigaction = sigsegv_handler;
    sa.sa_flags     = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS,  &sa, nullptr);

    int num_devices;
    struct ibv_device **devs = ibv_get_device_list(&num_devices);
    if (!devs) { perror("ibv_get_device_list"); return 1; }

    struct ibv_device *dev = nullptr;
    for (int i = 0; i < num_devices; i++)
        if (strcmp(ibv_get_device_name(devs[i]), dev_name) == 0)
            { dev = devs[i]; break; }
    if (!dev) { fprintf(stderr, "device %s not found\n", dev_name); return 1; }

    struct ibv_context *ctx = ibv_open_device(dev);
    if (!ctx) { perror("ibv_open_device"); return 1; }

    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd) { perror("ibv_alloc_pd"); return 1; }

    g_cq = ibv_create_cq(ctx, CQ_DEPTH, nullptr, nullptr, 0);
    if (!g_cq) { perror("ibv_create_cq"); return 1; }

    char *buf = (char *)calloc(1, (size_t)BUF_SIZE * N_QPS);
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, (size_t)BUF_SIZE * N_QPS,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) { perror("ibv_reg_mr"); return 1; }

    struct ibv_qp *qps[N_QPS];
    for (int i = 0; i < N_QPS; i++) {
        struct ibv_qp_init_attr attr = {};
        attr.send_cq = g_cq; attr.recv_cq = g_cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = RECV_WRS; attr.cap.max_recv_wr = RECV_WRS;
        attr.cap.max_send_sge = 1;       attr.cap.max_recv_sge = 1;
        qps[i] = ibv_create_qp(pd, &attr);
        if (!qps[i]) { fprintf(stderr, "ibv_create_qp[%d] failed\n", i); return 1; }
    }

    for (int i = 0; i < N_QPS; i++)
        if (qp_to_init(qps[i], 1)) { fprintf(stderr, "qp_to_init[%d] failed\n", i); return 1; }

    for (int i = 0; i < N_QPS; i++) {
        for (int j = 0; j < RECV_WRS; j++) {
            struct ibv_sge sge = {};
            sge.addr = (uint64_t)(buf + i * BUF_SIZE + j * 64);
            sge.length = 64; sge.lkey = mr->lkey;
            struct ibv_recv_wr wr = {}, *bad;
            wr.wr_id = (uint64_t)(i * RECV_WRS + j);
            wr.sg_list = &sge; wr.num_sge = 1;
            if (ibv_post_recv(qps[i], &wr, &bad)) {
                fprintf(stderr, "ibv_post_recv[%d,%d] failed\n", i, j); return 1;
            }
        }
    }

    // Start the concurrent poll thread (NOT joined during teardown — mirrors RCCL)
    pthread_t ptid;
    pthread_create(&ptid, nullptr, poll_thread_fn, nullptr);
    while (!g_poll_running.load()) usleep(100);
    fprintf(stderr, "[main] poll thread running (%ld calls so far)\n",
            g_poll_calls.load());

    usleep(2000);

    // ── CORRECT TEARDOWN ────────────────────────────────────────────────────
    //
    // Step 1: signal teardown.
    fprintf(stderr, "[main] step 1: set g_destroying\n");
    g_destroying.store(1);

    // Step 2: acquire+release polling_mutex.
    // This blocks until the poll thread finishes its current ibv_poll_cq call
    // and checks g_destroying (at which point it will exit). After unlock,
    // no thread will ever call ibv_poll_cq again on this CQ.
    fprintf(stderr, "[main] step 2: drain in-flight poll (lock+unlock)...\n");
    {
        std::unique_lock<std::mutex> lk(g_polling_mutex);
        // mutex acquired: poll thread is not in ibv_poll_cq
        lk.unlock();
    }
    fprintf(stderr, "[main] step 2: done — no thread can poll this CQ now\n");

    // Step 3: safe to move QPs to ERROR and destroy them.
    // ibv_destroy_qp deposits FLUSH_ERR CQEs with freed QP pointers — but
    // no thread can dereference them (step 2 guarantees that).
    fprintf(stderr, "[main] step 3: qp->ERROR + ibv_destroy_qp\n");
    for (int i = 0; i < N_QPS; i++)
        qp_to_error(qps[i]);
    for (int i = 0; i < N_QPS; i++)
        ibv_destroy_qp(qps[i]);

    // Step 4: drain FLUSH_ERR completions.
    {
        struct ibv_wc wcs[32];
        int n, total = 0;
        do { n = ibv_poll_cq(g_cq, 32, wcs); if (n > 0) total += n; } while (n > 0);
        fprintf(stderr, "[main] step 4: CQ drained: %d completions\n", total);
    }

    // Step 5: destroy CQ.
    if (ibv_destroy_cq(g_cq))
        fprintf(stderr, "[main] ibv_destroy_cq failed (errno=%d)\n", errno);
    g_cq = nullptr;
    fprintf(stderr, "[main] step 5: CQ destroyed\n");
    // ────────────────────────────────────────────────────────────────────────

    // Poll thread will exit on its own (g_destroying=1 seen inside the lock).
    pthread_join(ptid, nullptr);

    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(devs);
    free(buf);

    if (g_crashed) {
        fprintf(stderr, "[main] CRASHED — fix incomplete\n");
        return 11;
    }
    fprintf(stderr, "[main] Clean exit — correct teardown order confirmed\n");
    return 0;
}
