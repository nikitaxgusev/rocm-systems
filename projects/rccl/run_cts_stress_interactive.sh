#!/bin/bash
# CTS Overflow Stress Test
#
# Reproduces hang/segfault caused by CTS cache overflow on AINIC.
#
# Rank 0 posts NUM_CONNECTIONS PostRecv calls.
# Ranks 1…N connect to rank 0 and fire PostSendWithRetry after a barrier.
# With CTS ON (RCCL_IB_P2P_DISABLE_CTS=0) the test should HANG.
# With CTS OFF (RCCL_IB_P2P_DISABLE_CTS=1) the test should PASS.
#
# Environment variables:
#   NUM_CONNECTIONS  — comma-separated list of connection counts (default: "512,1024,2048")
#   HEAD_NODE        — node to launch mpirun from (default: mia1-p01-g20)
#   TEST_NODES       — comma-separated node list (default: mia1-p01-g20,mia1-p01-g27)
#   NP               — total MPI ranks (default: number of nodes, 1 per node)
#   TIMEOUT_SECS     — per-case timeout in seconds (default: 300)
#   SKIP_BUILD       — set to 1 to skip build phase

set -u

RCCL_DIR="/home/nikita.gusev2@amd.com/v2_clean/rocm-systems/projects/rccl"
MPI_HOME="/home/nikita.gusev2@amd.com/openmpi-5.0.9/install"
INSTALL_PREFIX="${RCCL_DIR}/rccl_installed_cts"
LOG_DIR="${RCCL_DIR}/bench_logs_cts_stress_$(date +%Y%m%d_%H%M%S)"

HEAD_NODE="${HEAD_NODE:-mia1-p01-g20}"
TEST_NODES="${TEST_NODES:-mia1-p01-g20,mia1-p01-g27}"

IFS=',' read -ra NODE_ARRAY <<< "${TEST_NODES}"
NUM_NODES=${#NODE_ARRAY[@]}

HOSTLIST_MPI=$(printf '%s:1,' "${NODE_ARRAY[@]}" | sed 's/,$//')
NP="${NP:-${NUM_NODES}}"

CONN_COUNTS_STR="${NUM_CONNECTIONS:-1024}"
IFS=',' read -ra CONN_COUNTS <<< "${CONN_COUNTS_STR}"

TIMEOUT_SECS="${TIMEOUT_SECS:-120}"
SKIP_BUILD="${SKIP_BUILD:-0}"

export PATH="${MPI_HOME}/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${MPI_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "${LOG_DIR}"

echo "================================================================"
echo "=== CTS Overflow Stress Test — $(date)"
echo "=== Head node:  ${HEAD_NODE}"
echo "=== Test nodes: ${TEST_NODES} (${NUM_NODES} nodes)"
echo "=== MPI ranks:  ${NP}  (hosts: ${HOSTLIST_MPI})"
echo "=== Connection sweep: ${CONN_COUNTS[*]}"
echo "=== Timeout per case: ${TIMEOUT_SECS}s"
echo "=== Logs: ${LOG_DIR}"
echo "================================================================"

# -------------------------------------------------------------------
# Phase 1: Build (optional)
# -------------------------------------------------------------------
if [ "${SKIP_BUILD}" != "1" ]; then
    echo ""
    echo "=== Phase 1: Building RCCL (debug + MPI tests, gfx950) ==="
    ssh "${HEAD_NODE}" "
        cd ${RCCL_DIR}
        export MPI_PATH=${MPI_HOME}
        ./install.sh --debug --no_clean --enable-mpi-tests --amdgpu_targets gfx950 --prefix=${INSTALL_PREFIX} -i 2>&1
    " 2>&1 | tee "${LOG_DIR}/build.log"
    BUILD_RC=${PIPESTATUS[0]}

    if [ ${BUILD_RC} -ne 0 ]; then
        echo "=== BUILD FAILED (rc=${BUILD_RC}) — check ${LOG_DIR}/build.log ==="
        exit 1
    fi
    echo "=== Build succeeded ==="
else
    echo "=== Skipping build (SKIP_BUILD=1) ==="
fi

# Find test binary
TEST_BIN="${RCCL_DIR}/build/debug/test/rccl-UnitTestsMPI"
if [ ! -f "${TEST_BIN}" ]; then
    TEST_BIN=$(find "${RCCL_DIR}/build/debug" -name "rccl-UnitTestsMPI" -type f 2>/dev/null | head -1)
fi

if [ -z "${TEST_BIN:-}" ] || [ ! -f "${TEST_BIN}" ]; then
    echo "=== ERROR: rccl-UnitTestsMPI binary not found ==="
    find "${RCCL_DIR}/build/debug" -name "*MPI*" -type f 2>/dev/null
    exit 1
fi

echo "=== Test binary: ${TEST_BIN} ==="

# -------------------------------------------------------------------
# Phase 2: Run stress tests
# -------------------------------------------------------------------
GTEST_FILTER="--gtest_filter=NetIbMPITest.CtsDepthStress"

set +e
declare -A RESULTS

run_stress_case() {
    local CASE_NAME="$1"
    local NUM_CONNS="$2"
    shift 2
    local EXTRA_ENV=("$@")
    local LOG_FILE="${LOG_DIR}/${CASE_NAME}_${NUM_CONNS}.log"

    local MPI_FLAGS=(
        -np "${NP}"
        -H "${HOSTLIST_MPI}"
        --map-by ppr:1:node
        --bind-to none
        --allow-run-as-root
        -mca btl_tcp_if_include tw-eth5
        -x PATH
        -x LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${MPI_HOME}/lib"
        -x RCCL_AINIC_ROCE=1
        -x IONIC_LOCKFREE=all
        -x NCCL_SOCKET_IFNAME=tw-eth5
        -x NCCL_IB_GID_INDEX=1
        -x HSA_NO_SCRATCH_RECLAIM=1
        -x NCCL_IB_MERGE_NICS=0
        -x RCCL_CTS_MAX_OUTSTANDING=1
        -x NCCL_DEBUG=info
        -x NCCL_DEBUG_SUBSYS=INIT,NET
        -x NUM_CONNECTIONS=${NUM_CONNS}
        -x CTS_QP_DEPTH=256
        -x CTS_NUM_CONNS=1
        -x CTS_SNAP_EVERY=1
    )

    for env_var in "${EXTRA_ENV[@]}"; do
        MPI_FLAGS+=(-x "${env_var}")
    done

    echo ""
    echo "================================================================"
    echo "=== ${CASE_NAME} / ${NUM_CONNS} conns / ${NP} ranks  at $(date) ==="
    echo "================================================================"

    ssh "${HEAD_NODE}" "
        export PATH=${MPI_HOME}/bin:\${PATH}
        export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${MPI_HOME}/lib:\${LD_LIBRARY_PATH}
        timeout ${TIMEOUT_SECS} mpirun ${MPI_FLAGS[*]} ${TEST_BIN} ${GTEST_FILTER} 2>&1
    " 2>&1 | tee "${LOG_FILE}"
    rc=${PIPESTATUS[0]}

    local KEY="${CASE_NAME}_${NUM_CONNS}"
    if [ $rc -eq 124 ]; then
        RESULTS[${KEY}]="TIMEOUT (CTS overflow)"
        echo "=== ${KEY}: TIMEOUT ==="
    elif [ $rc -eq 139 ] || [ $rc -eq 134 ]; then
        RESULTS[${KEY}]="CRASH (signal $(( rc - 128 )))"
        echo "=== ${KEY}: CRASH (signal $(( rc - 128 ))) ==="
    elif [ $rc -ne 0 ]; then
        RESULTS[${KEY}]="FAILED (rc=$rc)"
        echo "=== ${KEY}: FAILED (exit code $rc) ==="
    else
        RESULTS[${KEY}]="PASS"
        echo "=== ${KEY}: PASS ==="
    fi

    sleep 5
}

for NC in "${CONN_COUNTS[@]}"; do
    echo ""
    echo "################################################################"
    echo "### NUM_CONNECTIONS=${NC}"
    echo "################################################################"

    # CTS ON (DISABLE_CTS=0) → expect overflow / hang
    # run_stress_case "cts_ON" "${NC}" "RCCL_IB_P2P_DISABLE_CTS=1"

    # CTS OFF (DISABLE_CTS=1) → should always pass
    run_stress_case "cts_OFF" "${NC}" "RCCL_IB_P2P_DISABLE_CTS=0"
done

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo ""
echo "================================================================"
echo "=== SUMMARY  (${NP} ranks, ${NUM_NODES} nodes) ==="
echo "================================================================"
printf "%-45s : %s\n" "SCENARIO" "RESULT"
echo "----------------------------------------------------------------"
for NC in "${CONN_COUNTS[@]}"; do
    printf "%-45s : %s\n" "CTS ON  (DISABLE_CTS=0) / ${NC} conns"  "${RESULTS[cts_ON_${NC}]:-N/A}"
    printf "%-45s : %s\n" "CTS OFF (DISABLE_CTS=1) / ${NC} conns" "${RESULTS[cts_OFF_${NC}]:-N/A}"
    echo "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
done
echo "================================================================"
echo "=== Logs: ${LOG_DIR} ==="
echo "=== Completed at $(date) ==="
