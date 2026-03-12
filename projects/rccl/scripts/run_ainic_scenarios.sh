#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $0 [--dry-run] [--np N] [--ppn N] [--node1 HOST] [--node2 HOST] [--cycle N]

Options:
  --np N        Total number of ranks
  --ppn N       Ranks per node
  --node1 HOST  First node hostname/IP
  --node2 HOST  Second node hostname/IP
  --cycle N     Number of iterations passed to benchmark via -N (default: 10)
  --dry-run     Print commands without executing
  -h, --help    Show this help

Examples:
  $0 --np 4 --ppn 2
  $0 --np 16 --ppn 8 --cycle 20
  $0 --np 4 --ppn 2 --dry-run

Notes:
  - This script assumes exactly 2 nodes.
  - For 2 nodes, --np must equal --ppn * 2.
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && (( "$1" > 0 ))
}

NODE1="${NODE1:-10.235.192.31}"
NODE2="${NODE2:-10.235.192.32}"

RANKS_PER_NODE="${RANKS_PER_NODE:-2}"

PERF_BIN_DIR="${PERF_BIN_DIR:-/opt/rccl/bin}"
RESULTS_DIR="${RESULTS_DIR:-./ainic_results_allreduce_alltoall_nics_merge_2_new_env_vars_11march}"
NIC_IF="${NIC_IF:-ai0}"
SIZE_BEGIN="${SIZE_BEGIN:-8}"
SIZE_END="${SIZE_END:-8G}"
SIZE_FACTOR="${SIZE_FACTOR:-2}"
GID_INDEX="${GID_INDEX:-1}"

CYCLE="${CYCLE:-10}"

DRY_RUN=0
CLI_NP=""
CLI_PPN=""
CLI_CYCLE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --np)
      [[ $# -ge 2 ]] || { echo "ERROR: --np requires a value"; usage; exit 1; }
      CLI_NP="$2"
      shift 2
      ;;
    --ppn)
      [[ $# -ge 2 ]] || { echo "ERROR: --ppn requires a value"; usage; exit 1; }
      CLI_PPN="$2"
      shift 2
      ;;
    --node1)
      [[ $# -ge 2 ]] || { echo "ERROR: --node1 requires a value"; usage; exit 1; }
      NODE1="$2"
      shift 2
      ;;
    --node2)
      [[ $# -ge 2 ]] || { echo "ERROR: --node2 requires a value"; usage; exit 1; }
      NODE2="$2"
      shift 2
      ;;
    --cycle)
      [[ $# -ge 2 ]] || { echo "ERROR: --cycle requires a value"; usage; exit 1; }
      CLI_CYCLE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$CLI_PPN" ]]; then
  RANKS_PER_NODE="$CLI_PPN"
fi

if [[ -n "$CLI_NP" ]]; then
  NP="$CLI_NP"
else
  NP=$(( RANKS_PER_NODE * 2 ))
fi

if [[ -n "$CLI_CYCLE" ]]; then
  CYCLE="$CLI_CYCLE"
fi

is_positive_int "$RANKS_PER_NODE" || {
  echo "ERROR: --ppn / RANKS_PER_NODE must be a positive integer, got: $RANKS_PER_NODE"
  usage
  exit 1
}

is_positive_int "$NP" || {
  echo "ERROR: --np must be a positive integer, got: $NP"
  usage
  exit 1
}

is_positive_int "$CYCLE" || {
  echo "ERROR: --cycle / CYCLE must be a positive integer, got: $CYCLE"
  usage
  exit 1
}

EXPECTED_NP=$(( RANKS_PER_NODE * 2 ))
if (( NP != EXPECTED_NP )); then
  echo "ERROR: invalid rank setup for 2 nodes: --np=$NP but --ppn=$RANKS_PER_NODE, expected np=$EXPECTED_NP"
  usage
  exit 1
fi

COLLECTIVES=(
  all_reduce
  #all_gather
  #reduce_scatter
  #gather
  #scatter
  #broadcast
  #reduce
  alltoall
)

declare -a SCENARIO_NAMES=(
  "ainic_off"
  "ainic_on__cts_off__udma_off"
  "ainic_on__cts_off__udma_roundrobin"
  "ainic_on__cts_off__udma_high"
  "ainic_on__cts_on__udma_off"
  "ainic_on__cts_on__udma_roundrobin"
  "ainic_on__cts_on__udma_high"
)

declare -a SCENARIO_AINIC=(   0  1  1  1  1  1  1 )
declare -a SCENARIO_CTS_INL=( "" 0  0  0  1  1  1 )
declare -a SCENARIO_CTS_OFF=( "" 0  0  0  1  1  1 )
declare -a SCENARIO_UDMA=(    "" 4  1  2  4  1  2 )

log_header() {
  local sname="$1" coll="$2" perf_bin="$3" ainic="$4" cts_inl="$5" cts_off="$6" udma="$7" cmd_str="$8"
  local log_file="$9"

  local header
  header=$(
    cat <<EOF
======================================================================
LAUNCH  : $(timestamp)
SCENARIO: ${sname}
COLL    : ${coll}
BIN     : ${perf_bin}
NODES   : ${NODE1}, ${NODE2}
NP      : ${NP} (${RANKS_PER_NODE} ranks per node)
SIZES   : ${SIZE_BEGIN} .. ${SIZE_END} (x${SIZE_FACTOR})
CYCLE   : ${CYCLE}
NET     : NIC_IF=${NIC_IF}  GID_INDEX=${GID_INDEX}
FLAGS   : RCCL_AINIC_ROCE=${ainic}  RCCL_CTS_INLINE_DATA=${cts_inl:-default}  RCCL_CTS_OFFLOAD_ENABLED=${cts_off:-default}  RCCL_AINIC_UDMA_BALANCE=${udma:-default}
CMD     : ${cmd_str}
======================================================================
EOF
  )

  mkdir -p "$(dirname "$log_file")"
  printf "%s\n" "$header" >> "$log_file"
}

run_one() {
  local scenario_idx="$1"
  local coll="$2"

  local sname="${SCENARIO_NAMES[$scenario_idx]}"
  local ainic="${SCENARIO_AINIC[$scenario_idx]}"
  local cts_inl="${SCENARIO_CTS_INL[$scenario_idx]}"
  local cts_off="${SCENARIO_CTS_OFF[$scenario_idx]}"
  local udma="${SCENARIO_UDMA[$scenario_idx]}"

  local perf_bin="${PERF_BIN_DIR}/${coll}_perf"
  local log_file="${RESULTS_DIR}/${coll}.log"

  local -a xflags=(
    -x LD_LIBRARY_PATH
    -x PATH
    -x "RCCL_AINIC_ROCE=${ainic}"
    -x IONIC_LOCKFREE=all
    -x "NCCL_SOCKET_IFNAME=${NIC_IF}"
    -x "NCCL_IB_GID_INDEX=${GID_INDEX}"
    -x HSA_NO_SCRATCH_RECLAIM=1
    -x NCCL_DEBUG=VERSION
    -x NCCL_IB_MERGE_NICS=0
  )

  if [[ -n "$cts_inl" ]]; then xflags+=( -x "RCCL_CTS_INLINE_DATA=${cts_inl}" ); fi
  if [[ -n "$cts_off" ]]; then xflags+=( -x "RCCL_CTS_OFFLOAD_ENABLED=${cts_off}" ); fi
  if [[ -n "$udma" ]]; then xflags+=( -x "RCCL_AINIC_UDMA_BALANCE=${udma}" ); fi

  local cmd=(
    mpirun -np "$NP"
      -H "${NODE1}:${RANKS_PER_NODE},${NODE2}:${RANKS_PER_NODE}"
      --map-by "ppr:${RANKS_PER_NODE}:node"
      --bind-to numa
      --allow-run-as-root
      -mca "btl_tcp_if_include" "${NIC_IF}"
      "${xflags[@]}"
      "$perf_bin"
        -b "$SIZE_BEGIN" -e "$SIZE_END" -f "$SIZE_FACTOR" -g 1 -N "$CYCLE"
  )

  local cmd_str
  printf -v cmd_str '%q ' "${cmd[@]}"

  echo "$(timestamp)  [${sname}]  ${coll}  ->  ${log_file}"

  if (( DRY_RUN )); then
    echo "CMD: ${cmd_str}"
    return 0
  fi

  log_header "$sname" "$coll" "$perf_bin" "$ainic" "$cts_inl" "$cts_off" "$udma" "$cmd_str" "$log_file"

  "${cmd[@]}" 2>&1 | tee -a "$log_file"
}

mkdir -p "$RESULTS_DIR"

total=$(( ${#SCENARIO_NAMES[@]} * ${#COLLECTIVES[@]} ))
current=0

echo "======================================================================"
echo "AINIC scenario sweep – $(timestamp)"
echo "Nodes      : ${NODE1}, ${NODE2}"
echo "Ranks/node : ${RANKS_PER_NODE}"
echo "Total ranks: ${NP}"
echo "Sizes      : ${SIZE_BEGIN} .. ${SIZE_END} (x${SIZE_FACTOR})"
echo "Cycle (-N) : ${CYCLE}"
echo "Collectives: ${COLLECTIVES[*]}"
echo "Scenarios  : ${#SCENARIO_NAMES[@]}"
echo "Total runs : ${total}"
echo "Results    : ${RESULTS_DIR}"
if (( DRY_RUN )); then
  echo "*** DRY RUN – commands printed, not executed ***"
fi
echo "======================================================================"

for sidx in "${!SCENARIO_NAMES[@]}"; do
  sname="${SCENARIO_NAMES[$sidx]}"

  echo ""
  echo "--- Scenario: ${sname} ---"

  for coll in "${COLLECTIVES[@]}"; do
    (( current++ )) || true
    echo "[${current}/${total}] ${sname} / ${coll}"
    run_one "$sidx" "$coll"
  done
done

echo ""
echo "$(timestamp) All ${total} runs completed. Results in ${RESULTS_DIR}"
