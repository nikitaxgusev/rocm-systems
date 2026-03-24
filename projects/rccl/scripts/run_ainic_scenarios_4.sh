#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# AINIC Scenario Benchmark Runner (v3)
#
# Loop order:  iterations (outer) → configs → collectives (inner)
# Each benchmark invocation uses -N 1; --iters controls repetitions.
# Supports mpirun (direct) and srun (SLURM).
# ──────────────────────────────────────────────────────────────────────

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && (( "$1" > 0 ))
}

# ── Configuration Table ──────────────────────────────────────────────
# "name | AINIC_ROCE | CTS_INLINE_DATA | CTS_OFFLOAD_ENABLED | UDMA_BALANCE | FORCE_INLINE | GDR_FLUSH_DISABLE"
# Empty field = don't set (library default).
declare -a CFGS=(
  "ainic_off|0|||||"                                              # 0
  "ainic_on__cts_off__udma_off|1|0|0|4||"                         # 1
  "ainic_on__cts_off__udma_roundrobin|1|0|0|1||"                  # 2
  "ainic_on__cts_off__udma_high|1|0|0|2||"                        # 3
  "ainic_on__cts_off__udma_off__inl0_gdr1|1|0|0|4|0|1"           # 4
  "ainic_on__cts_off__udma_roundrobin__inl0_gdr1|1|0|0|1|0|1"    # 5
  "ainic_on__cts_off__udma_high__inl0_gdr1|1|0|0|2|0|1"          # 6
  "ainic_on__cts_on__udma_off|1|1|1|4||"                          # 7
  "ainic_on__cts_on__udma_roundrobin|1|1|1|1||"                   # 8
  "ainic_on__cts_on__udma_high|1|1|1|2||"                         # 9
  "ainic_on__cts_on__udma_off__inl0_gdr1|1|1|1|4|0|1"            # 10
  "ainic_on__cts_on__udma_roundrobin__inl0_gdr1|1|1|1|1|0|1"     # 11
  "ainic_on__cts_on__udma_high__inl0_gdr1|1|1|1|2|0|1"           # 12
)

DEFAULT_CONFIGS="all"

list_configs() {
  echo "Available configurations:"
  echo "────────────────────────────────────────────────────────────────────"
  for i in "${!CFGS[@]}"; do
    local name ainic cts_inl cts_off udma finl gdr
    IFS='|' read -r name ainic cts_inl cts_off udma finl gdr <<< "${CFGS[$i]}"
    local flags="ROCE=${ainic}"
    [[ -n "$cts_inl" ]] && flags+="  CTS_INL=${cts_inl}" || flags+="  CTS_INL=def"
    [[ -n "$cts_off" ]] && flags+="  CTS_OFF=${cts_off}" || flags+="  CTS_OFF=def"
    [[ -n "$udma" ]]    && flags+="  UDMA=${udma}"        || flags+="  UDMA=def"
    [[ -n "$finl" ]]    && flags+="  FINL=${finl}"         || flags+="  FINL=def"
    [[ -n "$gdr" ]]     && flags+="  GDR=${gdr}"           || flags+="  GDR=def"
    printf "  [%2d]  %-55s %s\n" "$i" "$name" "$flags"
  done
  echo ""
  echo "Default: ${DEFAULT_CONFIGS}"
}

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --nodes HOST1,HOST2[,...]  Comma-separated node list (default: 2 known nodes)
  --ppn N                    Ranks per node (default: 2)
  --np N                     Total ranks (default: ppn × num_nodes)
  --iters N | --cycle N      Outer iteration count (default: 5)
  --configs IDX,IDX,...|all  Config indices to run (default: ${DEFAULT_CONFIGS})
  --timeout SEC              Per-run timeout in seconds (default: 600, 0 = none)
  --results-dir DIR          Output directory (default: ./ainic_results)
  --launcher mpirun|srun     Launch method (default: auto-detect SLURM)
  --gen-sbatch FILE          Generate sbatch wrapper script and exit
  --partition NAME           SLURM partition name (used by --gen-sbatch)
  --dry-run                  Print commands without executing
  --list-configs             List all available configs and exit
  -h, --help                 Show this help

Loop order:  iterations (outer) → configs → collectives (inner).
Each benchmark invocation uses -N 1.

Examples:
  $0 --ppn 8 --iters 5
  $0 --ppn 8 --nodes h1,h2,h3,h4 --configs 0,4,5,11
  $0 --ppn 8 --iters 10 --configs all
  $0 --ppn 8 --gen-sbatch bench.sbatch
  $0 --ppn 8 --dry-run --list-configs
EOF
}

# ── Defaults ─────────────────────────────────────────────────────────
NODES_STR="${NODES:-10.235.192.31,10.235.192.32}"
PPN="${RANKS_PER_NODE:-2}"
CLI_NP=""
ITERS=5
CONFIGS_STR="${DEFAULT_CONFIGS}"
RUN_TIMEOUT="${RUN_TIMEOUT:-600}"
RESUME_ITER=1
RESUME_CFG=0
LAUNCHER="auto"
GEN_SBATCH=""
PARTITION=""
DRY_RUN=0

PERF_BIN_DIR="${PERF_BIN_DIR:-/home/nikita.gusev2@amd.com/rocm-systems/projects/rccl/rccl_installed/bin}"
RESULTS_DIR="${RESULTS_DIR:-./ainic_results}"
MPI_HOME="${MPI_HOME:-/home/nikita.gusev2@amd.com/openmpi-5.0.9/install}"
RCCL_HOME="${RCCL_HOME:-/home/nikita.gusev2@amd.com/rocm-systems/projects/rccl/rccl_installed}"
NIC_IF="${NIC_IF:-ai0}"
SIZE_BEGIN="${SIZE_BEGIN:-8}"
SIZE_END="${SIZE_END:-8G}"
SIZE_FACTOR="${SIZE_FACTOR:-2}"
GID_INDEX="${GID_INDEX:-1}"

COLLECTIVES=(
  #all_reduce
  alltoall
)

# ── Parse CLI ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes)
      [[ $# -ge 2 ]] || { echo "ERROR: --nodes requires value"; exit 1; }
      NODES_STR="$2"; shift 2 ;;
    --ppn)
      [[ $# -ge 2 ]] || { echo "ERROR: --ppn requires value"; exit 1; }
      PPN="$2"; shift 2 ;;
    --np)
      [[ $# -ge 2 ]] || { echo "ERROR: --np requires value"; exit 1; }
      CLI_NP="$2"; shift 2 ;;
    --iters|--cycle)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires value"; exit 1; }
      ITERS="$2"; shift 2 ;;
    --configs)
      [[ $# -ge 2 ]] || { echo "ERROR: --configs requires value"; exit 1; }
      CONFIGS_STR="$2"; shift 2 ;;
    --timeout)
      [[ $# -ge 2 ]] || { echo "ERROR: --timeout requires value"; exit 1; }
      RUN_TIMEOUT="$2"; shift 2 ;;
    --resume-iter)
      [[ $# -ge 2 ]] || { echo "ERROR: --resume-iter requires value"; exit 1; }
      RESUME_ITER="$2"; shift 2 ;;
    --resume-cfg)
      [[ $# -ge 2 ]] || { echo "ERROR: --resume-cfg requires value"; exit 1; }
      RESUME_CFG="$2"; shift 2 ;;
    --results-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --results-dir requires value"; exit 1; }
      RESULTS_DIR="$2"; shift 2 ;;
    --launcher)
      [[ $# -ge 2 ]] || { echo "ERROR: --launcher requires value"; exit 1; }
      LAUNCHER="$2"; shift 2 ;;
    --partition)
      [[ $# -ge 2 ]] || { echo "ERROR: --partition requires value"; exit 1; }
      PARTITION="$2"; shift 2 ;;
    --gen-sbatch)
      [[ $# -ge 2 ]] || { echo "ERROR: --gen-sbatch requires filename"; exit 1; }
      GEN_SBATCH="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --list-configs)
      list_configs; exit 0 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1"; usage; exit 1 ;;
  esac
done

# ── Resolve Nodes ────────────────────────────────────────────────────
# Under SLURM with default nodes, prefer SLURM_JOB_NODELIST.
if [[ -n "${SLURM_JOB_NODELIST:-}" && "$NODES_STR" == "10.235.192.31,10.235.192.32" ]]; then
  NODES_STR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)
fi

IFS=',' read -ra NODES <<< "$NODES_STR"
NUM_NODES=${#NODES[@]}

if [[ -n "$CLI_NP" ]]; then
  NP="$CLI_NP"
else
  NP=$(( PPN * NUM_NODES ))
fi

# ── Validate ─────────────────────────────────────────────────────────
is_positive_int "$PPN"   || { echo "ERROR: --ppn must be a positive integer, got: $PPN"; exit 1; }
is_positive_int "$NP"    || { echo "ERROR: --np must be a positive integer, got: $NP"; exit 1; }
is_positive_int "$ITERS" || { echo "ERROR: --iters must be a positive integer, got: $ITERS"; exit 1; }

EXPECTED_NP=$(( PPN * NUM_NODES ))
(( NP == EXPECTED_NP )) || {
  echo "ERROR: --np=$NP != --ppn=$PPN × ${NUM_NODES} nodes (expected $EXPECTED_NP)"
  exit 1
}

# ── Resolve Configs ──────────────────────────────────────────────────
if [[ "$CONFIGS_STR" == "all" ]]; then
  SELECTED=()
  for i in "${!CFGS[@]}"; do SELECTED+=("$i"); done
else
  IFS=',' read -ra SELECTED <<< "$CONFIGS_STR"
fi

for idx in "${SELECTED[@]}"; do
  [[ "$idx" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid config index: $idx"; exit 1; }
  (( idx < ${#CFGS[@]} )) || { echo "ERROR: config index $idx out of range (0..$(( ${#CFGS[@]} - 1 )))"; exit 1; }
done

# ── Resolve Launcher ─────────────────────────────────────────────────
if [[ "$LAUNCHER" == "auto" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LAUNCHER="srun"
  else
    LAUNCHER="mpirun"
  fi
fi

[[ "$LAUNCHER" == "mpirun" || "$LAUNCHER" == "srun" ]] || {
  echo "ERROR: --launcher must be 'mpirun' or 'srun', got: $LAUNCHER"
  exit 1
}

# ── Generate sbatch (if requested) ───────────────────────────────────
if [[ -n "$GEN_SBATCH" ]]; then
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

  # Build partition line
  if [[ -n "$PARTITION" ]]; then
    PART_LINE="#SBATCH --partition=${PARTITION}"
  else
    PART_LINE="##SBATCH --partition=YOUR_PARTITION  # ← uncomment and set"
  fi

  cat > "$GEN_SBATCH" <<SBEOF
#!/bin/bash
#SBATCH --job-name=ainic-bench
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks=${NP}
#SBATCH --ntasks-per-node=${PPN}
#SBATCH --gpus-per-node=${PPN}
#SBATCH --time=08:00:00
#SBATCH --output=ainic-bench-%j.log
#SBATCH --error=ainic-bench-%j.err
${PART_LINE}
# ── Edit above as needed (QOS, account, time, etc.) ──
# If you change --nodes, also update --ntasks = nodes × ntasks-per-node.

export PERF_BIN_DIR="${PERF_BIN_DIR}"
export RESULTS_DIR="${RESULTS_DIR}"
export NIC_IF="${NIC_IF}"
export SIZE_BEGIN="${SIZE_BEGIN}"
export SIZE_END="${SIZE_END}"
export SIZE_FACTOR="${SIZE_FACTOR}"
export GID_INDEX="${GID_INDEX}"

${SCRIPT_PATH} \\
  --launcher srun \\
  --ppn ${PPN} \\
  --iters ${ITERS} \\
  --configs "${CONFIGS_STR}" \\
  --timeout ${RUN_TIMEOUT}
SBEOF
  chmod +x "$GEN_SBATCH"
  echo "Generated: ${GEN_SBATCH}"
  echo "Submit with: sbatch ${GEN_SBATCH}"
  exit 0
fi

# ── Build mpirun host string ─────────────────────────────────────────
MPIRUN_HOSTS=""
for node in "${NODES[@]}"; do
  [[ -n "$MPIRUN_HOSTS" ]] && MPIRUN_HOSTS+=","
  MPIRUN_HOSTS+="${node}:${PPN}"
done

# ── Logging ──────────────────────────────────────────────────────────
log_header() {
  local sname="$1" coll="$2" iter="$3" cmd_str="$4" log_file="$5"
  local ainic="$6" cts_inl="$7" cts_off="$8" udma="$9" finl="${10}" gdr="${11}"

  mkdir -p "$(dirname "$log_file")"
  cat >> "$log_file" <<EOF
======================================================================
LAUNCH  : $(timestamp)
SCENARIO: ${sname}
COLL    : ${coll}
ITER    : ${iter}/${ITERS}
NODES   : ${NODES_STR} (${NUM_NODES} nodes)
NP      : ${NP} (${PPN} per node)
LAUNCHER: ${LAUNCHER}
SIZES   : ${SIZE_BEGIN} .. ${SIZE_END} (x${SIZE_FACTOR})
TIMEOUT : ${RUN_TIMEOUT}s
FLAGS   : RCCL_AINIC_ROCE=${ainic}  CTS_INLINE_DATA=${cts_inl:-def}  CTS_OFFLOAD=${cts_off:-def}  UDMA_BALANCE=${udma:-def}  FORCE_INLINE=${finl:-def}  GDR_FLUSH_DISABLE=${gdr:-def}
CMD     : ${cmd_str}
======================================================================
EOF
}

# ── Run One ──────────────────────────────────────────────────────────
run_one() {
  local cfg_idx="$1" coll="$2" iter="$3"

  local name ainic cts_inl cts_off udma finl gdr
  IFS='|' read -r name ainic cts_inl cts_off udma finl gdr <<< "${CFGS[$cfg_idx]}"

  local perf_bin="${PERF_BIN_DIR}/${coll}_perf"
  local log_file="${RESULTS_DIR}/${coll}.log"

  local -a cmd=()

  if [[ "$LAUNCHER" == "mpirun" ]]; then
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
    [[ -n "$cts_inl" ]] && xflags+=( -x "RCCL_CTS_INLINE_DATA=${cts_inl}" )
    [[ -n "$cts_off" ]] && xflags+=( -x "RCCL_CTS_OFFLOAD_ENABLED=${cts_off}" )
    [[ -n "$udma" ]]    && xflags+=( -x "RCCL_AINIC_UDMA_BALANCE=${udma}" )
    [[ -n "$finl" ]]    && xflags+=( -x "RCCL_AINIC_FORCE_INLINE=${finl}" )
    [[ -n "$gdr" ]]     && xflags+=( -x "RCCL_AINIC_GDR_FLUSH_DISABLE=${gdr}" )

    cmd=(
      mpirun -np "$NP"
        -H "${MPIRUN_HOSTS}"
        --map-by "ppr:${PPN}:node"
        --bind-to numa
        --allow-run-as-root
        -mca "btl_tcp_if_include" "${NIC_IF}"
        "${xflags[@]}"
        "$perf_bin"
          -b "$SIZE_BEGIN" -e "$SIZE_END" -f "$SIZE_FACTOR" -g 1 -n 100 -w 10 -N 1
    )

  else  # srun
    local -a env_vars=(
      "RCCL_AINIC_ROCE=${ainic}"
      "IONIC_LOCKFREE=all"
      "NCCL_SOCKET_IFNAME=${NIC_IF}"
      "NCCL_IB_GID_INDEX=${GID_INDEX}"
      "HSA_NO_SCRATCH_RECLAIM=1"
      "NCCL_DEBUG=VERSION"
      "NCCL_IB_MERGE_NICS=0"
      "OMPI_MCA_btl_tcp_if_include=${NIC_IF}"
    )
    [[ -n "$cts_inl" ]] && env_vars+=( "RCCL_CTS_INLINE_DATA=${cts_inl}" )
    [[ -n "$cts_off" ]] && env_vars+=( "RCCL_CTS_OFFLOAD_ENABLED=${cts_off}" )
    [[ -n "$udma" ]]    && env_vars+=( "RCCL_AINIC_UDMA_BALANCE=${udma}" )
    [[ -n "$finl" ]]    && env_vars+=( "RCCL_AINIC_FORCE_INLINE=${finl}" )
    [[ -n "$gdr" ]]     && env_vars+=( "RCCL_AINIC_GDR_FLUSH_DISABLE=${gdr}" )

    cmd=(
      env "${env_vars[@]}"
      srun --ntasks="$NP" --ntasks-per-node="$PPN"
        "$perf_bin"
          -b "$SIZE_BEGIN" -e "$SIZE_END" -f "$SIZE_FACTOR" -g 1 -n 100 -w 10 -N 1
    )
  fi

  local cmd_str
  printf -v cmd_str '%q ' "${cmd[@]}"

  echo "$(timestamp)  [iter ${iter}/${ITERS}]  ${name}  ${coll}  ->  ${log_file}"

  if (( DRY_RUN )); then
    echo "  CMD: ${cmd_str}"
    return 0
  fi

  log_header "$name" "$coll" "$iter" "$cmd_str" "$log_file" \
             "$ainic" "$cts_inl" "$cts_off" "$udma" "$finl" "$gdr"

  local rc=0
  if (( RUN_TIMEOUT > 0 )); then
    timeout --signal=SIGTERM --kill-after=30 "${RUN_TIMEOUT}" \
      bash -c '"$@" 2>&1 | tee -a "$0"' "$log_file" "${cmd[@]}" || rc=$?
  else
    "${cmd[@]}" 2>&1 | tee -a "$log_file" || rc=$?
  fi

  if (( rc == 124 || rc == 137 )); then
    echo "*** TIMEOUT (${RUN_TIMEOUT}s) — killed: ${name} / ${coll} ***" | tee -a "$log_file"
  elif (( rc != 0 )); then
    echo "*** FAILED (exit $rc): ${name} / ${coll} ***" | tee -a "$log_file"
  fi

  return 0
}

# ── Main ─────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="${RCCL_HOME}/lib:${MPI_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="${MPI_HOME}/bin${PATH:+:$PATH}"

mkdir -p "$RESULTS_DIR"

total=$(( ITERS * ${#SELECTED[@]} * ${#COLLECTIVES[@]} ))

# Calculate how many tasks to skip based on resume point
skipped_iters=$(( RESUME_ITER - 1 ))
skipped_cfgs=0
for cfg_idx in "${SELECTED[@]}"; do
  (( cfg_idx < RESUME_CFG )) && (( skipped_cfgs++ )) || true
done
current=$(( skipped_iters * ${#SELECTED[@]} * ${#COLLECTIVES[@]} + skipped_cfgs * ${#COLLECTIVES[@]} ))

echo "======================================================================"
echo "AINIC scenario sweep — $(timestamp)"
echo "Nodes      : ${NODES_STR} (${NUM_NODES} nodes)"
echo "Launcher   : ${LAUNCHER}"
echo "Ranks/node : ${PPN}"
echo "Total ranks: ${NP}"
echo "Sizes      : ${SIZE_BEGIN} .. ${SIZE_END} (x${SIZE_FACTOR})"
echo "Iterations : ${ITERS}"
echo "Timeout    : ${RUN_TIMEOUT}s per run (0 = none)"
echo "Collectives: ${COLLECTIVES[*]}"
echo "Configs (${#SELECTED[@]}):"
for idx in "${SELECTED[@]}"; do
  IFS='|' read -r _n _ <<< "${CFGS[$idx]}"
  echo "  [${idx}] ${_n}"
done
echo "Total runs : ${total}"
echo "Results    : ${RESULTS_DIR}"
if (( DRY_RUN )); then
  echo "*** DRY RUN – commands printed, not executed ***"
fi
echo "======================================================================"

for iter in $(seq 1 "$ITERS"); do
  echo ""
  echo "══════ Iteration ${iter}/${ITERS} — $(timestamp) ══════"

  for cfg_idx in "${SELECTED[@]}"; do
    # Resume: skip iterations/configs already completed
    if (( iter < RESUME_ITER )); then
      continue
    fi
    if (( iter == RESUME_ITER && cfg_idx < RESUME_CFG )); then
      continue
    fi

    IFS='|' read -r cfg_name _ <<< "${CFGS[$cfg_idx]}"
    echo ""
    echo "--- Config [${cfg_idx}]: ${cfg_name} ---"

    for coll in "${COLLECTIVES[@]}"; do
      (( current++ )) || true
      echo "[${current}/${total}]  iter=${iter}  ${cfg_name} / ${coll}"
      run_one "$cfg_idx" "$coll" "$iter"
    done
  done
done

echo ""
echo "$(timestamp)  All ${total} runs completed. Results in ${RESULTS_DIR}"
