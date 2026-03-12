#!/usr/bin/env bash
set -u -o pipefail

NODE1="${NODE1:-10.235.192.31}"
NODE2="${NODE2:-10.235.192.32}"
RANKS_PER_NODE="${RANKS_PER_NODE:-8}"
NP="${NP:-16}"

NIC_IF="${NIC_IF:-ai0}"
GID_INDEX="${GID_INDEX:-1}"

PERF_BIN="${PERF_BIN:-/opt/rccl/bin/alltoall_perf}"
SIZE_BEGIN="${SIZE_BEGIN:-1G}"
SIZE_END="${SIZE_END:-8G}"
SIZE_FACTOR="${SIZE_FACTOR:-2}"
GPUS_PER_RANK="${GPUS_PER_RANK:-1}"
AGG_ITERS="${AGG_ITERS:-1}"

RESULTS_DIR="${RESULTS_DIR:-./a2a_results}"
RUN_TAG="${RUN_TAG:-alltoall_knob_sweep_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${RESULTS_DIR}/${RUN_TAG}"

COMMON_LOG="${OUT_DIR}/common.log"
RESULTS_TSV="${OUT_DIR}/parsed_algbw.tsv"
OVERALL_TSV="${OUT_DIR}/overall_avg.tsv"
BYSIZE_TSV="${OUT_DIR}/best_test1_vs_test2_by_size.tsv"
FAMILY_TSV="${OUT_DIR}/best_family_summary.tsv"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

mkdir -p "${OUT_DIR}"

# Test1 sweep
read -r -a TEST1_GDR_VALUES <<< "${TEST1_GDR_VALUES:-0 1}"
read -r -a TEST1_INLINE_VALUES <<< "${TEST1_INLINE_VALUES:-0 1}"

# Test2 sweep
read -r -a TEST2_GDR_VALUES <<< "${TEST2_GDR_VALUES:-0 1}"
read -r -a TEST2_OPT_RECV_VALUES <<< "${TEST2_OPT_RECV_VALUES:-0 1}"
read -r -a TEST2_INLINE_VALUES <<< "${TEST2_INLINE_VALUES:-0 1}"

pretty_print() {
  local file="$1"
  if command -v column >/dev/null 2>&1; then
    column -ts $'\t' "$file"
  else
    cat "$file"
  fi
}

log() {
  echo "$*" | tee -a "${COMMON_LOG}"
}

check_prereqs() {
  if [[ ! -x "${PERF_BIN}" ]]; then
    echo "ERROR: PERF_BIN not found or not executable: ${PERF_BIN}" >&2
    exit 1
  fi
  if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found in PATH" >&2
    exit 1
  fi
}

init_outputs() {
  {
    echo -e "label\ttest_kind\tstatus\tsize_B\toop_time_us\toop_algbw_GBs\tip_time_us\tip_algbw_GBs"
  } > "${RESULTS_TSV}"

  {
    echo -e "label\ttest_kind\tstatus\tavg_oop_algbw_GBs\tavg_ip_algbw_GBs\tavg_combined_algbw_GBs\tnum_sizes"
  } > "${OVERALL_TSV}"

  {
    echo -e "size_B\tbest_test1_oop_label\tbest_test1_oop_algbw_GBs\tbest_test2_oop_label\tbest_test2_oop_algbw_GBs\toop_winner\tbest_test1_ip_label\tbest_test1_ip_algbw_GBs\tbest_test2_ip_label\tbest_test2_ip_algbw_GBs\tip_winner"
  } > "${BYSIZE_TSV}"

  {
    echo -e "family\tbest_label\tavg_combined_algbw_GBs\tavg_oop_algbw_GBs\tavg_ip_algbw_GBs\tnum_sizes"
  } > "${FAMILY_TSV}"

  {
    echo "============================================================"
    echo "RCCL alltoall knob sweep"
    echo "Started: $(date)"
    echo "Output dir: ${OUT_DIR}"
    echo "NODE1=${NODE1}"
    echo "NODE2=${NODE2}"
    echo "RANKS_PER_NODE=${RANKS_PER_NODE}"
    echo "NP=${NP}"
    echo "NIC_IF=${NIC_IF}"
    echo "GID_INDEX=${GID_INDEX}"
    echo "PERF_BIN=${PERF_BIN}"
    echo "SIZE_BEGIN=${SIZE_BEGIN}"
    echo "SIZE_END=${SIZE_END}"
    echo "SIZE_FACTOR=${SIZE_FACTOR}"
    echo "TEST1_GDR_VALUES=${TEST1_GDR_VALUES[*]}"
    echo "TEST1_INLINE_VALUES=${TEST1_INLINE_VALUES[*]}"
    echo "TEST2_GDR_VALUES=${TEST2_GDR_VALUES[*]}"
    echo "TEST2_OPT_RECV_VALUES=${TEST2_OPT_RECV_VALUES[*]}"
    echo "TEST2_INLINE_VALUES=${TEST2_INLINE_VALUES[*]}"
    echo "============================================================"
    echo
  } > "${COMMON_LOG}"
}

run_case() {
  local label="$1"
  local test_kind="$2"
  shift 2

  local tmp_log
  tmp_log="$(mktemp)"

  log "------------------------------------------------------------"
  log "Running ${label} (${test_kind})"
  log "Started: $(date)"
  printf 'Command:' | tee -a "${COMMON_LOG}"
  printf ' %q' "$@" | tee -a "${COMMON_LOG}"
  printf '\n' | tee -a "${COMMON_LOG}"
  log "------------------------------------------------------------"

  if (( DRY_RUN )); then
    printf '[DRY-RUN] %q ' "$@" | tee -a "${COMMON_LOG}"
    printf '\n' | tee -a "${COMMON_LOG}"
    echo -e "${label}\t${test_kind}\tDRY_RUN\t-\t-\t-\t-\t-" >> "${RESULTS_TSV}"
    rm -f "${tmp_log}"
    return 0
  fi

  "$@" > "${tmp_log}" 2>&1
  local rc=$?

  cat "${tmp_log}" >> "${COMMON_LOG}"
  echo >> "${COMMON_LOG}"

  if (( rc != 0 )); then
    log "Case ${label} FAILED with exit code ${rc}"
    echo -e "${label}\t${test_kind}\tFAILED\t-\t-\t-\t-\t-" >> "${RESULTS_TSV}"
    rm -f "${tmp_log}"
    return 0
  fi

  if ! awk -v label="${label}" -v kind="${test_kind}" '
    BEGIN { found=0 }
    $1 ~ /^[0-9]+$/ && $6 ~ /^[0-9.]+$/ && $7 ~ /^[0-9.]+$/ && $10 ~ /^[0-9.]+$/ && $11 ~ /^[0-9.]+$/ {
      found=1
      printf "%s\t%s\tOK\t%s\t%s\t%s\t%s\t%s\n", label, kind, $1, $6, $7, $10, $11
    }
    END { if (!found) exit 2 }
  ' "${tmp_log}" >> "${RESULTS_TSV}"; then
    log "Case ${label} finished but no parsable RCCL data rows were found"
    echo -e "${label}\t${test_kind}\tNO_DATA\t-\t-\t-\t-\t-" >> "${RESULTS_TSV}"
  else
    log "Case ${label} parsed successfully"
  fi

  rm -f "${tmp_log}"
}

build_test1_cmd() {
  local gdr_flush_disable="$1"
  local force_inline="$2"

  local -a cmd=(
    mpirun
    -np "${NP}"
    -H "${NODE1}:${RANKS_PER_NODE},${NODE2}:${RANKS_PER_NODE}"
    --map-by "ppr:${RANKS_PER_NODE}:node"
    --bind-to numa
    --allow-run-as-root
    -x LD_LIBRARY_PATH
    -x PATH
    -mca btl_tcp_if_include "${NIC_IF}"
    -x IONIC_LOCKFREE=all
    -x "NCCL_SOCKET_IFNAME=${NIC_IF}"
    -x "NCCL_IB_GID_INDEX=${GID_INDEX}"
    -x HSA_NO_SCRATCH_RECLAIM=1
    -x NCCL_IB_MERGE_NICS=0
    -x RCCL_AINIC_ROCE=1
    -x RCCL_CTS_INLINE_DATA=0
    -x RCCL_CTS_OFFLOAD_ENABLED=0
    -x RCCL_AINIC_UDMA_BALANCE=4
    -x "RCCL_AINIC_GDR_FLUSH_DISABLE=${gdr_flush_disable}"
    -x "RCCL_AINIC_FORCE_INLINE=${force_inline}"
    "${PERF_BIN}"
    -b "${SIZE_BEGIN}"
    -e "${SIZE_END}"
    -f "${SIZE_FACTOR}"
    -g "${GPUS_PER_RANK}"
    -N "${AGG_ITERS}"
  )

  printf '%s\0' "${cmd[@]}"
}

build_test2_cmd() {
  local gdr_flush_disable="$1"
  local optional_recv_completion="$2"
  local ib_use_inline="$3"

  local -a cmd=(
    mpirun
    -np "${NP}"
    -H "${NODE1}:${RANKS_PER_NODE},${NODE2}:${RANKS_PER_NODE}"
    --map-by "ppr:${RANKS_PER_NODE}:node"
    --bind-to numa
    --allow-run-as-root
    -x LD_LIBRARY_PATH
    -x PATH
    -mca btl_tcp_if_include "${NIC_IF}"
    -x IONIC_LOCKFREE=all
    -x "NCCL_SOCKET_IFNAME=${NIC_IF}"
    -x "NCCL_IB_GID_INDEX=${GID_INDEX}"
    -x HSA_NO_SCRATCH_RECLAIM=1
    -x NCCL_IB_MERGE_NICS=0
    -x RCCL_AINIC_ROCE=0
    -x "NCCL_GDR_FLUSH_DISABLE=${gdr_flush_disable}"
    -x "NCCL_NET_OPTIONAL_RECV_COMPLETION=${optional_recv_completion}"
    -x "NCCL_IB_USE_INLINE=${ib_use_inline}"
    "${PERF_BIN}"
    -b "${SIZE_BEGIN}"
    -e "${SIZE_END}"
    -f "${SIZE_FACTOR}"
    -g "${GPUS_PER_RANK}"
    -N "${AGG_ITERS}"
  )

  printf '%s\0' "${cmd[@]}"
}

generate_overall_summary() {
  awk -F'\t' '
    NR == 1 { next }
    $3 == "OK" {
      label = $1
      kind[label] = $2
      n[label]++
      sum_oop[label] += $6
      sum_ip[label]  += $8
    }
    $3 != "OK" {
      if (!seen_bad[$1]++) {
        bad_kind[$1] = $2
        bad_status[$1] = $3
      }
    }
    END {
      for (l in n) {
        avg_oop = sum_oop[l] / n[l]
        avg_ip = sum_ip[l] / n[l]
        avg_comb = (sum_oop[l] + sum_ip[l]) / (2 * n[l])
        printf "%s\t%s\tOK\t%.4f\t%.4f\t%.4f\t%d\n", l, kind[l], avg_oop, avg_ip, avg_comb, n[l]
      }
      for (l in bad_status) {
        if (!(l in n)) {
          printf "%s\t%s\t%s\t-\t-\t-\t0\n", l, bad_kind[l], bad_status[l]
        }
      }
    }
  ' "${RESULTS_TSV}" | sort -t$'\t' -k3,3 -k6,6nr >> "${OVERALL_TSV}"
}

generate_bysize_summary() {
  awk -F'\t' '
    function winner(a_label, a_val, b_label, b_val, av, bv) {
      av = a_val + 0
      bv = b_val + 0
      if (a_label == "" && b_label == "") return "N/A"
      if (a_label != "" && b_label == "") return a_label
      if (a_label == "" && b_label != "") return b_label
      if (av > bv) return a_label
      if (bv > av) return b_label
      return "TIE"
    }
    NR == 1 { next }
    $3 == "OK" {
      size = $4
      sizes[size] = 1

      if ($2 == "test1") {
        if (!(size in best1_oop) || ($6 + 0) > best1_oop[size]) {
          best1_oop[size] = $6 + 0
          best1_oop_label[size] = $1
        }
        if (!(size in best1_ip) || ($8 + 0) > best1_ip[size]) {
          best1_ip[size] = $8 + 0
          best1_ip_label[size] = $1
        }
      }

      if ($2 == "test2") {
        if (!(size in best2_oop) || ($6 + 0) > best2_oop[size]) {
          best2_oop[size] = $6 + 0
          best2_oop_label[size] = $1
        }
        if (!(size in best2_ip) || ($8 + 0) > best2_ip[size]) {
          best2_ip[size] = $8 + 0
          best2_ip_label[size] = $1
        }
      }
    }
    END {
      for (size in sizes) {
        oop_w = winner(best1_oop_label[size], best1_oop[size], best2_oop_label[size], best2_oop[size])
        ip_w  = winner(best1_ip_label[size],  best1_ip[size],  best2_ip_label[size],  best2_ip[size])

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
          size,
          (best1_oop_label[size] == "" ? "-" : best1_oop_label[size]),
          (best1_oop_label[size] == "" ? "-" : sprintf("%.4f", best1_oop[size])),
          (best2_oop_label[size] == "" ? "-" : best2_oop_label[size]),
          (best2_oop_label[size] == "" ? "-" : sprintf("%.4f", best2_oop[size])),
          oop_w,
          (best1_ip_label[size] == "" ? "-" : best1_ip_label[size]),
          (best1_ip_label[size] == "" ? "-" : sprintf("%.4f", best1_ip[size])),
          (best2_ip_label[size] == "" ? "-" : best2_ip_label[size]),
          (best2_ip_label[size] == "" ? "-" : sprintf("%.4f", best2_ip[size])),
          ip_w
      }
    }
  ' "${RESULTS_TSV}" | sort -t$'\t' -k1,1n >> "${BYSIZE_TSV}"
}

generate_family_summary() {
  awk -F'\t' '
    NR == 1 { next }
    $3 == "OK" {
      if ($2 == "test1") {
        if (!seen1 || ($6 + 0) > best1_comb) {
          best1_label = $1
          best1_oop = $4 + 0
          best1_ip  = $5 + 0
          best1_comb = $6 + 0
          best1_n = $7 + 0
          seen1 = 1
        }
      }
      if ($2 == "test2") {
        if (!seen2 || ($6 + 0) > best2_comb) {
          best2_label = $1
          best2_oop = $4 + 0
          best2_ip  = $5 + 0
          best2_comb = $6 + 0
          best2_n = $7 + 0
          seen2 = 1
        }
      }
    }
    END {
      if (seen1) printf "test1\t%s\t%.4f\t%.4f\t%.4f\t%d\n", best1_label, best1_comb, best1_oop, best1_ip, best1_n
      if (seen2) printf "test2\t%s\t%.4f\t%.4f\t%.4f\t%d\n", best2_label, best2_comb, best2_oop, best2_ip, best2_n
    }
  ' "${OVERALL_TSV}" >> "${FAMILY_TSV}"
}

append_final_summary_to_log() {
  {
    echo
    echo "============================================================"
    echo "FINAL SUMMARY"
    echo "Finished: $(date)"
    echo "============================================================"
    echo
    echo "[1] Parsed algbw per run"
    pretty_print "${RESULTS_TSV}"
    echo
    echo "[2] Overall average algbw across parsed sizes"
    pretty_print "${OVERALL_TSV}"
    echo
    echo "[3] Best Test1 vs best Test2 by size"
    pretty_print "${BYSIZE_TSV}"
    echo
    echo "[4] Best run inside each family"
    pretty_print "${FAMILY_TSV}"
    echo
    echo "Artifacts:"
    echo "  Common log:   ${COMMON_LOG}"
    echo "  Parsed TSV:   ${RESULTS_TSV}"
    echo "  Overall TSV:  ${OVERALL_TSV}"
    echo "  By-size TSV:  ${BYSIZE_TSV}"
    echo "  Family TSV:   ${FAMILY_TSV}"
  } | tee -a "${COMMON_LOG}"
}

main() {
  check_prereqs
  init_outputs

  # Test1 sweep
  for gdr in "${TEST1_GDR_VALUES[@]}"; do
    for inl in "${TEST1_INLINE_VALUES[@]}"; do
      label="test1_rccl_gdrflush${gdr}_forceinline${inl}"
      mapfile -d '' cmd < <(build_test1_cmd "${gdr}" "${inl}")
      run_case "${label}" "test1" "${cmd[@]}"
    done
  done

  # Test2 sweep
  for gdr in "${TEST2_GDR_VALUES[@]}"; do
    for opt in "${TEST2_OPT_RECV_VALUES[@]}"; do
      for inl in "${TEST2_INLINE_VALUES[@]}"; do
        label="test2_nccl_gdrflush${gdr}_optrecv${opt}_inline${inl}"
        mapfile -d '' cmd < <(build_test2_cmd "${gdr}" "${opt}" "${inl}")
        run_case "${label}" "test2" "${cmd[@]}"
      done
    done
  done

  generate_overall_summary
  generate_bysize_summary
  generate_family_summary
  append_final_summary_to_log

  echo
  echo "Done."
  echo "Common log: ${COMMON_LOG}"
}

main "$@"