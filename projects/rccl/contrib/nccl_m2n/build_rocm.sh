#!/bin/bash
# RCCL-style hipify + hipcc build of nccl_m2n (v0.2) for gfx950 in coco-rccl.
# Produces libnccl_m2n.so + basic_api_test_local + basic_api_test_mpi.
set -uo pipefail

ROCM=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH=$ROCM/bin:$ROCM/llvm/bin:/opt/venv/bin:$PATH
HIPCC=$ROCM/bin/hipcc
HIPIFY=$ROCM/bin/hipify-perl
CXX=$ROCM/bin/amdclang++
GFX=gfx950

NCCL_M2N=/home/nigusev/m2n-enabling-wt/projects/rccl/contrib/nccl_m2n
SRC=$NCCL_M2N/src
TESTS=$NCCL_M2N/tests
TP=$NCCL_M2N/third_party

RCCL_PREFIX=/tmp/m2n-build/rccl-install
RCCL_INC=$RCCL_PREFIX/include
RCCL_LIB=$RCCL_PREFIX/lib/librccl.so

GTEST_DIR=/home/nigusev/googletest/googletest
GTEST_INC=$GTEST_DIR/include

MPI_INC=/opt/openmpi/include
MPI_LIBDIR=/opt/openmpi/lib

WORK=/tmp/m2n-build/m2n
HIP=$WORK/hipify
HSRC=$HIP/src
HTEST=$HIP/tests
OBJ=$WORK/obj
LIB=$WORK/lib
BIN=$WORK/bin
LOG=/tmp/m2n-build/m2n-build.log

rm -rf "$WORK"
mkdir -p "$HSRC" "$HTEST" "$HIP/third_party" "$OBJ" "$LIB" "$BIN"
: > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }

log "=== m2n build $(date) host=$(hostname -s) gfx=$GFX ==="

# --------------------------------------------------------------------------
# Step 1: hipify src/ and tests/ (RCCL-style: hipify-perl per file)
# --------------------------------------------------------------------------
log "--- step 1: hipify ---"
cp -a "$TP/." "$HIP/third_party/" 2>/dev/null || true
hipify_dir() {
  local indir="$1" outdir="$2"
  for f in "$indir"/*; do
    [ -f "$f" ] || continue
    local b; b=$(basename "$f")
    case "$b" in
      *.cc|*.cu|*.cuh|*.h|*.hpp)
        "$HIPIFY" -experimental -quiet-warnings "$f" -o "$outdir/$b" >> "$LOG" 2>&1 ;;
      *) cp -a "$f" "$outdir/$b" ;;
    esac
  done
}
hipify_dir "$SRC" "$HSRC"
hipify_dir "$TESTS" "$HTEST"

# --------------------------------------------------------------------------
# Step 1b: post-hipify patches (PTX atomics + gfx950 sw named barrier)
# --------------------------------------------------------------------------
log "--- step 1b: post-hipify patches ---"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_PY=$SCRIPT_DIR/patch_hipify.py
[ -f "$PATCH_PY" ] || PATCH_PY=/home/nigusev/patch_hipify.py
python3 "$PATCH_PY" "$HSRC" >> "$LOG" 2>&1 || { log "PATCH FAILED"; exit 1; }

# --------------------------------------------------------------------------
# Common flags
# --------------------------------------------------------------------------
CXXFLAGS="-std=c++17 -O3 -fPIC -Wall -D__HIP_PLATFORM_AMD__ -DNCCL_OS_LINUX \
  --offload-arch=$GFX -Wno-unused-function -Wno-sign-compare -Wno-unused-result \
  -Wno-unused-variable -Wno-unused-but-set-variable -x hip"
INC="-isystem $RCCL_INC -isystem $RCCL_INC/nccl_device -I$HSRC -isystem $HIP/third_party"

LIB_CC="m2n_config.cc m2n_group.cc reshard_call_setup.cc reshard_cache.cc \
  reshard_mesh.cc reshard_loadbalance.cc reshard_prepare.cc reshard_split_prepare.cc \
  pack_staging.cc m2n_init.cc staging_buffer.cc staging_profile.cc staging_prepare.cc"
LIB_CU="reshard_user_window.cu reshard_split_comm.cu reshard_split_user_window.cu \
  reshard_staging.cu staging_kernel.cu"
LIB_SRCS=""
for f in $LIB_CC $LIB_CU; do LIB_SRCS="$LIB_SRCS $HSRC/$f"; done

# --------------------------------------------------------------------------
# Step 2: build gtest.a
# --------------------------------------------------------------------------
log "--- step 2: gtest ---"
$CXX -std=c++17 -O2 -pthread -isystem "$GTEST_INC" -I"$GTEST_DIR" \
  -c "$GTEST_DIR/src/gtest-all.cc" -o "$OBJ/gtest-all.o" >> "$LOG" 2>&1 || { log "GTEST FAILED"; exit 1; }
ar rcs "$LIB/gtest.a" "$OBJ/gtest-all.o"

# --------------------------------------------------------------------------
# Step 3: compile 18 library TUs
# --------------------------------------------------------------------------
log "--- step 3: compile library TUs ---"
nfail=0
for f in $LIB_CC $LIB_CU; do
  log "  CXX $f"
  $HIPCC $CXXFLAGS $INC -c "$HSRC/$f" -o "$OBJ/${f}.o" >> "$LOG" 2>&1 || { log "  FAILED: $f"; nfail=$((nfail+1)); }
done
[ $nfail -eq 0 ] || { log "LIB COMPILE FAILURES: $nfail"; exit 1; }

# --------------------------------------------------------------------------
# Step 4: link libnccl_m2n.so
# --------------------------------------------------------------------------
log "--- step 4: link libnccl_m2n.so ---"
OBJS=""
for f in $LIB_CC $LIB_CU; do OBJS="$OBJS $OBJ/${f}.o"; done
$HIPCC --offload-arch=$GFX -fPIC -shared -o "$LIB/libnccl_m2n.so" $OBJS \
  "$RCCL_LIB" -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$RCCL_PREFIX/lib" >> "$LOG" 2>&1 \
  || { log "LINK libnccl_m2n.so FAILED"; exit 1; }
log "  built: $(ls -la $LIB/libnccl_m2n.so)"

# --------------------------------------------------------------------------
# Step 5: basic_api_test_local (gtest, no MPI; links lib sources directly)
# --------------------------------------------------------------------------
log "--- step 5: basic_api_test_local ---"
LOCAL_SRCS="$HTEST/nonblocking_comm_test.cc $HTEST/basic_api_test_local.cc \
  $HTEST/pack_staging_pool_test.cc $HTEST/devcomm_cache_test.cc $HTEST/test_helpers.cu"
$HIPCC $CXXFLAGS -DNCCL_M2N_TESTING -DCUDART_CB= $INC -I"$HTEST" -isystem "$GTEST_INC" \
  $LOCAL_SRCS $LIB_SRCS -x none "$LIB/gtest.a" "$RCCL_LIB" \
  -Wl,-rpath,"$RCCL_PREFIX/lib" -lpthread -ldl -o "$BIN/basic_api_test_local" >> "$LOG" 2>&1 \
  && log "  built basic_api_test_local" || { log "basic_api_test_local FAILED"; }

# --------------------------------------------------------------------------
# Step 6: basic_api_test_mpi (gtest + MPI; links libnccl_m2n.so)
# --------------------------------------------------------------------------
log "--- step 6: basic_api_test_mpi ---"
MPI_SRCS="$HTEST/basic_api_test_mpi.cc $HTEST/pipe_graph_path_test.cc $HTEST/test_helpers.cu"
$HIPCC $CXXFLAGS -DCUDART_CB= -DOMPI_SKIP_MPICXX=1 $INC -I"$HTEST" -isystem "$GTEST_INC" -I"$MPI_INC" \
  $MPI_SRCS -x none "$LIB/gtest.a" -L"$LIB" -lnccl_m2n "$RCCL_LIB" -L"$MPI_LIBDIR" -lmpi \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$RCCL_PREFIX/lib" -Wl,-rpath,"$MPI_LIBDIR" \
  -lpthread -o "$BIN/basic_api_test_mpi" >> "$LOG" 2>&1 \
  && log "  built basic_api_test_mpi" || { log "basic_api_test_mpi FAILED"; }

log "=== m2n build done $(date) ==="
log "artifacts:"; ls -la "$LIB"/libnccl_m2n.so "$BIN"/ 2>&1 | tee -a "$LOG"
