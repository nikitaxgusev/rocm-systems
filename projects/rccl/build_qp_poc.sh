#!/bin/bash
#SBATCH --job-name=rccl-qp-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/home/nigusev/rocm-systems-telemetry/qp_tracking_build_test.log

set -ex

RCCL_DIR=/home/nigusev/rocm-systems-telemetry/projects/rccl
RCCL_TESTS_DIR=/home/nigusev/rocm-systems-telemetry/projects/rccl-tests
BUILD_DIR=${RCCL_DIR}/build_qp

echo "=== QP Tracking POC Build & Test ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Branch: $(cd $RCCL_DIR && git branch --show-current)"
echo "Commit: $(cd $RCCL_DIR && git log --oneline -1)"
echo "ROCm: $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
echo ""

# ---- Step 1: Build RCCL with cmake (local GPU only) ----
echo "=== Step 1: Building RCCL ==="
BUILD_DIR=${RCCL_DIR}/build_qp
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LOCAL_GPU_TARGET_ONLY=ON \
      -DBUILD_TESTS=ON \
      ${RCCL_DIR} 2>&1

make -j$(nproc) 2>&1
echo ""
echo "Build artifacts:"
ls -la ${BUILD_DIR}/librccl.so* 2>/dev/null || echo "librccl.so not found"
echo ""

# ---- Step 2: Verify QP tracking symbols ----
echo "=== Step 2: Verify QP tracking symbols ==="
strings ${BUILD_DIR}/librccl.so | grep -i "QP count reached" || echo "WARN string not found"
strings ${BUILD_DIR}/librccl.so | grep -i "QP_WARN_THRESHOLD" || echo "THRESHOLD param not found"
strings ${BUILD_DIR}/librccl.so | grep "QP active=" || echo "QP active log not found"
echo ""

# ---- Step 3: Rebuild rccl-tests against new library ----
echo "=== Step 3: Rebuild rccl-tests ==="
cd ${RCCL_TESTS_DIR}
make clean 2>/dev/null || true
make -j$(nproc) NCCL_HOME=${BUILD_DIR} MPI=0 2>&1 || true
echo ""
ls -la ${RCCL_TESTS_DIR}/build/*_perf 2>/dev/null | head -5 || echo "perf binaries not found"
echo ""

# ---- Step 4: Run tests with QP tracking ----
echo "=== Step 4: Run tests with QP tracking ==="
export LD_LIBRARY_PATH=${BUILD_DIR}:${LD_LIBRARY_PATH}
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
export HSA_FORCE_FINE_GRAIN_PCIE=1

echo "Using librccl from:"
ldd ${RCCL_TESTS_DIR}/build/all_gather_perf 2>/dev/null | grep rccl || echo "check manually"
echo ""

echo "--- all_gather_perf: 2 GPUs ---"
${RCCL_TESTS_DIR}/build/all_gather_perf -b 8 -e 128M -f 2 -g 2 2>&1 || true
echo ""

echo "--- all_gather_perf: 4 GPUs ---"
${RCCL_TESTS_DIR}/build/all_gather_perf -b 8 -e 128M -f 2 -g 4 2>&1 || true
echo ""

echo "--- all_reduce_perf: 2 GPUs ---"
${RCCL_TESTS_DIR}/build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 2>&1 || true
echo ""

# ---- Step 5: Unit tests ----
echo "=== Step 5: Unit Tests ==="
if [ -f ${BUILD_DIR}/test/rccl-UnitTests ]; then
    ${BUILD_DIR}/test/rccl-UnitTests --gtest_filter="AllGather.*:AllReduce.Correctness*" 2>&1 || true
else
    echo "Unit tests binary not found"
    ls ${BUILD_DIR}/test/ 2>/dev/null || true
fi
echo ""

echo "=== Done: $(date) ==="
