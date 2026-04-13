#!/bin/bash
#
# RCCL NIC Counter Test — Dell 300x Thor2 (2-node)
#
# Run ON a compute node:
#   cd /home/nigusev/dev/rocm-systems-telemetry/projects/rccl
#   bash run_dell_TESTS_telem.sh                  # default: 15M single size
#   bash run_dell_TESTS_telem.sh 256M             # single size
#   bash run_dell_TESTS_telem.sh 512M 2G 3        # range: 512M..2G, 3 iters
#   bash run_dell_TESTS_telem.sh 1G 1G 5          # single size via range syntax
#
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RCCL_DIR=${SCRIPT_DIR}/rccl_main_develop_install_4Apr
RCCL_TESTS_DIR=/home/nigusev/dev/rocm-systems-telemetry/projects/rccl-tests/build
MPI_DIR=/home/vkutovoi/rocSHMEM/install/ompi
ROCM_DIR=/opt/rocm

NODE1=dell300x-ccs-aus-k13-41.cs-aus.dcgpu
NODE2=dell300x-ccs-aus-k13-47.cs-aus.dcgpu
NUM_GPUS=8
NIC_IF=eno8303

# Parse args: 1 arg = single size; 2 args = begin end; 3 args = begin end iters
if [ $# -ge 2 ]; then
    MSG_BEGIN=${1}
    MSG_END=${2}
    NUM_ITERS=${3:-3}
else
    MSG_BEGIN=${1:-15M}
    MSG_END=${MSG_BEGIN}
    NUM_ITERS=${2:-5}
fi

OUTPUT_DIR=${SCRIPT_DIR}/counter_out
mkdir -p ${OUTPUT_DIR}
OUT_FILE=${OUTPUT_DIR}/alltoall_$(date +%Y%m%d_%H%M%S).out

export LD_LIBRARY_PATH=${RCCL_DIR}/lib:${MPI_DIR}/lib:${ROCM_DIR}/lib:${LD_LIBRARY_PATH}
export PATH=${RCCL_DIR}/bin:${MPI_DIR}/bin:${ROCM_DIR}/bin:${PATH}
# Auto-discover RoCE IB devices from sysfs (rocep* on kernel 6.8)
NCCL_IB_HCA_AUTO=$(ls /sys/class/infiniband/ 2>/dev/null | sort -V | paste -sd,)
export NCCL_IB_HCA=${NCCL_IB_HCA:-${NCCL_IB_HCA_AUTO}}

echo "=========================================="
echo "RCCL NIC Counter Test — Dell 300x (2-node)"
echo "=========================================="
echo "Node 1: ${NODE1}"
echo "Node 2: ${NODE2}"
echo "GPUs/node: ${NUM_GPUS}"
echo "NIC: ${NIC_IF}"
echo "Message range: ${MSG_BEGIN} .. ${MSG_END}"
echo "Iterations: ${NUM_ITERS}"
echo "Output: ${OUT_FILE}"
echo ""

if [ ! -f "${RCCL_TESTS_DIR}/alltoall_perf" ]; then
    echo "ERROR: alltoall_perf not found at ${RCCL_TESTS_DIR}/"
    exit 1
fi

echo "Running alltoall_perf with counter collection..."
echo "Launching from compute node ${NODE1}..."

cd ${SCRIPT_DIR} && \
${MPI_DIR}/bin/mpirun \
    -np $((NUM_GPUS * 2)) \
    -H ${NODE1}:${NUM_GPUS},${NODE2}:${NUM_GPUS} \
    --map-by ppr:${NUM_GPUS}:node --bind-to none \
    -x LD_LIBRARY_PATH \
    -x RCCL_TESTS_NET_COUNTER_ENABLE=1 \
    -x NCCL_IB_HCA \
    -x NCCL_DEBUG=WARN \
    -x NCCL_SOCKET_IFNAME=${NIC_IF} \
    --mca btl tcp,self \
    --mca btl_tcp_if_include ${NIC_IF} \
    --mca pml ob1 \
    --timeout 600 \
    ${RCCL_TESTS_DIR}/alltoall_perf \
    -b ${MSG_BEGIN} -e ${MSG_END} -f 2 -g 1 -c 0 -n ${NUM_ITERS} 2>&1 | /usr/bin/tee ${OUT_FILE}

echo ""
echo "Done. Output saved to: ${OUT_FILE}"
