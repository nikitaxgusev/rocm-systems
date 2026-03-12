#!/usr/bin/env bash
# Regenerate ext-src/rocm_netib.patch from your edited net_ib_rocm.cc.
# Run from project root (projects/rccl). Uses BUILD_DIR or default build/release.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build/release}"
NET_IB_CC="$PROJECT_ROOT/src/transport/net_ib.cc"
NET_IB_ROCM_CC="$BUILD_DIR/hipify/src/transport/net_ib_rocm.cc"
PATCH_FILE="$PROJECT_ROOT/ext-src/rocm_netib.patch"
TMPDIR="${TMPDIR:-/tmp}"
WORK="$TMPDIR/rocm_netib_patch_$$"

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

if [[ ! -f "$NET_IB_CC" ]]; then
  echo "Error: $NET_IB_CC not found. Run from project root (projects/rccl)." >&2
  exit 1
fi
if [[ ! -f "$NET_IB_ROCM_CC" ]]; then
  echo "Error: $NET_IB_ROCM_CC not found. Set BUILD_DIR if your build is elsewhere." >&2
  exit 1
fi

mkdir -p "$WORK/a/src/transport" "$WORK/b/src/transport"
cp "$NET_IB_CC" "$WORK/a/src/transport/net_ib.cc"
cp "$NET_IB_ROCM_CC" "$WORK/b/src/transport/net_ib.cc"

# Reverse the sed renames from cmake/rocmIb.cmake (order: last applied first)
REV="$WORK/b/src/transport/net_ib.cc"
sed -i 's/rocmNetIbSetNetAttr/ncclIbSetNetAttr/g' "$REV"
sed -i 's/rocmNetIbFinalize/ncclIbFinalize/g' "$REV"
sed -i 's/rocmNetIb/ncclNetIb/g' "$REV"
sed -i 's/rocmIbCloseListen/ncclIbCloseListen/g' "$REV"
sed -i 's/rocmIbCloseRecv/ncclIbCloseRecv/g' "$REV"
sed -i 's/rocmIbCloseSend/ncclIbCloseSend/g' "$REV"
sed -i 's/rocmIbIflush/ncclIbIflush/g' "$REV"
sed -i 's/rocmIbIrecv/ncclIbIrecv/g' "$REV"
sed -i 's/rocmIbIsend/ncclIbIsend/g' "$REV"
sed -i 's/rocmIbDeregMr/ncclIbDeregMr/g' "$REV"
sed -i 's/rocmIbRegMr/ncclIbRegMr/g' "$REV"
sed -i 's/rocmIbRegMrDmaBuf/ncclIbRegMrDmaBuf/g' "$REV"
sed -i 's/rocmIbTest/ncclIbTest/g' "$REV"
sed -i 's/rocmIbAccept/ncclIbAccept/g' "$REV"
sed -i 's/rocmIbConnect,/ncclIbConnect,/g' "$REV"
sed -i 's/rocmIbConnect /ncclIbConnect /g' "$REV"
sed -i 's/rocmIbConnect(/ncclIbConnect(/g' "$REV"
sed -i 's/rocmIbGetProperties/ncclIbGetProperties/g' "$REV"
sed -i 's/rocmIbGetPhysProperties/ncclIbGetPhysProperties/g' "$REV"
sed -i 's/rocmIbDevices/ncclIbDevices/g' "$REV"
sed -i 's/rocmIbInit/ncclIbInit/g' "$REV"
sed -i 's/rocmIbMakeVDevice/ncclIbMakeVDevice/g' "$REV"
sed -i 's/rocmIbMakeVDeviceInternal/ncclIbMakeVDeviceInternal/g' "$REV"
sed -i 's/rcclRocmNetP2pPolicy/rcclNetP2pPolicy/g' "$REV"
sed -i 's/rocmIbReqTypeStr/reqTypeStr/g' "$REV"
sed -i 's/rocmIbPostFifo/ncclIbPostFifo/g' "$REV"
sed -i 's/rocmIbDeregMrInternal/ncclIbDeregMrInternal/g' "$REV"
sed -i 's/rocmIbGetNetCommDevBase/ncclIbGetNetCommDevBase/g' "$REV"
sed -i 's/rocmIbRegMrDmaBufInternal/ncclIbRegMrDmaBufInternal/g' "$REV"
sed -i 's/rocmIbFreeRequest/ncclIbFreeRequest/g' "$REV"
sed -i 's/rocmIbGetRequest/ncclIbGetRequest/g' "$REV"
sed -i 's/rocmIbCheckVProps/ncclIbCheckVProps/g' "$REV"
sed -i 's/RocmForceEnableGdrdma/ForceEnableGdrdma/g' "$REV"
sed -i 's/rocmIbRtsQp/ncclIbRtsQp/g' "$REV"
sed -i 's/rocmIbRtrQp/ncclIbRtrQp/g' "$REV"
sed -i 's/rocmIbDestroyBase/ncclIbDestroyBase/g' "$REV"
sed -i 's/rocmIbInitCommDevBase/ncclIbInitCommDevBase/g' "$REV"
sed -i 's/rocmIbDmaBufSupport/ncclIbDmaBufSupport/g' "$REV"
sed -i 's/rocmIbGdrSupport/ncclIbGdrSupport/g' "$REV"
sed -i 's/rocmIbAsyncThread/ncclIbAsyncThread/g' "$REV"
sed -i 's/rocmIbProviderName/ibProviderName/g' "$REV"
sed -i 's/rocmIbLock/ncclIbLock/g' "$REV"
sed -i 's/rocmIbDevs/ncclIbDevs/g' "$REV"
sed -i 's/rocmIbMergedDevs/ncclIbMergedDevs/g' "$REV"
sed -i 's/rcclParamRocmIb/rcclParamIb/g' "$REV"
sed -i 's/ncclParamRocmIb/ncclParamIb/g' "$REV"
sed -i 's/RCCL_PARAM(RocmIb/RCCL_PARAM(Ib/g' "$REV"
sed -i 's/NCCL_PARAM(RocmIb/NCCL_PARAM(Ib/g' "$REV"

(cd "$WORK" && diff -u a/src/transport/net_ib.cc b/src/transport/net_ib.cc) > "$PATCH_FILE" || true
if [[ ! -s "$PATCH_FILE" ]]; then
  echo "No diff produced (files identical?). Patch not updated." >&2
  exit 1
fi
echo "Updated $PATCH_FILE"
echo "Re-run cmake in your build dir to regenerate net_ib_rocm.cc from this patch."
