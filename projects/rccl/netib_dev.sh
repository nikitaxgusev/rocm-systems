#!/bin/bash
# netib_dev.sh — development workflow for rocm_netib.patch
#
# The patch is against the ORIGINAL net_ib.cc (pre-hipify).
# The build system applies the patch, then does sed renames (hipify) to produce
# net_ib_rocm.cc. This script lets you work with either file.
#
# USAGE:
#   ./netib_dev.sh init     — create work copy from current patch (pre-hipify)
#   ./netib_dev.sh regen    — regenerate patch from work copy
#   ./netib_dev.sh deploy   — hipify work copy → build file (for quick testing)
#   ./netib_dev.sh pull     — reverse-hipify build file → work copy (if you edited build file)
#   ./netib_dev.sh verify   — verify patch applies cleanly
#   ./netib_dev.sh status   — show pending changes vs current patch

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
NET_IB_ORIG="$REPO_ROOT/src/transport/net_ib.cc"
PATCH_FILE="$REPO_ROOT/ext-src/rocm_netib.patch"
BUILD_FILE="$REPO_ROOT/build/debug/hipify/src/transport/net_ib_rocm.cc"
WORK_FILE="$REPO_ROOT/ext-src/net_ib_rocm.work.cc"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

require_file() { [ -f "$1" ] || { echo -e "${RED}ERROR: not found: $1${NC}"; exit 1; }; }

# The sed renames from cmake/rocmIb.cmake (order matters)
hipify_file() {
    local F="$1"
    sed -i \
        -e 's/NCCL_PARAM(Ib/NCCL_PARAM(RocmIb/g' \
        -e 's/RCCL_PARAM(Ib/RCCL_PARAM(RocmIb/g' \
        -e 's/ncclParamIb/ncclParamRocmIb/g' \
        -e 's/rcclParamIb/rcclParamRocmIb/g' \
        -e 's/ncclIbMergedDevs/rocmIbMergedDevs/g' \
        -e 's/ncclIbDevs/rocmIbDevs/g' \
        -e 's/ncclIbLock/rocmIbLock/g' \
        -e 's/ibProviderName/rocmIbProviderName/g' \
        -e 's/ncclIbAsyncThread/rocmIbAsyncThread/g' \
        -e 's/ncclIbGdrSupport/rocmIbGdrSupport/g' \
        -e 's/ncclIbDmaBufSupport/rocmIbDmaBufSupport/g' \
        -e 's/ncclIbInitCommDevBase/rocmIbInitCommDevBase/g' \
        -e 's/ncclIbDestroyBase/rocmIbDestroyBase/g' \
        -e 's/ncclIbRtrQp/rocmIbRtrQp/g' \
        -e 's/ncclIbRtsQp/rocmIbRtsQp/g' \
        -e 's/ForceEnableGdrdma/RocmForceEnableGdrdma/g' \
        -e 's/ncclIbCheckVProps/rocmIbCheckVProps/g' \
        -e 's/ncclIbGetRequest/rocmIbGetRequest/g' \
        -e 's/ncclIbFreeRequest/rocmIbFreeRequest/g' \
        -e 's/ncclIbRegMrDmaBufInternal/rocmIbRegMrDmaBufInternal/g' \
        -e 's/ncclIbGetNetCommDevBase/rocmIbGetNetCommDevBase/g' \
        -e 's/ncclIbDeregMrInternal/rocmIbDeregMrInternal/g' \
        -e 's/ncclIbPostFifo/rocmIbPostFifo/g' \
        -e 's/reqTypeStr/rocmIbReqTypeStr/g' \
        -e 's/rcclNetP2pPolicy/rcclRocmNetP2pPolicy/g' \
        -e 's/ncclIbMakeVDeviceInternal/rocmIbMakeVDeviceInternal/g' \
        -e 's/ncclIbMakeVDevice/rocmIbMakeVDevice/g' \
        -e 's/ncclIbInit/rocmIbInit/g' \
        -e 's/ncclIbDevices/rocmIbDevices/g' \
        -e 's/ncclIbGetPhysProperties/rocmIbGetPhysProperties/g' \
        -e 's/ncclIbGetProperties/rocmIbGetProperties/g' \
        -e 's/ncclIbListen(/rocmIbListen(/g' \
        -e 's/ncclIbListen,/rocmIbListen,/g' \
        -e 's/ncclIbConnect(/rocmIbConnect(/g' \
        -e 's/ncclIbConnect /rocmIbConnect /g' \
        -e 's/ncclIbConnect,/rocmIbConnect,/g' \
        -e 's/ncclIbAccept/rocmIbAccept/g' \
        -e 's/ncclIbTest/rocmIbTest/g' \
        -e 's/ncclIbRegMrDmaBuf/rocmIbRegMrDmaBuf/g' \
        -e 's/ncclIbRegMr/rocmIbRegMr/g' \
        -e 's/ncclIbDeregMr/rocmIbDeregMr/g' \
        -e 's/ncclIbIsend/rocmIbIsend/g' \
        -e 's/ncclIbIrecv/rocmIbIrecv/g' \
        -e 's/ncclIbIflush/rocmIbIflush/g' \
        -e 's/ncclIbCloseSend/rocmIbCloseSend/g' \
        -e 's/ncclIbCloseRecv/rocmIbCloseRecv/g' \
        -e 's/ncclIbCloseListen/rocmIbCloseListen/g' \
        -e 's/ncclIbSetNetAttr/rocmNetIbSetNetAttr/g' \
        -e 's/ncclIbFinalize/rocmNetIbFinalize/g' \
        -e 's/ncclNetIb/rocmNetIb/g' \
        "$F"
}

# Reverse the hipify renames
unhipify_file() {
    local F="$1"
    sed -i \
        -e 's/rocmNetIb/ncclNetIb/g' \
        -e 's/rocmNetIbFinalize/ncclIbFinalize/g' \
        -e 's/rocmNetIbSetNetAttr/ncclIbSetNetAttr/g' \
        -e 's/rocmIbCloseListen/ncclIbCloseListen/g' \
        -e 's/rocmIbCloseRecv/ncclIbCloseRecv/g' \
        -e 's/rocmIbCloseSend/ncclIbCloseSend/g' \
        -e 's/rocmIbIflush/ncclIbIflush/g' \
        -e 's/rocmIbIrecv/ncclIbIrecv/g' \
        -e 's/rocmIbIsend/ncclIbIsend/g' \
        -e 's/rocmIbDeregMr/ncclIbDeregMr/g' \
        -e 's/rocmIbRegMr/ncclIbRegMr/g' \
        -e 's/rocmIbRegMrDmaBuf/ncclIbRegMrDmaBuf/g' \
        -e 's/rocmIbTest/ncclIbTest/g' \
        -e 's/rocmIbAccept/ncclIbAccept/g' \
        -e 's/rocmIbConnect,/ncclIbConnect,/g' \
        -e 's/rocmIbConnect /ncclIbConnect /g' \
        -e 's/rocmIbConnect(/ncclIbConnect(/g' \
        -e 's/rocmIbListen,/ncclIbListen,/g' \
        -e 's/rocmIbListen(/ncclIbListen(/g' \
        -e 's/rocmIbGetProperties/ncclIbGetProperties/g' \
        -e 's/rocmIbGetPhysProperties/ncclIbGetPhysProperties/g' \
        -e 's/rocmIbDevices/ncclIbDevices/g' \
        -e 's/rocmIbInit/ncclIbInit/g' \
        -e 's/rocmIbMakeVDeviceInternal/ncclIbMakeVDeviceInternal/g' \
        -e 's/rocmIbMakeVDevice/ncclIbMakeVDevice/g' \
        -e 's/rcclRocmNetP2pPolicy/rcclNetP2pPolicy/g' \
        -e 's/rocmIbReqTypeStr/reqTypeStr/g' \
        -e 's/rocmIbPostFifo/ncclIbPostFifo/g' \
        -e 's/rocmIbDeregMrInternal/ncclIbDeregMrInternal/g' \
        -e 's/rocmIbGetNetCommDevBase/ncclIbGetNetCommDevBase/g' \
        -e 's/rocmIbRegMrDmaBufInternal/ncclIbRegMrDmaBufInternal/g' \
        -e 's/rocmIbFreeRequest/ncclIbFreeRequest/g' \
        -e 's/rocmIbGetRequest/ncclIbGetRequest/g' \
        -e 's/rocmIbCheckVProps/ncclIbCheckVProps/g' \
        -e 's/RocmForceEnableGdrdma/ForceEnableGdrdma/g' \
        -e 's/rocmIbRtsQp/ncclIbRtsQp/g' \
        -e 's/rocmIbRtrQp/ncclIbRtrQp/g' \
        -e 's/rocmIbDestroyBase/ncclIbDestroyBase/g' \
        -e 's/rocmIbInitCommDevBase/ncclIbInitCommDevBase/g' \
        -e 's/rocmIbDmaBufSupport/ncclIbDmaBufSupport/g' \
        -e 's/rocmIbGdrSupport/ncclIbGdrSupport/g' \
        -e 's/rocmIbAsyncThread/ncclIbAsyncThread/g' \
        -e 's/rocmIbProviderName/ibProviderName/g' \
        -e 's/rocmIbLock/ncclIbLock/g' \
        -e 's/rocmIbDevs/ncclIbDevs/g' \
        -e 's/rocmIbMergedDevs/ncclIbMergedDevs/g' \
        -e 's/rcclParamRocmIb/rcclParamIb/g' \
        -e 's/ncclParamRocmIb/ncclParamIb/g' \
        -e 's/RCCL_PARAM(RocmIb/RCCL_PARAM(Ib/g' \
        -e 's/NCCL_PARAM(RocmIb/NCCL_PARAM(Ib/g' \
        "$F"
}

cmd_init() {
    require_file "$NET_IB_ORIG"
    require_file "$PATCH_FILE"
    if [ -f "$WORK_FILE" ]; then
        echo -e "${YELLOW}Work copy already exists: $WORK_FILE${NC}"
        read -r -p "Overwrite? [y/N] " ans
        [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    fi
    cp "$NET_IB_ORIG" "$WORK_FILE"
    patch "$WORK_FILE" < "$PATCH_FILE"
    echo -e "${GREEN}✓ Work copy created: $WORK_FILE${NC}"
    echo "  Edit that file, then run:  ./netib_dev.sh deploy && ./netib_dev.sh regen"
}

cmd_regen() {
    require_file "$NET_IB_ORIG"
    require_file "$WORK_FILE"
    diff -u \
        --label "a/src/transport/net_ib.cc" \
        --label "b/src/transport/net_ib.cc" \
        "$NET_IB_ORIG" \
        "$WORK_FILE" \
        > "$PATCH_FILE" || true
    HUNK_COUNT=$(grep -c '^@@' "$PATCH_FILE" || true)
    echo -e "${GREEN}✓ Patch regenerated: $PATCH_FILE ($HUNK_COUNT hunks)${NC}"
}

cmd_deploy() {
    require_file "$WORK_FILE"
    mkdir -p "$(dirname "$BUILD_FILE")"
    cp "$WORK_FILE" "$BUILD_FILE"
    hipify_file "$BUILD_FILE"
    echo -e "${GREEN}✓ Deployed (hipified) to: $BUILD_FILE${NC}"
    echo "  Now rebuild: make -C build/debug -j"
}

cmd_pull() {
    require_file "$BUILD_FILE"
    cp "$BUILD_FILE" "$WORK_FILE"
    unhipify_file "$WORK_FILE"
    echo -e "${GREEN}✓ Pulled (un-hipified) from build file to: $WORK_FILE${NC}"
    echo "  Run './netib_dev.sh regen' to update the patch."
}

cmd_verify() {
    require_file "$NET_IB_ORIG"
    require_file "$PATCH_FILE"
    TMP=$(mktemp)
    trap "rm -f $TMP" EXIT
    cp "$NET_IB_ORIG" "$TMP"
    if patch "$TMP" < "$PATCH_FILE" > /dev/null 2>&1; then
        if [ -f "$WORK_FILE" ]; then
            if diff -q "$TMP" "$WORK_FILE" > /dev/null 2>&1; then
                echo -e "${GREEN}✓ Patch applies cleanly and matches work copy${NC}"
            else
                echo -e "${YELLOW}✓ Patch applies but differs from work copy:${NC}"
                diff -u "$TMP" "$WORK_FILE" | head -40 || true
            fi
        else
            echo -e "${GREEN}✓ Patch applies cleanly${NC}"
        fi
    else
        echo -e "${RED}✗ Patch FAILED to apply${NC}"
        exit 1
    fi
}

cmd_status() {
    if [ ! -f "$WORK_FILE" ]; then
        echo -e "${YELLOW}No work copy. Run './netib_dev.sh init' first.${NC}"
        exit 0
    fi
    TMP=$(mktemp)
    trap "rm -f $TMP" EXIT
    cp "$NET_IB_ORIG" "$TMP"
    patch -s "$TMP" < "$PATCH_FILE" 2>/dev/null || true
    if diff -q "$TMP" "$WORK_FILE" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Work copy matches current patch — no pending changes${NC}"
    else
        echo -e "${YELLOW}Pending changes in work copy vs patch:${NC}"
        diff -u "$TMP" "$WORK_FILE" || true
    fi
}

case "${1:-}" in
    init)    cmd_init ;;
    regen)   cmd_regen ;;
    deploy)  cmd_deploy ;;
    pull)    cmd_pull ;;
    verify)  cmd_verify ;;
    status)  cmd_status ;;
    *)
        echo "Usage: $0 {init|regen|deploy|pull|verify|status}"
        echo ""
        echo "  init    — create work copy by applying current patch to net_ib.cc"
        echo "  regen   — regenerate patch from work copy (clean, no hipify noise)"
        echo "  deploy  — hipify work copy and copy to build dir (fast testing)"
        echo "  pull    — reverse-hipify build file into work copy"
        echo "  verify  — confirm patch applies cleanly"
        echo "  status  — show pending changes vs current patch"
        echo ""
        echo "Typical workflow:"
        echo "  1. ./netib_dev.sh init        # create work copy"
        echo "  2. Edit ext-src/net_ib_rocm.work.cc"
        echo "  3. ./netib_dev.sh deploy      # quick test in build"
        echo "  4. ./netib_dev.sh regen       # save changes to patch"
        exit 1
        ;;
esac
