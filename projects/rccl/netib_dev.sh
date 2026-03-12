#!/bin/bash
# netib_dev.sh — development workflow for net_ib_rocm.cc
#
# USAGE:
#   ./netib_dev.sh init       — extract working copy from current patch
#   ./netib_dev.sh deploy     — copy working copy → build dir (after cmake overwrite)
#   ./netib_dev.sh regen      — regenerate patch from your modified working copy
#   ./netib_dev.sh verify     — verify patch applies cleanly and matches working copy
#   ./netib_dev.sh status     — show diff between working copy and current build file

set -e

# ── Configure these paths for your repo ──────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
NET_IB_ORIG="$REPO_ROOT/src/transport/net_ib.cc"
PATCH_FILE="$REPO_ROOT/ext-src/rocm_netib.patch"
BUILD_FILE="$REPO_ROOT/build/release/hipify/src/transport/net_ib_rocm.cc"
WORK_FILE="$REPO_ROOT/ext-src/net_ib_rocm.work.cc"   # your editable working copy
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

require_file() { [ -f "$1" ] || { echo -e "${RED}ERROR: not found: $1${NC}"; exit 1; }; }

cmd_init() {
    require_file "$NET_IB_ORIG"
    require_file "$PATCH_FILE"

    if [ -f "$WORK_FILE" ]; then
        echo -e "${YELLOW}Working copy already exists: $WORK_FILE${NC}"
        read -r -p "Overwrite? [y/N] " ans
        [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    fi

    cp "$NET_IB_ORIG" "$WORK_FILE"
    patch "$WORK_FILE" < "$PATCH_FILE"
    echo -e "${GREEN}✓ Working copy created: $WORK_FILE${NC}"
    echo    "  Edit that file, then run:  ./netib_dev.sh deploy   (to test in build)"
    echo    "                             ./netib_dev.sh regen    (to save patch)"
}

cmd_deploy() {
    require_file "$WORK_FILE"
    require_file "$BUILD_FILE"

    cp "$WORK_FILE" "$BUILD_FILE"
    echo -e "${GREEN}✓ Deployed to build: $BUILD_FILE${NC}"
    echo    "  Now run your build (make / ninja) from the build directory."
}

cmd_regen() {
    require_file "$NET_IB_ORIG"
    require_file "$WORK_FILE"

    diff -u \
        --label "a/src/transport/net_ib.cc" \
        --label "b/src/transport/net_ib.cc" \
        "$NET_IB_ORIG" \
        "$WORK_FILE" \
        > "$PATCH_FILE" || true   # diff exits 1 when files differ — that's expected

    HUNK_COUNT=$(grep -c '^@@' "$PATCH_FILE" || true)
    echo -e "${GREEN}✓ Patch regenerated: $PATCH_FILE  ($HUNK_COUNT hunks)${NC}"
    echo    "  Run './netib_dev.sh verify' to confirm it applies cleanly."
}

cmd_verify() {
    require_file "$NET_IB_ORIG"
    require_file "$PATCH_FILE"
    require_file "$WORK_FILE"

    TMP=$(mktemp)
    cp "$NET_IB_ORIG" "$TMP"
    patch "$TMP" < "$PATCH_FILE"

    if diff -q "$TMP" "$WORK_FILE" > /dev/null; then
        echo -e "${GREEN}✓ Patch applies cleanly and matches working copy exactly.${NC}"
    else
        echo -e "${RED}✗ Mismatch! Diff between patch-result and working copy:${NC}"
        diff "$TMP" "$WORK_FILE" || true
        rm -f "$TMP"
        exit 1
    fi
    rm -f "$TMP"
}

cmd_status() {
    require_file "$WORK_FILE"

    if [ ! -f "$BUILD_FILE" ]; then
        echo -e "${YELLOW}Build file not found (cmake not run yet?): $BUILD_FILE${NC}"
        exit 0
    fi

    if diff -q "$WORK_FILE" "$BUILD_FILE" > /dev/null; then
        echo -e "${GREEN}✓ Working copy matches build file — in sync.${NC}"
    else
        echo -e "${YELLOW}Working copy differs from build file (deploy to sync):${NC}"
        diff "$WORK_FILE" "$BUILD_FILE" || true
    fi
}

case "${1:-}" in
    init)    cmd_init ;;
    deploy)  cmd_deploy ;;
    regen)   cmd_regen ;;
    verify)  cmd_verify ;;
    status)  cmd_status ;;
    *)
        echo "Usage: $0 {init|deploy|regen|verify|status}"
        echo ""
        echo "  init    — create working copy by applying current patch to net_ib.cc"
        echo "  deploy  — copy working copy into the build dir (survives cmake reconfigure)"
        echo "  regen   — regenerate ext-src/rocm_netib.patch from your working copy"
        echo "  verify  — confirm patch round-trips cleanly"
        echo "  status  — show diff between working copy and current build file"
        exit 1
        ;;
esac
