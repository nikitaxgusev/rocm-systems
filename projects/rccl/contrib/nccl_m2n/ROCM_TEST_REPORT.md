# NCCL M2N on ROCm (gfx950) — build & basic API test report

## Scope

Enable and validate NCCL M2N (reshard, v0.2) on ROCm/RCCL by building it with
RCCL's own hipification approach and running the **basic API correctness
suite** that ships with the component (`contrib/nccl_m2n/tests`). These are the
project's "basic API" tests (`basic_api_test_local`, `basic_api_test_mpi`), not
the benchmarks or the placement/geometry tests.

## Provenance

| Item | Value |
| --- | --- |
| Base | `origin/develop` @ `4ed21d8f61` (fresh) |
| M2N source | `NVIDIA/nccl-extensions` `main`, `nccl_m2n` v0.2 |
| Cluster / GPU | ruby, `cv350-rck-g03-c18-18`, AMD Instinct MI350X, `gfx950` |
| Toolchain | ROCm container, `hipcc` / `hipify-perl`, `amdclang++` (LLVM) |
| RCCL | built from the same develop tree, `GPU_TARGETS=gfx950` |

## Build

End-to-end build succeeds:

- `librccl.so` — fresh develop, gfx950 (GIN enabled).
- `libnccl_m2n.so` — all 18 library translation units (13 `.cc` + 5 `.cu`).
- `basic_api_test_local` — links the library sources directly (`-DNCCL_M2N_TESTING`).

Hipification is done exactly the way RCCL does it (see "Hipification approach"
below), plus two gfx950-specific overlays applied to the hipified output:

1. **Scoped atomics** — the four inline-PTX acquire/release global atomics in
   `staging_primitives.cuh` (`st.release.{sys,gpu}`, `ld.acquire.{sys,gpu}`) are
   rewritten to HIP scoped atomics (`__hip_atomic_store/__hip_atomic_load` with
   `__HIP_MEMORY_SCOPE_SYSTEM` / `__HIP_MEMORY_SCOPE_AGENT`).
2. **Named barrier** — the inline-PTX named barrier `bar.sync id,count` in
   `staging_kernel.cu` is replaced by a block-scoped, sense-reversing
   shared-memory barrier. gfx950 has no hardware named barriers
   (`__builtin_amdgcn_s_barrier_signal/wait` require `gfx12-insts` and do not
   compile for gfx950).

The NVIDIA Hopper TMA PTX (`cp.async.bulk`, `mbarrier`) is compiled out on HIP
(`STAGING_TMA_AVAILABLE` is 0 unless `__CUDA_ARCH__ >= 900`), so it needs no
porting.

## Tests run — `basic_api_test_local`

Run on one gfx950 node, GIN enabled (`NCCL_GIN_ENABLE=1`, IB-proxy backend
`NCCL_GIN_TYPE=2`, bnxt_re NICs).

### Host-side suites (no GPU device-comm) — 32 passed, 1 failed, 5 skipped

| Suite | Passed | Failed | Skipped |
| --- | --- | --- | --- |
| `NonblockingCommWaitTest` | 8 | 0 | 0 |
| `ReshardMeshValidationTest` | 1 | 0 | 0 |
| `M2nGroupTest` | 3 | 0 | 0 |
| `NcclReshardStackTest` | 1 | 0 | 0 |
| `PackStagingPoolTest` | 13 | 0 | 0 |
| `DevCommCacheTest` | 6 | **1** | 0 |
| `LocalRankProgressTest` | 0 | 0 | 1 |
| `DevCommFenceFailureTest` | 0 | 0 | 4 |

- **Failure:** `DevCommCacheTest.CapacityFailurePreservesExistingEntriesAndHitsStillWork`.
- **Skipped:** `LocalRankProgressTest.*` (1) and `DevCommFenceFailureTest.*` (4)
  skip themselves at runtime on this configuration.

### GPU reshard suite — `Matrix/BasicApiLocalTest.Reshard` (207 cases)

All 207 reshard cases are blocked at `ncclDevCommCreate` (GIN device-comm
creation); none of the reshard transfers execute. Observed GIN behavior on
gfx950:

- `NCCL_GIN_TYPE=2` (IB proxy): the GIN support gate is **passed** — GIN
  connections are negotiated and QPs are created on the bnxt_re NICs — but
  `ncclDevCommCreate` then fails with an "unhandled system error" in the GIN
  host proxy, after which the 8-rank single-process run hangs.
- `NCCL_GIN_TYPE=7` (SDMA, intra-node): fails fast at the gate — "User requested
  GIN but not all ranks in the communicator support GIN".

The single-node, single-process 8-rank "intranet" GIN device-comm path is the
current blocker for the GPU reshard cases. The intended path for these transfers
is cross-node GIN (`basic_api_test_mpi` across multiple nodes), which is the
recommended next test.

## Hipification approach (same as RCCL)

RCCL hipifies **per source file at build time** into a parallel `hipify/` tree,
then compiles the hipified output with `hipcc`:

```
# projects/rccl/src/CMakeLists.txt
${hipify-perl_executable} -experimental -quiet-warnings ${RCCL_SOURCE_DIR}/${SRC_FILE} -o ${HIP_FILE}
```

`build_rocm.sh` uses the identical mechanism for `nccl_m2n`: it runs
`hipify-perl -experimental -quiet-warnings <src> -o <out>` on every
`.cc/.cu/.cuh/.h` under `src/` and `tests/` into `build-rocm/hipify/`, then
compiles the hipified tree with `hipcc --offload-arch=gfx950`. The only
additions are the two gfx950 overlays above, applied as a post-hipify pass
(`patch_hipify.py`) — the translation method itself is RCCL's.
