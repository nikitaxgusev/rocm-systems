import sys
hsrc = sys.argv[1]

# ---- Patch 1: staging_primitives.cuh -- PTX scoped atomics -> HIP scoped atomics
p = hsrc + "/staging_primitives.cuh"
s = open(p).read()
repl = [
 ('asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(ptr), "l"(val) : "memory");',
  "__hip_atomic_store(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);"),
 ('asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(val) : "l"(ptr) : "memory");',
  "val = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);"),
 ('asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(ptr), "l"(val) : "memory");',
  "__hip_atomic_store(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);"),
 ('asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(val) : "l"(ptr) : "memory");',
  "val = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);"),
]
for a, b in repl:
    assert a in s, "primitives: missing pattern:\n" + a
    s = s.replace(a, b)
open(p, "w").write(s)
print("patched staging_primitives.cuh")

# ---- Patch 2: staging_kernel.cu -- PTX named barrier -> sw sense-reversing barrier
p = hsrc + "/staging_kernel.cu"
s = open(p).read()
old_fn = (
    "__device__ __forceinline__ void barrier_sync_subset(int id, int count) {\n"
    '  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");\n'
    "}")
new_fn = (
    "// gfx950 has no hardware named barriers; emulate a subset barrier with a\n"
    "// block-scoped, sense-reversing shared-memory barrier. State is zeroed once\n"
    "// at kernel entry via barrier_subset_init().\n"
    "__shared__ unsigned m2n_bar_arrive[8];\n"
    "__shared__ unsigned m2n_bar_sense[8];\n"
    "__device__ __forceinline__ void barrier_subset_init() {\n"
    "  for (int i = (int)threadIdx.x; i < 8; i += (int)blockDim.x) {\n"
    "    m2n_bar_arrive[i] = 0u;\n"
    "    m2n_bar_sense[i] = 0u;\n"
    "  }\n"
    "  __syncthreads();\n"
    "}\n"
    "__device__ __forceinline__ void barrier_sync_subset(int id, int count) {\n"
    "  unsigned s = m2n_bar_sense[id];\n"
    "  __threadfence_block();\n"
    "  unsigned a = atomicInc(&m2n_bar_arrive[id], 0x7fffffffu) + 1u;\n"
    "  if (a == (unsigned)count) {\n"
    "    m2n_bar_arrive[id] = 0u;\n"
    "    __threadfence_block();\n"
    "    atomicExch(&m2n_bar_sense[id], s ^ 1u);\n"
    "  } else {\n"
    "    while (atomicAdd(&m2n_bar_sense[id], 0u) == s) { }\n"
    "  }\n"
    "  __threadfence_block();\n"
    "}")
assert old_fn in s, "kernel: barrier_sync_subset fn not found"
s = s.replace(old_fn, new_fn)
anchor = (
    "  StagingKernelParams* __restrict__ params, ncclDevComm devComm) {\n"
    "  const int channel_id = (int)blockIdx.x;")
inj = anchor + "\n  barrier_subset_init();"
assert anchor in s, "kernel: Direct kernel entry anchor not found"
s = s.replace(anchor, inj, 1)
open(p, "w").write(s)
print("patched staging_kernel.cu")
