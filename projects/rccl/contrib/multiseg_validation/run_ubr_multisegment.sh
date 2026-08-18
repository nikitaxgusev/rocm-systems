#!/usr/bin/env bash
###############################################################################
# AICOMRCCL-1179 - UBR_MultiSegment single-node validation runner
#
# Builds and runs RCCL's test/RegistrationMPITests.cpp "UBR_MultiSegment" suite
# on a single node (MI300X / gfx942), reusing a PREBUILT Release librccl.so so
# the RCCL device code is NOT recompiled.
#
# Why the object-only / minimal-link trick:
#   * RegistrationMPITests.cpp lives only in the rccl-UnitTestsMPI target
#     (built when ENABLE_MPI_TESTS=ON).
#   * A full "make rccl-UnitTestsMPI" re-links the whole binary and pulls in
#     other MPI test TUs that reference librccl-internal symbols which a Release
#     build keeps hidden -> undefined-symbol link errors.
#   * So we compile ONLY the MPI framework + RegistrationMPITests.cpp objects,
#     then link a minimal binary against the prebuilt librccl.so plus a tiny
#     getHostName() shim (the one framework symbol librccl keeps internal).
#
# Assumptions (override via env):
#   BUILD   - RCCL Release build dir that already contains librccl.so.*
#   MPI_HOME- OpenMPI install (mpirun, libmpi.so)
#   HPCX    - HPC-X root providing libhcoll (OpenMPI runtime dep)
#   NP      - number of ranks (single node). 2 is the reliable default here;
#             np=8 over loopback tended to stall on this host.
#
# Run INSIDE a ROCm 7.14 container whose /opt/rocm/bin/amdclang++ actually works
# (e.g. therock 7.14 framework image). See INSTRUCTIONS.txt.
###############################################################################
set -uo pipefail

BUILD="${BUILD:?set BUILD to the RCCL Release build dir with librccl.so}"
RCCL_SRC="${RCCL_SRC:-$BUILD/../..}"           # projects/rccl
MPI_HOME="${MPI_HOME:-/opt/openmpi-5.0.8}"
HPCX="${HPCX:-/opt/hpcx-v2.23-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64}"
EXTRA_LIBS="${EXTRA_LIBS:-/home/nigusev/mpi_extra_libs}"   # libxpmem/libnl/libibverbs
NP="${NP:-2}"
FILTER="${FILTER:-*MultiSegment*}"
TDIR="$BUILD/test/CMakeFiles/rccl-UnitTestsMPI.dir"

export PATH="$MPI_HOME/bin:/opt/venv/bin:/opt/python/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$EXTRA_LIBS:$HPCX/hcoll/lib:$HPCX/ucx/lib:$HPCX/ucc/lib:$BUILD:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

echo "=== host kernel (cuMem wants >= 6.8; older prints a WARN but still works via IPC) ==="; uname -r
echo "=== amdclang++ sanity ==="; /opt/rocm/bin/amdclang++ --version | head -1 || { echo "amdclang++ broken in this image"; exit 1; }

# 1) Configure the build tree with MPI tests enabled (only regenerates makefiles).
echo "=== configure ENABLE_MPI_TESTS=ON ==="
cmake -S "$RCCL_SRC" -B "$BUILD" -DENABLE_MPI_TESTS=ON -DMPI_PATH="$MPI_HOME" >/tmp/mseg_cfg.log 2>&1 \
  || { echo "cmake configure failed"; tail -20 /tmp/mseg_cfg.log; exit 1; }

# 2) Some 7.14 images ship amd_smi headers WITHOUT the fabric types while the
#    generated flags define AMDSMI_FABRIC_DIRECT. Drop it so amdsmi_wrap.h uses
#    its own compat typedefs (host test TUs only need the types to compile).
sed -i 's/-DAMDSMI_FABRIC_DIRECT //g' "$TDIR/flags.make"

# 3) Compile ONLY this target's objects + attempt its link (skips rccl rebuild).
echo "=== compile MPI test objects (link may report undefined internal symbols; expected) ==="
( cd "$BUILD" && make -f test/CMakeFiles/rccl-UnitTestsMPI.dir/build.make \
      test/CMakeFiles/rccl-UnitTestsMPI.dir/build -j"$(nproc)" ) >/tmp/mseg_obj.log 2>&1 || true
echo "objects built: $(grep -c 'Building CXX' /tmp/mseg_obj.log)"

# 4) getHostName shim (the only framework symbol librccl keeps internal in Release).
/opt/rocm/bin/amdclang++ -O3 -std=c++17 -fPIC -c "$(dirname "$0")/gethostname_shim.cpp" -o /tmp/mseg_shim.o

# 5) Minimal link: MPI framework + RegistrationMPITests only.
echo "=== link minimal binary (framework + RegistrationMPITests + shim) ==="
cd "$BUILD/test"
O=CMakeFiles/rccl-UnitTestsMPI.dir
OBJS="$O/common/main_mpi.cpp.o $O/common/MPIHelpers.cpp.o $O/common/MPITestCore.cpp.o $O/common/MPIEnvironment.cpp.o $O/common/TestChecks.cpp.o $O/RegistrationMPITests.cpp.o"
/opt/rocm/bin/amdclang++ -O3 -Wl,-rpath,"$BUILD" -L"$MPI_HOME/lib" -Wl,-rpath,"$MPI_HOME/lib" \
  $OBJS /tmp/mseg_shim.o -o rccl-UnitTestsMPI-mseg \
  -Wl,-rpath,"$MPI_HOME/lib:$BUILD:/opt/rocm/lib" \
  ../gtest/lib/libgtest.a ../gtest/lib/libgtest_main.a -ldl \
  "$MPI_HOME/lib/libmpi.so" ../librccl.so.1.0 ../gtest/lib/libgtest.a \
  --hip-link --offload-arch=gfx942 \
  $(ls /opt/rocm/lib/libamdhip64.so.* 2>/dev/null | head -1) \
  $(ls /opt/rocm*/lib/llvm/lib/clang/*/lib/linux/libclang_rt.builtins-x86_64.a 2>/dev/null | head -1) \
  $(ls /opt/rocm/lib/libhsa-runtime64.so.* 2>/dev/null | head -1) \
  -Wl,-rpath-link,/opt/rocm/lib
BIN="$BUILD/test/rccl-UnitTestsMPI-mseg"
[ -x "$BIN" ] || { echo "LINK FAILED"; exit 1; }
echo "=== LINK OK: $BIN ==="

# 6) Required RCCL env for the UBR_MultiSegment prerequisites.
export NCCL_CUMEM_ENABLE=1 NCCL_WIN_ENABLE=1 NCCL_MULTI_SEGMENT_REGISTER=1
export NCCL_LOCAL_REGISTER=1 RCCL_MPI_LOG_ALL_RANKS=1
export NCCL_SOCKET_IFNAME=lo NCCL_DEBUG=WARN
export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NP-1)))

echo "=== run $FILTER  (mpirun -np $NP, single node) ==="
mpirun --allow-run-as-root -np "$NP" --bind-to none \
  --mca btl ^openib --mca coll ^hcoll --mca pml ob1 \
  -x LD_LIBRARY_PATH -x NCCL_CUMEM_ENABLE -x NCCL_WIN_ENABLE -x NCCL_MULTI_SEGMENT_REGISTER \
  -x NCCL_LOCAL_REGISTER -x RCCL_MPI_LOG_ALL_RANKS -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG -x HIP_VISIBLE_DEVICES \
  "$BIN" --gtest_filter="$FILTER"
echo "=== mpirun rc=$? ==="
