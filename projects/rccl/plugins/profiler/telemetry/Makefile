#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
.DEFAULT_GOAL: build
ROCM_PATH ?= $(wildcard /opt/rocm)
CXX = $(ROCM_PATH)/lib/llvm/bin/amdclang++
BUILDDIR ?= .
HIPIFY_DIR := hipify-profiler

# Telemetry profiler plugin translation units.
SRC_FILES := plugin.cc print_event.cc
HIPIFY_SRC := $(addprefix $(HIPIFY_DIR)/,$(SRC_FILES))

build: ${BUILDDIR}/librccl-profiler-telemetry.so

${BUILDDIR}/librccl-profiler-telemetry.so: $(HIPIFY_SRC)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${BUILDDIR}
	$(CXX) -D__HIP_PLATFORM_AMD__ -I$(HIPIFY_DIR) -I$(HIPIFY_DIR)/nccl -I$(ROCM_PATH)/include -fPIC -shared -o $@ $^ -L$(ROCM_PATH)/lib -lrocprofiler-sdk-roctx

$(HIPIFY_DIR)/%.cc: %.cc
	@mkdir -p $(HIPIFY_DIR)/nccl
	@cp -f *.cc *.h $(HIPIFY_DIR)/
	@cp -f nccl/*.h $(HIPIFY_DIR)/nccl/
	@hipify-perl -inplace -quiet-warnings $(HIPIFY_DIR)/*.cc $(HIPIFY_DIR)/*.h

clean:
	rm -rf ${BUILDDIR}/librccl-profiler-telemetry.so $(HIPIFY_DIR)
