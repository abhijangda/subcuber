# Makefile for Strassen-based GEMM examples

# Variables
CUTLASS = ../../../cutlass/include/
CUTLASS_COMMON = ../../../cutlass/examples/common
UTIL_INCLUDE = ../../../cutlass/tools/util/include/
STRASSEN_CUTLASS = ../../../include
STRASSEN_CUTLASS_COMMON = ../../
SRC_INCLUDE = ../../../src

SPLIT_COMPILE=1
AMPERE_GENCODE = --generate-code=arch=compute_80,code=[compute_80,sm_80]
HOPPER_GENCODE = --generate-code=arch=compute_90a,code=[compute_90a,sm_90a]
ARCH = AMPERE

ifeq ($(ARCH), HOPPER)
	GENCODE = $(HOPPER_GENCODE)
endif
ifeq ($(ARCH), AMPERE)
	GENCODE = $(AMPERE_GENCODE)
endif

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -std=c++17 -O3 -Xptxas -v --split-compile=$(SPLIT_COMPILE)
INCLUDES = -I $(STRASSEN_CUTLASS) -I $(CUTLASS) -I $(CUTLASS_COMMON) -I $(UTIL_INCLUDE) -I $(STRASSEN_CUTLASS_COMMON) -I $(SRC_INCLUDE)

# Targets
all: strassen_winograd_threadblock strassen_winograd_fused_in_sum strassen_winograd_presum_64 strassen_winograd_presum_64_splitk strassen_winograd_presum_128 strassen_winograd_presum_128_splitk strassen_winograd_presum_128_level2

# Use Ampere Gencode because H100 gencode generates nans
strassen_winograd_threadblock: strassen_gemm_fp64.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DTHREADBLOCK -DTILE_SIZE_64 -o $@ $(AMPERE_GENCODE) -DSPLIT_K -DCUTLASS_API_v2

strassen_winograd_fused_in_sum: strassen_gemm_fp64.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DFUSED_IN_SUM -DTILE_SIZE_128 -o $@ $(GENCODE) -DSPLIT_K -DCUTLASS_API_v2

strassen_winograd_presum_64: strassen_gemm_fp64.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DPRESUM -DTILE_SIZE_64 -o $@ $(GENCODE) -DSUB_GEMM_PARALLEL -DCUTLASS_API_v2

strassen_winograd_presum_64_splitk: strassen_gemm_fp64.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DPRESUM -DTILE_SIZE_64 -o $@ $(GENCODE) -DSPLIT_K -DCUTLASS_API_v2

strassen_winograd_presum_128: strassen_gemm_fp64.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DPRESUM -DTILE_SIZE_128 -o $@ $(GENCODE) -DSUB_GEMM_PARALLEL -DCUTLASS_API_v2

strassen_winograd_presum_128_splitk: strassen_gemm_fp64.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DPRESUM -DTILE_SIZE_128 -o $@ $(GENCODE) -DSPLIT_K -DCUTLASS_API_v2

strassen_winograd_presum_128_level2: strassen_gemm_fp64_level2.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DPRESUM -DTILE_SIZE_128 -o $@ $(GENCODE) -DCUTLASS_API_v2

# Clean up generated files
clean:
	rm -f strassen_winograd_threadblock strassen_winograd_fused_in_sum strassen_winograd_presum_64 strassen_winograd_presum_64_splitk strassen_winograd_presum_128 strassen_winograd_presum_128_splitk strassen_winograd_presum_128_level2
