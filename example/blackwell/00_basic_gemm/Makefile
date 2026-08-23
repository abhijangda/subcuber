# Makefile for CUTLASS-based GEMM examples

# Variables
CUTLASS = ../../../cutlass/include/
CUTLASS_COMMON = ../../../cutlass/examples/common
UTIL_INCLUDE = ../../../cutlass/tools/util/include/
BLACKWELL_GENCODE = --generate-code=arch=compute_100a,code=[compute_100a,sm_100a]
GENCODE = $(BLACKWELL_GENCODE)

NVCC_FLAGS = -std=c++20 -O3 --expt-relaxed-constexpr -Xptxas -v,-O3 $(GENCODE)
CLANG_PATH = /home/anmoreira/llvm-project/build-gpu/bin
CXXFLAGS_COMMON =
CLANG_FLAGS = --cuda-gpu-arch=sm_86 -x cuda --cuda-path=/usr/local/cuda-12.2
INCLUDES = -I $(CUTLASS) -I $(CUTLASS_COMMON) -I $(UTIL_INCLUDE)
CUDA_LIBS = -L/usr/local/cuda-12.2/lib64 -lcudart  -lcublas 

# Conditional flags based on the compiler

ifeq ($(COMPILER),clang)
    COMPILER = $(CLANG_PATH)/clang++
    CXXFLAGS = $(CLANG_FLAGS)
else ifeq ($(COMPILER), nvcc)
    COMPILER = /usr/local/cuda/bin/nvcc
    CXXFLAGS = $(NVCC_FLAGS)
	CUDA_LIBS =
else
    COMPILER = /usr/local/cuda/bin/nvcc
	CXXFLAGS = $(NVCC_FLAGS)
	CUDA_LIBS =
endif

# Targets
all: basic_gemm_128

basic_gemm_128: basic_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) basic_gemm.cu $(INCLUDES) -o $@ $(CUDA_LIBS) -DTILE_SIZE_128

basic_gemm_256: basic_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) basic_gemm.cu $(INCLUDES) -o $@ $(CUDA_LIBS) -DTILE_SIZE_256

# Clean up generated files
clean: 
	rm -f basic_gemm_128

