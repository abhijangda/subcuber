NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_90a
NVCC_FLAGS ?= -std=c++20 -O3 -lineinfo -Xptxas=-v
ROOT := ../..
TARGET := $(ROOT)/build/rmsnorm_and_quantize

.PHONY: all run run-strassen run-quantize run-quantize-strassen check clean
all: $(TARGET)

$(TARGET): rmsnorm_and_quantize.cu $(ROOT)/src/cuda/kernels/elementwise/rmsnorm_and_quantize.cuh
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) -I$(ROOT) -I$(ROOT)/cutlass/include -I$(ROOT)/cutlass/tools/util/include $< -o $@

run: $(TARGET)
	timeout 30s $(TARGET) --rows=8192 --hidden=8192

run-strassen: $(TARGET)
	timeout 30s $(TARGET) --strassen --rows=8192 --hidden=8192

run-quantize: $(TARGET)
	timeout 30s $(TARGET) --quantize-only --rows=8192 --hidden=8192

run-quantize-strassen: $(TARGET)
	timeout 30s $(TARGET) --quantize-only --strassen --rows=8192 --hidden=8192

check: $(TARGET)
	timeout 30s $(TARGET) --rows=17 --hidden=128 --iterations=0
	timeout 30s $(TARGET) --rows=257 --hidden=384 --input-padding=4 --scale-padding=7 --iterations=0
	timeout 30s $(TARGET) --strassen --rows=18 --hidden=768 --iterations=0
	timeout 30s $(TARGET) --strassen --rows=258 --hidden=768 --input-padding=4 --scale-padding=7 --iterations=0
	timeout 30s $(TARGET) --strassen --rows=2 --hidden=256 --zero-input --iterations=0
	timeout 30s $(TARGET) --strassen --rows=18 --hidden=16384 --iterations=0
	timeout 30s $(TARGET) --strassen --rows=258 --hidden=16384 --input-padding=4 --scale-padding=7 --iterations=0
	timeout 30s $(TARGET) --strassen --rows=258 --hidden=32768 --iterations=0
	timeout 30s $(TARGET) --strassen --rows=258 --hidden=65536 --zero-input --iterations=0
	timeout 30s $(TARGET) --quantize-only --rows=17 --hidden=128 --input-padding=4 --scale-padding=7 --iterations=0
	timeout 30s $(TARGET) --quantize-only --rows=257 --hidden=384 --input-padding=4 --scale-padding=7 --iterations=23 --warmup=7
	timeout 30s $(TARGET) --quantize-only --rows=2 --hidden=128 --zero-input --iterations=0
	timeout 30s $(TARGET) --quantize-only --strassen --rows=258 --hidden=768 --input-padding=4 --scale-padding=7 --iterations=23 --warmup=7
	timeout 30s $(TARGET) --quantize-only --strassen --rows=2 --hidden=256 --zero-input --iterations=0
	timeout 30s $(TARGET) --quantize-only --rows=17 --hidden=384 --input-padding=8 --scale-padding=7 --iterations=23 --warmup=7
	timeout 30s $(TARGET) --quantize-only --strassen --rows=18 --hidden=768 --input-padding=8 --scale-padding=7 --iterations=23 --warmup=7

clean:
	rm -f $(TARGET)