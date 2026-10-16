# Makefile for cuda-matmul-ladder
.RECIPEPREFIX := >
ARCH    ?= sm_75
NVCC    := nvcc
NVFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo

WARMUP  := build/hello build/vector_add build/device_info
KERNELS := $(wildcard src/kernels/*.cuh)

.PHONY: all warmup matmul clean
all: warmup matmul
warmup: $(WARMUP)
matmul: build/matmul

build/%: warmup/%.cu | build
> $(NVCC) $(NVFLAGS) $< -o $@

build/matmul: src/main.cu src/common.cuh $(KERNELS) | build
> $(NVCC) $(NVFLAGS) $< -o $@ -lcublas

build:
> mkdir -p build

clean:
> rm -rf build
