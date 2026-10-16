# Makefile for cuda-matmul-ladder
.RECIPEPREFIX := >
ARCH     ?= sm_75
NVCC     := nvcc
NVFLAGS  := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
CXX      := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra     # -O2 on purpose: -O3 can interchange loops and hide cache effects

WARMUP  := build/hello build/vector_add build/device_info
KERNELS := $(wildcard src/kernels/*.cuh)

.PHONY: all warmup matmul cachesim cache_effects clean
all: warmup matmul cachesim cache_effects
warmup: $(WARMUP)
matmul: build/matmul
cachesim: build/cache_sim
cache_effects: build/cache_effects

build/%: warmup/%.cu | build
> $(NVCC) $(NVFLAGS) $< -o $@

build/matmul: src/main.cu src/common.cuh $(KERNELS) | build
> $(NVCC) $(NVFLAGS) $< -o $@ -lcublas

build/cache_sim: cache_sim/cache_sim.cpp | build
> $(CXX) $(CXXFLAGS) $< -o $@

build/cache_effects: experiments/cache_effects.cpp | build
> $(CXX) $(CXXFLAGS) $< -o $@

build:
> mkdir -p build

clean:
> rm -rf build
