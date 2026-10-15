cat > Makefile << 'EOF'
# Makefile for cuda-matmul-ladder
.RECIPEPREFIX := >
ARCH    ?= sm_75
NVCC    := nvcc
NVFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo

WARMUP  := build/hello build/vector_add build/device_info

.PHONY: all warmup clean
all: warmup
warmup: $(WARMUP)

build/%: warmup/%.cu | build
> $(NVCC) $(NVFLAGS) $< -o $@

build:
> mkdir -p build

clean:
> rm -rf build
EOF
