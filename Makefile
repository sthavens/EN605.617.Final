# Compiler and flags
NVCC        := nvcc
NVCC_FLAGS  := -O3 -std=c++17

CXX         := g++
CXX_FLAGS   := -O2 -std=c++17

# Main programs
CUDA_TARGET := assignment
CUDA_SRC    := assignment.cu

CPU_TARGET  := assignment_cpu
CPU_SRC     := assignment_cpu.cpp

# Python generator
GEN_SCRIPT  := generate_test_files.py
PYTHON      := python3

.PHONY: all build clean generate test cpu cuda

all: build

###################################
# Build CUDA version
###################################
cuda: $(CUDA_TARGET)

$(CUDA_TARGET): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_SRC) -o $(CUDA_TARGET)

###################################
# Build CPU version
###################################
cpu: $(CPU_TARGET)

$(CPU_TARGET): $(CPU_SRC)
	$(CXX) $(CXX_FLAGS) $(CPU_SRC) -o $(CPU_TARGET)

###################################
# Build both versions
###################################
build: cuda cpu


###################################
# Test target:
# Builds both CUDA & CPU versions
###################################
test: build
	@echo "Both CUDA and CPU versions built."
	@echo "CUDA: $(CUDA_TARGET)"
	@echo "CPU : $(CPU_TARGET)"
	@echo "You can now run your comparison tests."


###################################
# Clean
###################################
clean:
	rm -f $(CUDA_TARGET) $(CPU_TARGET) *.o
	rm -f *.txt
	rm -rf generated/

###################################
# Data generator
###################################
# Usage:
#   make generate SIZES="4k 8k 32m" NAME=prefix COMP=0.6
#   make generate SIZES="1g"
#
# Variables:
#   SIZES = list of sizes to generate (required)
#   NAME  = optional filename prefix
#   COMP  = compressibility (0.0–1.0)
#
generate:
ifndef SIZES
	$(error You must specify SIZES="4k 8k 32m ..." )
endif
	$(PYTHON) $(GEN_SCRIPT) $(SIZES) $(if $(NAME),--name $(NAME)) $(if $(COMP),--compressibility $(COMP))
