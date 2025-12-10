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
#   make generate SIZES="4k 8k 32m" COMP=0.6
#   make generate SIZES="1g"
#
# Variables:
#   SIZES = list of sizes to generate (required)
#   COMP  = compressibility (0.0–1.0, optional, default=0)
#
generate:
ifndef SIZES
	$(error You must specify SIZES="4k 8k 32m ..." )
endif
	$(PYTHON) $(GEN_SCRIPT) $(SIZES) $(if $(COMP),--compressibility $(COMP))

###################################
# Run and time GPU & CPU versions
###################################
# Usage: make run_test
#
run_test:
	@echo "==== Running GPU version ===="
	@for f in *.txt; do \
		echo "Processing $$f..."; \
		start=$$(date +%s.%N); \
		./$(CUDA_TARGET) "$$f"; \
		end=$$(date +%s.%N); \
		time=$$(echo "$$end - $$start" | bc); \
		size_in=$$(stat -c%s "$$f"); \
		size_out=$$(stat -c%s "$$f.output"); \
		ratio=$$(echo "scale=2; $$size_out/$$size_in" | bc); \
		echo "Time: $$time sec, Compression ratio: $$ratio"; \
	done
	@echo "Cleaning .txt.output files..."
	@rm -f *.txt.output
	@echo "==== Running CPU version ===="
	@for f in *.txt; do \
		echo "Processing $$f..."; \
		start=$$(date +%s.%N); \
		./$(CPU_TARGET) compress "$$f"; \
		end=$$(date +%s.%N); \
		time=$$(echo "$$end - $$start" | bc); \
		size_in=$$(stat -c%s "$$f"); \
		size_out=$$(stat -c%s "$$f.output"); \
		ratio=$$(echo "scale=2; $$size_out/$$size_in" | bc); \
		echo "Time: $$time sec, Compression ratio: $$ratio"; \
	done
	@echo "Cleaning .txt.output files..."
	@rm -f *.txt.output
	