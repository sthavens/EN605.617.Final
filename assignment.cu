#include <stdio.h>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 500 * 1024 * 1024 // 500MB buffer by default
#endif

struct HuffmanCode {
	bool* bits;
	int length;
};

__global__ void count_characters(char* input, int* counts, size_t length) {
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadId < length) {
		char c = input[threadId];
		atomicAdd(&counts[(unsigned char) c], 1);
	}
}

__global__ void build_huffman_tree(int* counts, HuffmanCode* codes) {
	// For now, we're just using fixed codes for demonstration purposes. Final version of project will have actual trees.

    // Define static boolean code arrays in device memory
    static __global__ bool replacement_a[] = { 1 };
    static __global__ bool replacement_b[] = { 0, 1 };
    static __global__ bool replacement_c[] = { 0, 0, 1 };
    static __global__ bool replacement_d[] = { 0, 0, 0, 1 };

	static __global__ HuffmanCode code_a = { replacement_a, 1 };
	static __global__ HuffmanCode code_b = { replacement_b, 2 };
	static __global__ HuffmanCode code_c = { replacement_c, 3 };
	static __global__ HuffmanCode code_d = { replacement_d, 4 };

    // Assign pointers to each code
    codes[(unsigned char)'a'] = code_a;
    codes[(unsigned char)'b'] = code_b;
    codes[(unsigned char)'c'] = code_c;
    codes[(unsigned char)'d'] = code_d;
}

__global__ void compress_data(char* input, bool* output, HuffmanCode* codes, size_t buffer_size, size_t* output_length) {
	size_t current_index = 0;
	unsigned int bit_index = 0;

	while (current_index < buffer_size) {
		unsigned char c = input[current_index];
		HuffmanCode code = codes[(unsigned char)c];

		for (int i = 0; i < code.length; i++) {
			output[bit_index++] = code.bits[i];
		}
		current_index++;
	}

	*output_length = bit_index;
}

__global__ void encrypt_data(bool* input, char* output, size_t length) {
	// Again, simple run length encoding for demonstration purposes, final project will have actual encryption of some sort.
	size_t current_index = 0;
	bool last_value = false;
	int run_length = 0;
	
	for (size_t i = 0; i < length; i++) {
		if (input[i] == last_value) {
			run_length++;
		} else {
			if (run_length > 0) {
				output[current_index++] = (int) last_value;
				output[current_index++] = (char) run_length;
			}
			last_value = input[i];
			run_length = 1;
		}
	}
}

__host__ void process_file(FILE* input, char* output, size_t length) {
	cudaStream_t stream;
	cudaEvent_t start, counted, compressed, encrypted, end;

	cudaEventCreate(&start);
	cudaEventCreate(&counted);
	cudaEventCreate(&compressed);
	cudaEventCreate(&encrypted);
	cudaEventCreate(&end);
	cudaStreamCreate(&stream);
	cudaEventRecord(start);

	int* character_counts;
	cudaMalloc(&character_counts, sizeof(int) * 256);

	void* buffer = malloc(sizeof(char) * BUFFER_SIZE);
	size_t read_bytes = fread(buffer, 1, BUFFER_SIZE, input);

	count_characters<<<1, 256>>>((char*) buffer, character_counts, length);
	cudaEventRecord(counted);


}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
}
