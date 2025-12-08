#include <stdio.h>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <queue>
#include <sys/stat.h>

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 500 * 1024 * 1024 // 500MB buffer by default
#endif

#define THREADS_PER_BLOCK 256
#define THREADS 1 << 20

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr,"%s:%d CUDA Error %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

struct HuffmanCode {
	bool bits[256]; //max code length of 256 bits
	unsigned long length; // actual length of the code in bits - allows for very large files to be read in.
};

struct Node {
	unsigned char character;
	unsigned int frequency;
	Node* left;
	Node* right;

	Node(unsigned char character, unsigned int frequency, Node* l = nullptr, Node* r = nullptr) 
		: character(character), frequency(frequency), left(l), right(r) {}
};

struct CompareNode {
	bool operator()(Node* const& n1, Node* const& n2) {
		if (n1->frequency == n2->frequency) {
			return n1->character > n2->character;
		}
		return n1->frequency > n2->frequency;
	}
};

struct HuffmanHeaderEntry {
    uint8_t symbol;          // the character value
    uint8_t code_length;     // number of bits in the code (0 if unused)
    uint32_t code_bits;      // store up to 32 bits for simplicity (for larger codes, adjust accordingly)
};

__global__ void count_characters(char* input, unsigned int* counts, size_t length) {
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadId < length) {
		char c = input[threadId];
		atomicAdd(&counts[(unsigned char) c], 1);
	}
}

__host__ int build_huffman_tree(unsigned int* counts, HuffmanCode* codes) {

	std::priority_queue<Node*, std::vector<Node*>, CompareNode> priorityQueue;
	// For now, we're just using fixed codes for demonstration purposes. Final version of project will have actual trees.
	int codes_generated = 0;
	for (unsigned int i = 0; i < 256; i++) {
		if (counts[i] > 0) {
			priorityQueue.push(new Node((unsigned char)i, counts[i]));
			codes_generated++;
		}
	}

	if (codes_generated == 0) {
		// Edge case: no characters present
		return 0;
	}

	if (codes_generated == 1) {
		bool path[256];
		Node* only_node = priorityQueue.top();
		codes[only_node->character].length = 1;
		memset(codes[only_node->character].bits, 0, 256);
		codes[only_node->character].bits[0] = false;
		return 1;
	}

	while (priorityQueue.size() > 1) {
		codes_generated++;
		Node* left = priorityQueue.top();
		priorityQueue.pop();
		Node* right = priorityQueue.top();
		priorityQueue.pop();

		Node* parent = new Node(0, left->frequency + right->frequency, left, right);
		priorityQueue.push(parent);
	}

	Node* root = priorityQueue.top();

	bool path[256] = {0};
	for (int i = 0; i < 256; i++) {
		codes[i].length = 0;
		memset(codes[i].bits, 0, 256);
	}
	build_codes_recursive(root, path, 0, codes);
	free_tree(root);
	return codes_generated;

}

__host__ void build_codes_recursive(Node* node, bool path[], int depth, HuffmanCode codes[256]) {
	if (!node) {
		fprintf(stderr, "Error: No huffman codes were produced.");
		return;
	}

	if (!node->left && !node->right) {
		codes[node->character].length = depth;
		memset(codes[node->character].bits, 0, sizeof(bool) * 256);
		for (int i = 0; i < depth; i++) {
			codes[node->character].bits[i] = path[i];
		}
		return;
	}

	if (node->left) {
		path[depth] = false;
		build_codes_recursive(node->left, path, depth + 1, codes);
	}
	if (node->right) {
		path[depth] = true;
		build_codes_recursive(node->right, path, depth + 1, codes);
	}
}

__host__ void free_tree(Node* node) {
	if (!node) return;
	free_tree(node->left);
	free_tree(node->right);
	delete node;
}

__global__ void compress_data(char* input, uint32_t* output, HuffmanCode* codes, unsigned long input_length, unsigned long long* output_length) {
	size_t threadId = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = threadId; i < input_length; i += stride) {
		const HuffmanCode& code = codes[(unsigned char) input[i]];
		unsigned long code_length = code.length;

		// Reserving a space for this particular code
		unsigned long long start_index = atomicAdd(output_length, (unsigned long long) code_length);

		for (unsigned long j = 0; j < code_length; j++) {
			bool bit = code.bits[j];
			if (bit) {
				unsigned long long bit_offset = start_index + j;
				unsigned long long byte_index = bit_offset >> 5;
				unsigned int bit_in_byte = bit_offset & 31;
				unsigned int mask = 1u << bit_in_byte;
				atomicOr(&output[byte_index], mask);
			}
		}
	}
}

__global__ void encrypt_data(char* input, char* output, size_t length) {
	size_t threadId = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;
	
	for (size_t i = threadId; i < length; i += stride) {
		output[i] = (char) ((int) input[i] + 32); // simple encryption by shifting ASCII value by 32
	}
}

__host__ int process_file(char* input) {
	cudaStream_t stream;
	cudaEvent_t start, counted, compressed, encrypted, end, copied;

	cudaEventCreate(&start);
	cudaEventCreate(&copied);
	cudaEventCreate(&counted);
	cudaEventCreate(&compressed);
	cudaEventCreate(&encrypted);
	cudaEventCreate(&end);
	cudaStreamCreate(&stream);
	cudaEventRecord(start);

	unsigned long size = 0; 
	struct stat st;
	if (stat(input, &st) == 0) {
		size = st.st_size;
	} else {
		fprintf(stderr, "Error: Unable to determine file size.\n");
		return -1;
	}
	size_t num_threads = (THREADS > size) ? THREADS : size;
	
	unsigned int* device_character_counts;
	CUDA_CHECK(cudaMalloc(&device_character_counts, sizeof(int) * 256));
	CUDA_CHECK(cudaMemset(device_character_counts, 0, sizeof(int) * 256)); // initialize counts to zero
	
	char* host_buffer;
	CUDA_CHECK(cudaMallocHost(&host_buffer, BUFFER_SIZE));
	
	void* device_buffer; 
	CUDA_CHECK(cudaMalloc(&device_buffer, size * sizeof(char)));
	
	FILE* input_file = fopen(input, "rb");
	if (!input_file) {
		fprintf(stderr, "Error: Unable to open input file.\n");
		return -1;
	}

	fseek(input_file, 0, SEEK_END);
	size_t read_bytes = fread(host_buffer, 1, BUFFER_SIZE, input_file);
	size_t length = 0;
	while (read_bytes > 0) {
		CUDA_CHECK(cudaMemcpyAsync(device_buffer + length, host_buffer, read_bytes, cudaMemcpyHostToDevice, stream));
		length += read_bytes;
		read_bytes = fread(host_buffer, 1, BUFFER_SIZE, input_file);
	}
	fclose(input_file);
	cudaEventRecord(copied);
	
	cudaStreamWaitEvent(stream, copied, 0);

	encrypt_data<<<num_threads / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>((char*) device_buffer, (char*) device_buffer, length);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	count_characters<<<num_threads / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>((char*) device_buffer, device_character_counts, length);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	cudaEventRecord(counted, stream);

	unsigned int * host_character_counts;
	CUDA_CHECK(cudaMallocHost(&host_character_counts, sizeof(int) * 256));
	CUDA_CHECK(cudaMemcpy(host_character_counts, device_character_counts, sizeof(int) * 256, cudaMemcpyDeviceToHost)); //Blocking until character counts are copied
	
	cudaStreamWaitEvent(stream, counted, 0);
	
	HuffmanCode* character_codes;
	CUDA_CHECK(cudaMallocHost(&character_codes, sizeof(HuffmanCode) * 256));
	unsigned long long* device_output_length;
	CUDA_CHECK(cudaMalloc(&device_output_length, sizeof(unsigned long long)));
	CUDA_CHECK(cudaMemset(device_output_length, 0, sizeof(unsigned long long)));
	int tree_size = build_huffman_tree(host_character_counts, character_codes);

	if (tree_size == 0) {
		fprintf(stderr, "Error: No characters to encode.\n");
		return -1;
	}

	HuffmanCode* device_character_codes;
	CUDA_CHECK(cudaMalloc(&device_character_codes, sizeof(HuffmanCode) * 256));
	CUDA_CHECK(cudaMemcpyAsync(device_character_codes, character_codes, sizeof(HuffmanCode) * 256, cudaMemcpyHostToDevice, stream));
	uint32_t* device_output_buffer;
	const unsigned long long max_code_length = 256ULL;
	unsigned long long max_bits = (unsigned long long) length * max_code_length;
	if (max_bits / max_code_length != (unsigned long long) length) {
		fprintf(stderr, "Error: Input file too large to process.\n");
		return -1;
	}
	size_t num_uints = (size_t)((max_bits + 31) >> 5);
	CUDA_CHECK(cudaMalloc(&device_output_buffer, sizeof(uint32_t) * num_uints));
	CUDA_CHECK(cudaMemset(device_output_buffer, 0, sizeof(uint32_t) * num_uints));
	compress_data<<<num_threads / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>((char*) device_buffer, device_output_buffer, device_character_codes, length, device_output_length);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	
	cudaEventRecord(compressed, stream);

	cudaStreamWaitEvent(stream, compressed, 0);

	unsigned long long total_bits = 0;
	CUDA_CHECK(cudaMemcpy(&total_bits, device_output_length, sizeof(unsigned long long), cudaMemcpyDeviceToHost)); 

	size_t output_bytes = (size_t)((total_bits + 7) >> 3);
	uint8_t* host_output = (uint8_t*) malloc(output_bytes);
	if (!host_output) {
		fprintf(stderr, "Error: Unable to allocate host output buffer.\n");
		return -1;
	}
	CUDA_CHECK(cudaMemcpy(host_output, device_output_buffer, output_bytes, cudaMemcpyDeviceToHost));

	HuffmanHeaderEntry header[256];
	for (int i = 0; i < 256; i++) {
		header[i].symbol = (uint8_t)i;
		header[i].code_length = (uint8_t)character_codes[i].length;
		
		uint32_t bits = 0;
		for (int j = 0; j < character_codes[i].length && j < 32; j++) {
			if (character_codes[i].bits[j]) {
				bits |= (1u << j);
			}
		}
		header[i].code_bits = bits;
	}

	char out_filename[1024];
	snprintf(out_filename, sizeof(out_filename), "%s.output", input);
	FILE* out = fopen(out_filename, "wb");
	fwrite(header, sizeof(HuffmanHeaderEntry), 256, out);
	fwrite(host_output, 1, output_bytes, out);
	fclose(out);
	cudaEventRecord(end, stream);

	// Free host memory
	cudaFreeHost(host_buffer);
	cudaFreeHost(host_character_counts);
	cudaFreeHost(character_codes);
	free(host_output);

	// Free device memory
	cudaFree(device_character_counts);
	cudaFree(device_buffer);
	cudaFree(device_output_length);
	cudaFree(device_character_codes);
	cudaFree(device_output_buffer);

	// Destroy stream and events
	cudaStreamDestroy(stream);
	cudaEventDestroy(start);
	cudaEventDestroy(copied);
	cudaEventDestroy(counted);
	cudaEventDestroy(compressed);
	cudaEventDestroy(encrypted);
	cudaEventDestroy(end);

	return 0;
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;

	if (argc != 2) {
		printf("Please provide only 1 argument: the filename\n");
		return -1;
	}

	

	return process_file(argv[1]);
}
