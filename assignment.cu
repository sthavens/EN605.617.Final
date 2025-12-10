/********************************************************************
 *  Huffman‑compression demo (CUDA)
 *
 *  The program:
 *   1. Reads a file into a pinned host buffer.
 *   2. Encrypts the data in‑place on the device.
 *   3. Counts character frequencies (atomic histogram).
 *   4. Builds a Huffman tree on the host.
 *   5. Copies the codes to the device and compresses the data.
 *   6. Writes a simple header + compressed bit‑stream to <file>.output
 *
 *  This file has been reformatted for readability – no functional
 *  changes were made.
 ********************************************************************/

#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <queue>
#include <sys/stat.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef BUFFER_SIZE
#define BUFFER_SIZE (500 * 1024 * 1024)   // 500 MiB default buffer
#endif

#define THREADS_PER_BLOCK 256
#define THREADS           (1 << 20)

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            fprintf(stderr, "%s:%d CUDA Error %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(e));                                \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

/* ----------------------------------------------------------------- *
 *  Data structures
 * ----------------------------------------------------------------- */
struct HuffmanCode {
    bool   bits[256];   // max code length = 256 bits
    unsigned long length;
};

struct GenericCode {
    bool*          bits;
    unsigned long  length;
};

struct Node {
    unsigned char character;
    unsigned int  frequency;
    Node*         left;
    Node*         right;

    Node(unsigned char c, unsigned int f, Node* l = nullptr, Node* r = nullptr)
        : character(c), frequency(f), left(l), right(r) {}
};

struct CompareNode {
    bool operator()(Node* const& n1, Node* const& n2) const {
        if (n1->frequency == n2->frequency)
            return n1->character > n2->character;
        return n1->frequency > n2->frequency;
    }
};

struct HuffmanHeaderEntry {
    uint8_t symbol;          // character value
    uint8_t code_length;     // number of bits (0 = unused)
    uint8_t bits[32];        // up to 32 bytes = 256 bits
};

/* ----------------------------------------------------------------- *
 *  Device kernels
 * ----------------------------------------------------------------- */
__global__ void count_characters(char* input,
                                 unsigned int* counts,
                                 size_t length)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < length) {
        unsigned char c = static_cast<unsigned char>(input[tid]);
        atomicAdd(&counts[c], 1);
    }
}

__global__ void compress_data(char*          input,
                             uint32_t*      output,
                             HuffmanCode*   codes,
                             unsigned long  input_len,
                             unsigned long long* out_len)
{
    size_t tid   = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < input_len; i += stride) {
        const HuffmanCode& code = codes[static_cast<unsigned char>(input[i])];
        unsigned long code_len = code.length;

        // Reserve space for this code in the bit‑stream
        unsigned long long start = atomicAdd(out_len,
                                            static_cast<unsigned long long>(code_len));

        for (unsigned long j = 0; j < code_len; ++j) {
            if (code.bits[j]) {
                unsigned long long bit_offset = start + j;
                unsigned long long word_idx  = bit_offset >> 5;   // /32
                unsigned int       bit_in_word = bit_offset & 31; // %32
                unsigned int       mask = 1u << bit_in_word;
                atomicOr(&output[word_idx], mask);
            }
        }
    }
}

__global__ void encrypt_data(char* input,
                            char* output,
                            size_t length)
{
    size_t tid   = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < length; i += stride) {
        // Simple Caesar‑style shift (+32)
        output[i] = static_cast<char>(static_cast<int>(input[i]) + 32);
    }
}

/* ----------------------------------------------------------------- *
 *  Host helpers
 * ----------------------------------------------------------------- */
static void build_codes_recursive(Node*          node,
                                 bool           path[],
                                 int            depth,
                                 HuffmanCode    codes[256])
{
    if (!node) {
        fprintf(stderr, "Error: No Huffman codes were produced.\n");
        return;
    }

    // Leaf node → store the accumulated path
    if (!node->left && !node->right) {
        codes[node->character].length = depth;
        memset(codes[node->character].bits, 0, sizeof(bool) * 256);
        for (int i = 0; i < depth; ++i)
            codes[node->character].bits[i] = path[i];
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

static void free_tree(Node* node)
{
    if (!node) return;
    free_tree(node->left);
    free_tree(node->right);
    delete node;
}

/* Build a Huffman tree from the histogram and fill `codes`. Returns
 * the number of distinct symbols (0 = empty input). */
static int build_huffman_tree(unsigned int* counts,
                              HuffmanCode*  codes)
{
    std::priority_queue<Node*, std::vector<Node*>, CompareNode> pq;

    // Insert all symbols that actually appear
    int distinct = 0;
    for (unsigned int i = 0; i < 256; ++i) {
        if (counts[i] > 0) {
            pq.push(new Node(static_cast<unsigned char>(i), counts[i]));
            ++distinct;
        }
    }

    if (distinct == 0) return 0;          // empty file
    if (distinct == 1) {                  // single‑symbol file
        Node* only = pq.top();
        codes[only->character].length = 1;
        memset(codes[only->character].bits, 0, 256);
        codes[only->character].bits[0] = false;
        return 1;
    }

    // Build the tree
    while (pq.size() > 1) {
        Node* left  = pq.top(); pq.pop();
        Node* right = pq.top(); pq.pop();
        Node* parent = new Node(0, left->frequency + right->frequency,
                               left, right);
        pq.push(parent);
    }

    // Generate codes
    Node* root = pq.top();
    bool path[256] = {0};
    for (int i = 0; i < 256; ++i) {
        codes[i].length = 0;
        memset(codes[i].bits, 0, 256);
    }
    build_codes_recursive(root, path, 0, codes);
    free_tree(root);
    return distinct;
}

/* ----------------------------------------------------------------- *
 *  Main processing routine
 * ----------------------------------------------------------------- */
static int process_file(const char* input_path)
{
    /* ---- CUDA stream / events ----------------------------------- */
    cudaStream_t stream;
    cudaEvent_t  start, copied, counted, compressed, encrypted, end;

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&copied));
    CUDA_CHECK(cudaEventCreate(&counted));
    CUDA_CHECK(cudaEventCreate(&compressed));
    CUDA_CHECK(cudaEventCreate(&encrypted));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(start));

    /* ---- Determine file size ------------------------------------ */
    struct stat st;
    if (stat(input_path, &st) != 0) {
        fprintf(stderr, "Error: Unable to determine file size.\n");
        return -1;
    }
    const unsigned long file_size = static_cast<unsigned long>(st.st_size);

    /* ---- Allocate host / device buffers -------------------------- */
    unsigned int*  d_counts;
    CUDA_CHECK(cudaMalloc(&d_counts, sizeof(unsigned int) * 256));
    CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(unsigned int) * 256));

    char* h_pinned_buf;
    CUDA_CHECK(cudaMallocHost(&h_pinned_buf, BUFFER_SIZE));

    void* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, file_size * sizeof(char)));

    /* ---- Read file into pinned host buffer and copy to device ---- */
    FILE* f = fopen(input_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Unable to open input file.\n");
        return -1;
    }

    size_t copied_bytes = 0;
    while (copied_bytes < file_size) {
        size_t to_read = std::min((size_t) BUFFER_SIZE,
                                 file_size - copied_bytes);
        size_t got = fread(h_pinned_buf, 1, to_read, f);
        if (got == 0) break;   // EOF / error

        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(d_input) + copied_bytes,
            h_pinned_buf, got, cudaMemcpyHostToDevice, stream));
        copied_bytes += got;
    }
    fclose(f);
    CUDA_CHECK(cudaEventRecord(copied));
    CUDA_CHECK(cudaStreamWaitEvent(stream, copied, 0));

    /* ---- Kernel launch configuration ---------------------------- */
    size_t num_threads = (THREADS > file_size) ? THREADS : file_size;
    size_t num_blocks  = (num_threads + THREADS_PER_BLOCK - 1) /
                         THREADS_PER_BLOCK;

    /* ---- Simple encryption (in‑place) -------------------------- */
    encrypt_data<<<static_cast<int>(num_blocks), THREADS_PER_BLOCK,
                  0, stream>>>(
        static_cast<char*>(d_input),
        static_cast<char*>(d_input),
        file_size);
    CUDA_CHECK(cudaGetLastError());

    /* ---- Histogram (character counts) --------------------------- */
    count_characters<<<static_cast<int>(num_blocks), THREADS_PER_BLOCK,
                       0, stream>>>(
        static_cast<char*>(d_input),
        d_counts,
        file_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(counted, stream));

    /* ---- Copy histogram back to host --------------------------- */
    unsigned int* h_counts;
    CUDA_CHECK(cudaMallocHost(&h_counts,
                              sizeof(unsigned int) * 256));
    CUDA_CHECK(cudaMemcpy(h_counts, d_counts,
                          sizeof(unsigned int) * 256,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaStreamWaitEvent(stream, counted, 0));

    /* ---- Build Huffman tree on host ---------------------------- */
    HuffmanCode* h_codes = (HuffmanCode*)malloc(sizeof(HuffmanCode) * 256);
    int distinct = build_huffman_tree(h_counts, h_codes);
    if (distinct == 0) {
        fprintf(stderr, "Error: No characters to encode.\n");
        return -1;
    }

    /* ---- Transfer codes to device ------------------------------ */
    HuffmanCode* d_codes;
    CUDA_CHECK(cudaMalloc(&d_codes,
                          sizeof(HuffmanCode) * 256));
    CUDA_CHECK(cudaMemcpyAsync(d_codes, h_codes,
                               sizeof(HuffmanCode) * 256,
                               cudaMemcpyHostToDevice, stream));

    /* ---- Allocate output buffer (worst‑case size) --------------- */
    const unsigned long long max_code_len = 256ULL;
    unsigned long long max_bits = static_cast<unsigned long long>(file_size) *
                                 max_code_len;
    if (max_bits / max_code_len != static_cast<unsigned long long>(file_size)) {
        fprintf(stderr, "Error: Input file too large to process.\n");
        return -1;
    }
    size_t num_uints = static_cast<size_t>((max_bits + 31) >> 5);
    uint32_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output,
                          sizeof(uint32_t) * num_uints));
    CUDA_CHECK(cudaMemset(d_output, 0,
                          sizeof(uint32_t) * num_uints));

    unsigned long long* d_out_len;
    CUDA_CHECK(cudaMalloc(&d_out_len,
                          sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_out_len, 0,
                          sizeof(unsigned long long)));

    /* ---- Compression kernel ------------------------------------- */
    compress_data<<<static_cast<int>(num_blocks), THREADS_PER_BLOCK,
                    0, stream>>>(
        static_cast<char*>(d_input),
        d_output,
        d_codes,
        file_size,
        d_out_len);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(compressed, stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, compressed, 0));

    /* ---- Retrieve compressed bit‑stream ------------------------ */
    unsigned long long total_bits = 0;
    CUDA_CHECK(cudaMemcpy(&total_bits, d_out_len,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    size_t out_bytes = static_cast<size_t>((total_bits + 7) >> 3);
    uint8_t* h_output = (uint8_t*)malloc(out_bytes);
    if (!h_output) {
        fprintf(stderr, "Error: Unable to allocate host output buffer.\n");
        return -1;
    }
    CUDA_CHECK(cudaMemcpy(h_output, d_output,
                          out_bytes,
                          cudaMemcpyDeviceToHost));

    /* ---- Build header (symbol → code) -------------------------- */
    HuffmanHeaderEntry header[256];
    for (int i = 0; i < 256; ++i) {
        header[i].symbol      = static_cast<uint8_t>(i);
        header[i].code_length = static_cast<uint8_t>(h_codes[i].length);
        memset(header[i].bits, 0, 32);
        for (int j = 0; j < h_codes[i].length; ++j) {
            if (h_codes[i].bits[j]) {
                size_t byte_idx = j / 8;
                size_t bit_idx  = j % 8;
                header[i].bits[byte_idx] |= (1u << bit_idx);
            }
        }
    }

    /* ---- Write output file -------------------------------------- */
    char out_name[1024];
    snprintf(out_name, sizeof(out_name), "%s.output", input_path);
    FILE* out_f = fopen(out_name, "wb");
    if (!out_f) {
        fprintf(stderr, "Error opening output file.\n");
        return -1;
    }
    fwrite(header, sizeof(HuffmanHeaderEntry), 256, out_f);
    fwrite(h_output, 1, out_bytes, out_f);
    fclose(out_f);

    CUDA_CHECK(cudaEventRecord(end, stream));

    /* ---- Cleanup ------------------------------------------------ */
    cudaFreeHost(h_pinned_buf);
    cudaFreeHost(h_counts);
    free(h_codes);
    free(h_output);

    cudaFree(d_counts);
    cudaFree(d_input);
    cudaFree(d_codes);
    cudaFree(d_output);
    cudaFree(d_out_len);

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(copied);
    cudaEventDestroy(counted);
    cudaEventDestroy(compressed);
    cudaEventDestroy(encrypted);
    cudaEventDestroy(end);

    return 0;
}

/* ----------------------------------------------------------------- *
 *  Entry point
 * ----------------------------------------------------------------- */
int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Usage: %s <input‑file>\n", argv[0]);
        return -1;
    }
    return process_file(argv[1]);
}