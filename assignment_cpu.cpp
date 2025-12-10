// assignment_cpu.cpp
// Reference CPU implementation for Huffman compress/decompress
// Matches CUDA file layout: 256-entry header (symbol, length, 32 bytes of bits) followed by compressed bytes.
// Encryption: simple +32 on compress, -32 on decompress (to match CUDA).

#include <bits/stdc++.h>
using namespace std;

struct HuffmanCode {
    bool bits[256];
    unsigned long length;
};

struct Node {
    unsigned char character;
    unsigned int frequency;
    Node* left;
    Node* right;
    Node(unsigned char c, unsigned int f, Node* l = nullptr, Node* r = nullptr)
        : character(c), frequency(f), left(l), right(r) {}
};

struct CompareNode {
    bool operator()(Node* const& a, Node* const& b) const {
        if (a->frequency == b->frequency) return a->character > b->character;
        return a->frequency > b->frequency;
    }
};

struct HuffmanHeaderEntry {
    uint8_t symbol;
    uint8_t code_length;
    uint8_t bits[32]; // 32 bytes == 256 bits
};

// --- Utility: read whole file into vector ---
static bool read_file_bytes(const string &path, vector<uint8_t> &out) {
    ifstream ifs(path, ios::binary | ios::ate);
    if (!ifs) return false;
    auto size = ifs.tellg();
    ifs.seekg(0, ios::beg);
    out.resize((size_t)size);
    if (!ifs.read((char*)out.data(), (streamsize)out.size())) return false;
    return true;
}

static bool write_file_bytes(const string &path, const vector<uint8_t> &data) {
    ofstream ofs(path, ios::binary);
    if (!ofs) return false;
    ofs.write((const char*)data.data(), (streamsize)data.size());
    return ofs.good();
}

// --- Encryption (match CUDA): +32 (wrap via unsigned char) ---
static void encrypt_inplace(vector<uint8_t> &buf) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = (uint8_t)((int)buf[i] + 32);
    }
}
static void decrypt_inplace(vector<uint8_t> &buf) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = (uint8_t)((int)buf[i] - 32);
    }
}

// --- Build Huffman tree (CPU) and codes ---
static void build_codes_recursive(Node* node, vector<bool> &path, vector<HuffmanCode> &codes) {
    if (!node) return;
    if (!node->left && !node->right) {
        int depth = (int)path.size();
        // copy path to codes[node->character]
        codes[node->character].length = depth;
        // zero out
        for (int i = 0; i < 256; ++i) codes[node->character].bits[i] = false;
        for (int i = 0; i < depth; ++i) codes[node->character].bits[i] = path[i];
        return;
    }
    // left = 0
    if (node->left) {
        path.push_back(false);
        build_codes_recursive(node->left, path, codes);
        path.pop_back();
    }
    // right = 1
    if (node->right) {
        path.push_back(true);
        build_codes_recursive(node->right, path, codes);
        path.pop_back();
    }
}

static int build_huffman_from_counts(const array<uint32_t,256> &counts, vector<HuffmanCode> &out_codes) {
    // priority queue of nodes
    priority_queue<Node*, vector<Node*>, CompareNode> pq;
    int symbols = 0;
    for (int i = 0; i < 256; ++i) {
        if (counts[i] > 0) {
            pq.push(new Node((unsigned char)i, counts[i]));
            ++symbols;
        }
    }
    if (symbols == 0) return 0;
    if (symbols == 1) {
        // Single symbol: assign length 1 code '0'
        out_codes.assign(256, HuffmanCode());
        for (int i=0;i<256;++i) out_codes[i].length = 0;
        Node* only = pq.top();
        out_codes[only->character].length = 1;
        for (int j=0;j<256;++j) out_codes[only->character].bits[j] = false;
        out_codes[only->character].bits[0] = false;
        // cleanup node
        delete only;
        return 1;
    }
    while (pq.size() > 1) {
        Node* l = pq.top(); pq.pop();
        Node* r = pq.top(); pq.pop();
        Node* p = new Node(0, l->frequency + r->frequency, l, r);
        pq.push(p);
    }
    Node* root = pq.top();
    // initialize codes
    out_codes.assign(256, HuffmanCode());
    for (int i = 0; i < 256; ++i) { out_codes[i].length = 0; for (int j=0;j<256;++j) out_codes[i].bits[j] = false; }
    vector<bool> path;
    build_codes_recursive(root, path, out_codes);
    // free tree
    // post-order delete
    function<void(Node*)> free_tree = [&](Node* n){
        if(!n) return;
        free_tree(n->left);
        free_tree(n->right);
        delete n;
    };
    free_tree(root);
    return symbols;
}

// --- Build decode trie from header entries (bits stored LSB-first per byte) ---
struct DecodeNode {
    int symbol; // -1 internal, >=0 leaf
    DecodeNode* left;
    DecodeNode* right;
    DecodeNode(): symbol(-1), left(nullptr), right(nullptr) {}
};

static DecodeNode* build_decode_trie_from_header(const HuffmanHeaderEntry header[256]) {
    DecodeNode* root = new DecodeNode();
    for (int i = 0; i < 256; ++i) {
        uint8_t len = header[i].code_length;
        if (len == 0) continue; // unused
        DecodeNode* cur = root;
        for (int b = 0; b < (int)len; ++b) {
            int byte_index = b / 8;
            int bit_index = b % 8;
            bool bit = (header[i].bits[byte_index] >> bit_index) & 1u;
            if (!bit) {
                if (!cur->left) cur->left = new DecodeNode();
                cur = cur->left;
            } else {
                if (!cur->right) cur->right = new DecodeNode();
                cur = cur->right;
            }
        }
        // set leaf
        cur->symbol = i;
    }
    return root;
}

static void free_decode_trie(DecodeNode* n) {
    if (!n) return;
    free_decode_trie(n->left);
    free_decode_trie(n->right);
    delete n;
}

// --- Encode using codes (single-pass atomic reservation in GPU is replaced by simple serial encoding in CPU) ---
static void encode_bitstream(const vector<uint8_t> &input_bytes, const vector<HuffmanCode> &codes, vector<uint8_t> &out_bytes, uint64_t &out_bits) {
    // Compute total bits first
    uint64_t total_bits = 0;
    for (uint8_t b : input_bytes) total_bits += codes[b].length;
    out_bits = total_bits;
    size_t out_size = (size_t)((total_bits + 7) / 8);
    out_bytes.assign(out_size, 0);
    uint64_t bitpos = 0;
    for (uint8_t c : input_bytes) {
        const HuffmanCode &code = codes[c];
        for (unsigned long i = 0; i < code.length; ++i) {
            bool bit = code.bits[i];
            if (bit) {
                size_t byteidx = (size_t)(bitpos >> 3);
                int bitidx = (int)(bitpos & 7);
                out_bytes[byteidx] |= (uint8_t)(1u << bitidx);
            }
            ++bitpos;
        }
    }
}

// --- Decode bitstream using trie ---
static bool decode_bitstream(const uint8_t *data, size_t data_bytes, uint64_t total_bits, DecodeNode* trie, vector<uint8_t> &out_symbols) {
    out_symbols.clear();
    uint64_t bitpos = 0;
    DecodeNode* cur = trie;
    while (bitpos < total_bits) {
        size_t byteidx = (size_t)(bitpos >> 3);
        int bitidx = (int)(bitpos & 7);
        bool bit = ((data[byteidx] >> bitidx) & 1u);
        if (!bit) {
            if (!cur->left) return false; // invalid encoding
            cur = cur->left;
        } else {
            if (!cur->right) return false;
            cur = cur->right;
        }
        if (cur->symbol >= 0) {
            out_symbols.push_back((uint8_t)cur->symbol);
            cur = trie;
        }
        ++bitpos;
    }
    return true;
}

// --- Helper: pack codes into header entries ---
static void build_header_from_codes(const vector<HuffmanCode> &codes, HuffmanHeaderEntry header[256]) {
    for (int i = 0; i < 256; ++i) {
        header[i].symbol = (uint8_t)i;
        header[i].code_length = (uint8_t)codes[i].length;
        memset(header[i].bits, 0, 32);
        for (unsigned int j = 0; j < codes[i].length && j < 256; ++j) {
            if (codes[i].bits[j]) {
                size_t byte_index = j / 8;
                size_t bit_index = j % 8;
                header[i].bits[byte_index] |= (uint8_t)(1u << bit_index);
            }
        }
    }
}

// --- CLI: compress mode ---
static int cmd_compress(const string &inpath) {
    vector<uint8_t> input;
    if (!read_file_bytes(inpath, input)) {
        fprintf(stderr, "Error: cannot read input file %s\n", inpath.c_str());
        return 1;
    }
    // encrypt first (match CUDA)
    encrypt_inplace(input);

    // build histogram
    array<uint32_t,256> counts;
    counts.fill(0);
    for (uint8_t b : input) counts[b]++;

    // build huffman codes
    vector<HuffmanCode> codes;
    int symbols = build_huffman_from_counts(counts, codes);
    if (symbols == 0) {
        fprintf(stderr, "Error: empty input or no symbols\n");
        return 1;
    }

    // encode bitstream
    vector<uint8_t> compressed_bytes;
    uint64_t total_bits = 0;
    encode_bitstream(input, codes, compressed_bytes, total_bits);

    // build header entries
    HuffmanHeaderEntry header[256];
    build_header_from_codes(codes, header);

    // write out file: header[256] + compressed bytes (no total_bits stored to match CUDA)
    string outfile = inpath + string(".output");
    ofstream ofs(outfile, ios::binary);
    if (!ofs) {
        fprintf(stderr, "Error: cannot open output file %s\n", outfile.c_str());
        return 1;
    }
    ofs.write((const char*)header, sizeof(header));
    ofs.write((const char*)compressed_bytes.data(), (streamsize)compressed_bytes.size());
    ofs.close();

    printf("Compressed: %s -> %s (symbols=%d, bits=%llu, bytes=%zu)\n",
        inpath.c_str(), outfile.c_str(), symbols, (unsigned long long)total_bits, compressed_bytes.size());
    return 0;
}

// --- CLI: decompress mode ---
static int cmd_decompress(const string &inpath) {
    // read whole file
    vector<uint8_t> filedata;
    if (!read_file_bytes(inpath, filedata)) {
        fprintf(stderr, "Error: cannot read input file %s\n", inpath.c_str());
        return 1;
    }
    size_t offset = 0;
    const size_t header_size = sizeof(HuffmanHeaderEntry) * 256;
    if (filedata.size() < header_size) {
        fprintf(stderr, "Error: file too small to contain header\n");
        return 1;
    }
    // read header
    HuffmanHeaderEntry header[256];
    memcpy(header, filedata.data(), header_size);
    offset += header_size;

    // optional: detect if there's a uint64 after header representing total_bits
    uint64_t total_bits = 0;
    bool have_total_bits = false;
    if (filedata.size() >= offset + sizeof(uint64_t)) {
        uint64_t candidate = 0;
        memcpy(&candidate, filedata.data() + offset, sizeof(uint64_t));
        // candidate should be <= (remaining_bytes - 8) * 8 if the uint64 is present
        size_t rem_after_candidate = filedata.size() - (offset + sizeof(uint64_t));
        uint64_t max_possible = (uint64_t)rem_after_candidate * 8ULL;
        if (candidate <= max_possible && candidate > 0) {
            // treat as present
            total_bits = candidate;
            have_total_bits = true;
            offset += sizeof(uint64_t);
        }
    }
    if (!have_total_bits) {
        // assume all remaining bytes are compressed data; use all bits
        size_t rem_bytes = filedata.size() - offset;
        total_bits = (uint64_t)rem_bytes * 8ULL;
    }

    // build decode trie
    DecodeNode* trie = build_decode_trie_from_header(header);

    // decode
    const uint8_t* compressed_ptr = filedata.data() + offset;
    size_t compressed_bytes = filedata.size() - offset;
    vector<uint8_t> decoded_symbols;
    bool ok = decode_bitstream(compressed_ptr, compressed_bytes, total_bits, trie, decoded_symbols);
    if (!ok) {
        fprintf(stderr, "Error: failed to decode bitstream (possibly wrong header or bit-length)\n");
        free_decode_trie(trie);
        return 1;
    }

    // decrypt (reverse of CUDA)
    for (size_t i = 0; i < decoded_symbols.size(); ++i) {
        decoded_symbols[i] = (uint8_t)((int)decoded_symbols[i] - 32);
    }

    // write output file: input + ".decompressed"
    string outfile = inpath + string(".decompressed");
    if (!write_file_bytes(outfile, decoded_symbols)) {
        fprintf(stderr, "Error: cannot write output file %s\n", outfile.c_str());
        free_decode_trie(trie);
        return 1;
    }

    printf("Decompressed: %s -> %s (symbols=%zu)\n", inpath.c_str(), outfile.c_str(), decoded_symbols.size());
    free_decode_trie(trie);
    return 0;
}

// --- main CLI ---
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage:\n  %s compress <infile>\n  %s decompress <infile>\n", argv[0], argv[0]);
        return 1;
    }
    string cmd = argv[1];
    string infile = argv[2];
    if (cmd == "compress") {
        return cmd_compress(infile);
    } else if (cmd == "decompress") {
        return cmd_decompress(infile);
    } else {
        fprintf(stderr, "Unknown command '%s'\n", cmd.c_str());
        return 1;
    }
}
