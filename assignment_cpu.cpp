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
        free_tree_
