#include "tensorflow/lite/parser/virtualized_model_parser.h"

#include <algorithm>
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The NeuralVirtualizer Authors
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <cstring>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <openssl/evp.h>

namespace tflite {
namespace parser {

// Base64 decoding lookup table
static const unsigned char base64_decode_table[256] = {
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64,
    64,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
    64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
};

// Base64 decoding function
// Converts Base64-encoded string to raw bytes
static std::vector<uint8_t> base64_decode(const std::string& encoded) {
    std::vector<uint8_t> result;
    if (encoded.empty()) return result;

    size_t in_len = encoded.size();
    size_t i = 0;
    unsigned char char_array_4[4], char_array_3[3];
    int j = 0;

    for (char c : encoded) {
        if (c == '=') break;
        unsigned char val = base64_decode_table[static_cast<unsigned char>(c)];
        if (val == 64) continue;  // Skip invalid characters

        char_array_4[j++] = val;
        if (j == 4) {
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (j = 0; j < 3; j++) {
                result.push_back(char_array_3[j]);
            }
            j = 0;
        }
    }

    if (j > 0) {
        for (int k = j; k < 4; k++) {
            char_array_4[k] = 0;
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

        for (int k = 0; k < j - 1; k++) {
            result.push_back(char_array_3[k]);
        }
    }

    return result;
}

// Operator type mapping
const std::map<int, std::string> op_type_mapping = {
        {0, "ADD"}, {1, "AVERAGE_POOL_2D"}, {2, "CONCATENATION"}, {3, "CONV_2D"},
        {4, "DEPTHWISE_CONV_2D"}, {5, "DEPTH_TO_SPACE"}, {6, "DEQUANTIZE"},
        {7, "EMBEDDING_LOOKUP"}, {8, "FLOOR"}, {9, "FULLY_CONNECTED"},
        {10, "HASHTABLE_LOOKUP"}, {11, "L2_NORMALIZATION"}, {12, "L2_POOL_2D"},
        {13, "LOCAL_RESPONSE_NORMALIZATION"}, {14, "LOGISTIC"}, {15, "LSH_PROJECTION"},
        {16, "LSTM"}, {17, "MAX_POOL_2D"}, {18, "MUL"}, {19, "RELU"},
        {20, "RELU_N1_TO_1"}, {21, "RELU6"}, {22, "RESHAPE"}, {23, "RESIZE_BILINEAR"},
        {24, "RNN"}, {25, "SOFTMAX"}, {26, "SPACE_TO_DEPTH"}, {27, "SVDF"},
        {28, "TANH"}, {29, "CONCAT_EMBEDDINGS"}, {30, "SKIP_GRAM"}, {31, "CALL"},
        {32, "CUSTOM"}, {33, "EMBEDDING_LOOKUP_SPARSE"}, {34, "PAD"},
        {35, "UNIDIRECTIONAL_SEQUENCE_RNN"}, {36, "GATHER"}, {37, "BATCH_TO_SPACE_ND"},
        {38, "SPACE_TO_BATCH_ND"}, {39, "TRANSPOSE"}, {40, "MEAN"}, {41, "SUB"},
        {42, "DIV"}, {43, "SQUEEZE"}, {44, "UNIDIRECTIONAL_SEQUENCE_LSTM"},
        {45, "STRIDED_SLICE"}, {46, "BIDIRECTIONAL_SEQUENCE_RNN"}, {47, "EXP"},
        {48, "TOPK_V2"}, {49, "SPLIT"}, {50, "LOG_SOFTMAX"}, {51, "DELEGATE"},
        {52, "BIDIRECTIONAL_SEQUENCE_LSTM"}, {53, "CAST"}, {54, "PRELU"},
        {55, "MAXIMUM"}, {56, "ARG_MAX"}, {57, "MINIMUM"}, {58, "LESS"},
        {59, "NEG"}, {60, "PADV2"}, {61, "GREATER"}, {62, "GREATER_EQUAL"},
        {63, "LESS_EQUAL"}, {64, "SELECT"}, {65, "SLICE"}, {66, "SIN"},
        {67, "TRANSPOSE_CONV"}, {68, "SPARSE_TO_DENSE"}, {69, "TILE"},
        {70, "EXPAND_DIMS"}, {71, "EQUAL"}, {72, "NOT_EQUAL"}, {73, "LOG"},
        {74, "SUM"}, {75, "SQRT"}, {76, "RSQRT"}, {77, "SHAPE"},
        {78, "POW"}, {79, "ARG_MIN"}, {80, "FAKE_QUANT"}, {81, "REDUCE_PROD"},
        {82, "REDUCE_MAX"}, {83, "PACK"}, {84, "LOGICAL_OR"}, {85, "ONE_HOT"},
        {86, "LOGICAL_AND"}, {87, "LOGICAL_NOT"}, {88, "UNPACK"}, {89, "REDUCE_MIN"},
        {90, "FLOOR_DIV"}, {91, "REDUCE_ANY"}, {92, "SQUARE"}, {93, "ZEROS_LIKE"},
        {94, "FILL"}, {95, "FLOOR_MOD"}, {96, "RANGE"}, {97, "RESIZE_NEAREST_NEIGHBOR"},
        {98, "LEAKY_RELU"}, {99, "SQUARED_DIFFERENCE"}, {100, "MIRROR_PAD"},
        {101, "ABS"}, {102, "SPLIT_V"}, {103, "UNIQUE"}, {104, "CEIL"},
        {105, "REVERSE_V2"}, {106, "ADD_N"}, {107, "GATHER_ND"}, {108, "COS"},
        {109, "WHERE"}, {110, "RANK"}, {111, "ELU"}, {112, "REVERSE_SEQUENCE"},
        {113, "MATRIX_DIAG"}, {114, "QUANTIZE"}, {115, "MATRIX_SET_DIAG"},
        {116, "ROUND"}, {117, "HARD_SWISH"}, {118, "IF"}, {119, "WHILE"},
        {120, "NON_MAX_SUPPRESSION_V4"}, {121, "NON_MAX_SUPPRESSION_V5"},
        {122, "SCATTER_ND"}, {123, "SELECT_V2"}, {124, "DENSIFY"},
        {125, "SEGMENT_SUM"}, {126, "BATCH_MATMUL"}, {127, "PLACEHOLDER_FOR_GREATER_OP_CODES"},
        {128, "CUMSUM"}, {129, "CALL_ONCE"}, {130, "BROADCAST_TO"},
        {131, "RFFT2D"}, {132, "CONV_3D"}, {133, "IMAG"}, {134, "REAL"},
        {135, "COMPLEX_ABS"}, {136, "HASHTABLE"}, {137, "HASHTABLE_FIND"},
        {138, "HASHTABLE_IMPORT"}, {139, "HASHTABLE_SIZE"}, {140, "REDUCE_ALL"},
        {141, "CONV_3D_TRANSPOSE"}, {142, "VAR_HANDLE"}, {143, "READ_VARIABLE"},
        {144, "ASSIGN_VARIABLE"}, {145, "BROADCAST_ARGS"}, {146, "RANDOM_STANDARD_NORMAL"},
        {147, "BUCKETIZE"}, {148, "RANDOM_UNIFORM"}, {149, "MULTINOMIAL"},
        {150, "GELU"}, {151, "DYNAMIC_UPDATE_SLICE"}, {152, "RELU_0_TO_1"},
        {153, "UNSORTED_SEGMENT_PROD"}, {154, "UNSORTED_SEGMENT_MAX"},
        {155, "UNSORTED_SEGMENT_SUM"}, {156, "ATAN2"}, {157, "UNSORTED_SEGMENT_MIN"},
        {158, "SIGN"}, {159, "BITCAST"}, {160, "BITWISE_XOR"}, {161, "RIGHT_SHIFT"},
        {162, "STABLEHLO_LOGISTIC"}, {163, "STABLEHLO_ADD"}, {164, "STABLEHLO_DIVIDE"},
        {165, "STABLEHLO_MULTIPLY"}, {166, "STABLEHLO_MAXIMUM"}, {167, "STABLEHLO_RESHAPE"},
        {168, "STABLEHLO_CLAMP"}, {169, "STABLEHLO_CONCATENATE"}, {170, "STABLEHLO_BROADCAST_IN_DIM"},
        {171, "STABLEHLO_CONVOLUTION"}, {172, "STABLEHLO_SLICE"}, {173, "STABLEHLO_CUSTOM_CALL"},
        {174, "STABLEHLO_REDUCE"}, {175, "STABLEHLO_ABS"}, {176, "STABLEHLO_AND"},
        {177, "STABLEHLO_COSINE"}, {178, "STABLEHLO_EXPONENTIAL"}, {179, "STABLEHLO_FLOOR"},
        {180, "STABLEHLO_LOG"}, {181, "STABLEHLO_MINIMUM"}, {182, "STABLEHLO_NEGATE"},
        {183, "STABLEHLO_OR"}, {184, "STABLEHLO_POWER"}, {185, "STABLEHLO_REMAINDER"},
        {186, "STABLEHLO_RSQRT"}, {187, "STABLEHLO_SELECT"}, {188, "STABLEHLO_SUBTRACT"},
        {189, "STABLEHLO_TANH"}};

// Operators with weights/biases
const std::set<std::string> PARAM_OPS = {
        "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_3D", "CONV_3D_TRANSPOSE",
        "TRANSPOSE_CONV", "FULLY_CONNECTED", "SVDF", "LSTM",
        "UNIDIRECTIONAL_SEQUENCE_LSTM", "BIDIRECTIONAL_SEQUENCE_LSTM",
        "RNN", "UNIDIRECTIONAL_SEQUENCE_RNN", "BIDIRECTIONAL_SEQUENCE_RNN",
        "MEAN", "RESHAPE", "RESIZE_BILINEAR", "TRANSPOSE", "GATHER",
        "SPLIT", "ADD", "MUL", "SUB", "SQUARED_DIFFERENCE", "BATCH_MATMUL",
        "STRIDED_SLICE", "PACK", "RANGE", "EXPAND_DIMS", "CAST", "GREATER_EQUAL", "SHAPE"};

// Mandatory parameter operators
const std::set<std::string> MANDATORY_PARAM_OPS = {
        "CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV", "CONV_3D",
        "CONV_3D_TRANSPOSE", "FULLY_CONNECTED", "SVDF", "LSTM",
        "UNIDIRECTIONAL_SEQUENCE_LSTM", "BIDIRECTIONAL_SEQUENCE_LSTM", "RNN",
        "UNIDIRECTIONAL_SEQUENCE_RNN", "BIDIRECTIONAL_SEQUENCE_RNN"};

std::string get_op_type_from_deprecated_builtin_code(int deprecated_builtin_code) {
    auto it = op_type_mapping.find(deprecated_builtin_code);
    if (it != op_type_mapping.end()) {
        return it->second;
    }
    return "UNKNOWN_OP_" + std::to_string(deprecated_builtin_code);
}

nlohmann::json convert_dtype_for_json(const nlohmann::json& obj) {
    return obj;
}

void VirtualizedModelParser::Fail(const std::string& message) {
    if (ok_) {
        ok_ = false;
        error_message_ = message;
    }
}

// AES-256-CTR decryption implementation
// Uses OpenSSL EVP API for AES-256-CTR mode decryption
// Parameters:
//   - ciphertext: Encrypted data to decrypt
//   - key: 256-bit (32-byte) encryption key
//   - nonce: 128-bit (16-byte) nonce (8-byte base + 8-byte counter)
// Returns: Decrypted plaintext as vector of bytes
std::vector<uint8_t> VirtualizedModelParser::aes_ctr_decrypt(
    const std::vector<uint8_t>& ciphertext,
    const std::array<uint8_t, 32>& key,
    const std::array<uint8_t, 16>& nonce) {

    std::vector<uint8_t> plaintext(ciphertext.size());

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        return {};
    }

    // Initialize AES-256-CTR decryption
    // Note: In CTR mode, encryption and decryption are identical operations
    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_ctr(), nullptr, key.data(), nonce.data()) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return {};
    }

    int len = 0;
    int plaintext_len = 0;

    // Decrypt the ciphertext
    if (EVP_DecryptUpdate(ctx, plaintext.data(), &len, ciphertext.data(),
                         static_cast<int>(ciphertext.size())) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return {};
    }
    plaintext_len = len;

    // Finalize decryption (CTR mode typically has no padding)
    if (EVP_DecryptFinal_ex(ctx, plaintext.data() + len, &len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return {};
    }
    plaintext_len += len;

    EVP_CIPHER_CTX_free(ctx);

    plaintext.resize(plaintext_len);
    return plaintext;
}

// XOR-based key deobfuscation implementation
// Recovers the real encryption key from the obfuscated key stored in the binary
// This prevents direct key extraction through static analysis of the compiled binary
std::array<uint8_t, 32> VirtualizedModelParser::deobfuscate_key(
    const std::array<uint8_t, 32>& obfuscated_key,
    uint8_t domain_offset) const {

    std::array<uint8_t, 32> real_key;

    // Apply XOR deobfuscation byte-by-byte
    // Formula: real_key[i] = obfuscated_key[i] XOR master_mask[i] XOR domain_offset
    for (size_t i = 0; i < 32; ++i) {
        real_key[i] = obfuscated_key[i] ^ xor_master_mask_[i] ^ domain_offset;
    }

    return real_key;
}

VirtualizedModelParser::VirtualizedModelParser(const std::string& v_infos_path,
                                                                                            const std::string& params_path,
                                                                                            bool use_mmap)
        : v_infos_path_(v_infos_path), params_path_(params_path), use_mmap_(use_mmap) {
    // Initialize mmap-related fields
    mmap_addr_ = nullptr;
    mmap_size_ = 0;
#ifdef _WIN32
    file_handle_ = INVALID_HANDLE_VALUE;
    mapping_handle_ = NULL;
#else
    file_descriptor_ = -1;
#endif

    // Encryption keys will be dynamically injected by Python virtualizer
    // These placeholder values will be overwritten during virtualization process
    op_encryption_key_ = {};
    param_encryption_key_ = {};
    graph_encryption_key_ = {};
    shape_encryption_key_ = {};
    op_nonce_base_ = {};
    param_nonce_base_ = {};
    graph_nonce_base_ = {};
    shape_nonce_base_ = {};

    // XOR obfuscation parameters (dynamically injected by Python)
    // Master mask and domain offsets for key deobfuscation
    xor_master_mask_ = {};
    xor_op_offset_ = 0x00;
    xor_param_offset_ = 0x00;
    xor_graph_offset_ = 0x00;
    xor_shape_offset_ = 0x00;

    input_ops_ = {};
    output_ops_ = {};

    // Modular arithmetic parameters (injected by inject_keys_to_backend.py)
    // These values enable restoration of real values from virtualized values
    op_modulus_ = 251;       // Covers TFLite operator types 0-189
    conn_modulus_ = 50021;   // Covers models with <5000 layers (default)

    _load_virtualized_data();
    if (!ok_) return;

    _load_model_metadata();
    if (!ok_) return;

    de_virtualize_op_types();
    if (!ok_) return;

    de_virtualize_params();
    if (!ok_) return;

    // NOTE: params_array_ or mmap memory is intentionally NOT released here (zero-copy optimization)
    // DevirtualizedParam now stores pointers (data_ptr) directly into params_array_ or mmap region
    // Memory will be released later after FlatBuffer construction completes
    // This avoids duplicate memory allocation but requires memory to stay alive longer

    de_virtualize_graph();
}

// Destructor: clean up mmap resources if used
VirtualizedModelParser::~VirtualizedModelParser() {
    if (use_mmap_) {
        _munmap_params_file();
    }
}

void VirtualizedModelParser::_load_virtualized_data() {
    std::ifstream json_file(v_infos_path_);
    if (!json_file.is_open()) {
        Fail("Failed to open " + v_infos_path_);
        return;
    }
    if (!(json_file >> v_infos_)) {
        Fail("Failed to parse JSON from " + v_infos_path_);
        return;
    }

    // Load parameter data using mmap (zero-copy) or traditional read
    if (use_mmap_) {
        // Memory-mapped file access (zero-copy optimization)
        // File is mapped into virtual address space, avoiding read() system call and buffer copying
        _mmap_params_file();
        if (!ok_) return;  // _mmap_params_file() sets ok_ to false on failure
    } else {
        // Traditional file read (legacy mode, retained for compatibility)
        std::ifstream params_file(params_path_, std::ios::binary);
        if (!params_file.is_open()) {
            Fail("Failed to open " + params_path_);
            return;
        }

        params_file.seekg(0, std::ios::end);
        const std::streampos file_size = params_file.tellg();
        if (file_size < 0) {
            Fail("Failed to determine size for " + params_path_);
            return;
        }
        params_file.seekg(0, std::ios::beg);

        const size_t size_bytes = static_cast<size_t>(file_size);
        if (size_bytes % sizeof(float) != 0) {
            Fail("Params file size must be aligned to 4-byte entries for " + params_path_);
            return;
        }

        params_array_.resize(size_bytes);
        params_file.read(reinterpret_cast<char*>(params_array_.data()),
                                        static_cast<std::streamsize>(size_bytes));
        if (!params_file) {
            Fail("Failed to read parameter data from " + params_path_);
            params_array_.clear();
        }
    }
}

void VirtualizedModelParser::_load_model_metadata() {
    // Load and decrypt model metadata from v_infos.json
    // Metadata includes: input_shapes, input_dtypes, input_shape_signatures, input_ops, output_ops

    // Check if encrypted_metadata field exists
    if (!v_infos_.contains("encrypted_metadata")) {
        Fail("Missing 'encrypted_metadata' field in v_infos.json");
        return;
    }

    // Get Base64-encoded encrypted metadata
    std::string encrypted_metadata_b64 = v_infos_["encrypted_metadata"];

    // Base64 decode
    std::vector<uint8_t> ciphertext = base64_decode(encrypted_metadata_b64);
    if (ciphertext.empty()) {
        Fail("Failed to Base64 decode encrypted_metadata");
        return;
    }

    // Construct nonce: shape_nonce_base (8 bytes) + 0x00*8 (8 bytes) = 16 bytes
    std::array<uint8_t, 16> nonce;
    std::copy(shape_nonce_base_.begin(), shape_nonce_base_.end(), nonce.begin());
    std::fill(nonce.begin() + 8, nonce.end(), 0x00);

    // Deobfuscate shape encryption key to get plaintext key
    // This ensures decryption key matches the plaintext key used in Python encryption
    auto plaintext_shape_key = deobfuscate_key(shape_encryption_key_, xor_shape_offset_);

    // Decrypt using plaintext key
    std::vector<uint8_t> plaintext_bytes = aes_ctr_decrypt(ciphertext, plaintext_shape_key, nonce);
    if (plaintext_bytes.empty()) {
        Fail("Failed to decrypt metadata");
        return;
    }

    // Parse JSON from decrypted plaintext using stream operator (no exceptions)
    std::string plaintext_str(plaintext_bytes.begin(), plaintext_bytes.end());
    std::istringstream metadata_stream(plaintext_str);
    nlohmann::json metadata;
    metadata_stream >> metadata;
    if (metadata_stream.fail() || metadata.is_null()) {
        Fail("Failed to parse metadata JSON");
        return;
    }

    // Extract input_shapes
    if (metadata.contains("input_shapes") && metadata["input_shapes"].is_array()) {
        for (const auto& shape_arr : metadata["input_shapes"]) {
            std::vector<int> shape;
            for (const auto& dim : shape_arr) {
                shape.push_back(dim.get<int>());
            }
            input_shapes_.push_back(shape);
        }
    }

    // Extract input_dtypes
    if (metadata.contains("input_dtypes") && metadata["input_dtypes"].is_array()) {
        for (const auto& dtype : metadata["input_dtypes"]) {
            input_dtypes_.push_back(dtype.get<std::string>());
        }
    }

    // Extract input_shape_signatures
    if (metadata.contains("input_shape_signatures") && metadata["input_shape_signatures"].is_array()) {
        for (const auto& sig_arr : metadata["input_shape_signatures"]) {
            std::vector<int> signature;
            for (const auto& dim : sig_arr) {
                signature.push_back(dim.get<int>());
            }
            input_shape_signatures_.push_back(signature);
        }
    }

    // Extract input_ops and output_ops (virtualized values, need modular arithmetic restoration)
    if (metadata.contains("input_ops") && metadata["input_ops"].is_array()) {
        input_ops_.clear();
        for (const auto& v_input_op : metadata["input_ops"]) {
            uint32_t virtualized_input_op = v_input_op.get<uint32_t>();
            // Apply modular arithmetic restoration: real_value = virtualized_value % conn_modulus
            uint32_t real_input_op = virtualized_input_op % conn_modulus_;
            input_ops_.push_back(static_cast<int>(real_input_op));
        }
    }

    if (metadata.contains("output_ops") && metadata["output_ops"].is_array()) {
        output_ops_.clear();
        for (const auto& v_output_op : metadata["output_ops"]) {
            uint32_t virtualized_output_op = v_output_op.get<uint32_t>();
            // Apply modular arithmetic restoration: real_value = virtualized_value % conn_modulus
            uint32_t real_output_op = virtualized_output_op % conn_modulus_;
            output_ops_.push_back(static_cast<int>(real_output_op));
        }
    }
}

std::vector<DevirtualizedOp> VirtualizedModelParser::de_virtualize_op_types() {
    if (!ok_) return {};

    // Extract operators array from v_infos_ object
    if (!v_infos_.contains("operators") || !v_infos_["operators"].is_array()) {
        Fail("Missing or invalid 'operators' field in v_infos.json");
        return {};
    }

    const auto& operators_array = v_infos_["operators"];
    std::vector<DevirtualizedOp> result;
    result.reserve(operators_array.size());

    for (size_t idx = 0; idx < operators_array.size(); ++idx) {
        const auto& v_op = operators_array[idx];

        // Decrypt operator type code from Base64-encoded ciphertext
        std::string v_op_code_data = v_op["v_op_code_data"].get<std::string>();
        std::vector<uint8_t> ciphertext = base64_decode(v_op_code_data);

        // Construct 16-byte nonce: 8-byte base + 8-byte counter (using semantic op index)
        std::array<uint8_t, 16> nonce;
        std::memcpy(nonce.data(), op_nonce_base_.data(), 8);
        int op_index = v_op["index"].get<int>();
        uint64_t counter = static_cast<uint64_t>(op_index);
        std::memcpy(nonce.data() + 8, &counter, 8);

        // Deobfuscate the encryption key to get plaintext key (matches Python encryption key)
        std::array<uint8_t, 32> plaintext_op_key = deobfuscate_key(op_encryption_key_, xor_op_offset_);

        // Decrypt using AES-256-CTR with plaintext key
        std::vector<uint8_t> plaintext = aes_ctr_decrypt(ciphertext, plaintext_op_key, nonce);
        if (plaintext.size() < 4) {
            Fail("Failed to decrypt operator type code for operator " + std::to_string(idx));
            return {};
        }

        // Extract 4-byte unsigned integer (little-endian) - this is the virtualized code
        uint32_t v_code;
        std::memcpy(&v_code, plaintext.data(), 4);

        // Apply modular arithmetic restoration: real_code = v_code % op_modulus
        uint32_t deprecated_builtin_code = v_code % op_modulus_;

        DevirtualizedOp op;
        op.index = op_index;  // Reuse the op_index from nonce construction above
        op.op_type = get_op_type_from_deprecated_builtin_code(static_cast<int>(deprecated_builtin_code));

        // Prefer decoding keyless v_builtin_options
        auto decode_activation = [](int code) -> std::string {
            switch (code) {
                case 1: return "RELU";
                case 2: return "RELU6";
                default: return "NONE";
            }
        };
        auto decode_padding = [](int code) -> std::string {
            switch (code) {
                case 1: return "VALID";
                default: return "SAME";
            }
        };
        auto decode_dtype = [](int code) -> std::string {
            switch (code) {
                case 1: return "INT32";
                case 2: return "BOOL";
                default: return "FLOAT32";
            }
        };

        nlohmann::json options = nlohmann::json::object();
        if (v_op.contains("v_builtin_options") && v_op["v_builtin_options"].is_string()) {
            // Decrypt builtin_options block (Base64 → AES-CTR) into int32 slots
            std::string vb64 = v_op["v_builtin_options"].get<std::string>();
            std::vector<uint8_t> ciphertext = base64_decode(vb64);

            // Construct nonce for PARAM domain: 8-byte base + 8-byte counter
            std::array<uint8_t, 16> nonce;
            std::memcpy(nonce.data(), param_nonce_base_.data(), 8);
            uint64_t param_counter = (static_cast<uint64_t>(op.index) << 32) | 0x00000002ULL;
            std::memcpy(nonce.data() + 8, &param_counter, 8);

            // Deobfuscate param key and decrypt
            std::array<uint8_t, 32> plaintext_param_key = deobfuscate_key(param_encryption_key_, xor_param_offset_);
            std::vector<uint8_t> pt = aes_ctr_decrypt(ciphertext, plaintext_param_key, nonce);

            std::vector<int32_t> slots;
            if (!pt.empty() && (pt.size() % 4 == 0)) {
                const size_t cnt = pt.size() / 4;
                slots.resize(cnt);
                for (size_t i = 0; i < cnt; ++i) {
                    int32_t v;
                    std::memcpy(&v, pt.data() + i * 4, 4);
                    slots[i] = v;
                }
            }

            // Helpers to fetch typed values from slots
            auto at_i = [&](size_t i, int def) -> int {
                return (i < slots.size()) ? static_cast<int>(slots[i]) : def;
            };
            auto at_b = [&](size_t i, bool def) -> bool {
                return (i < slots.size()) ? (slots[i] != 0) : def;
            };
            auto at_f_q = [&](size_t i, float def) -> float {
                return (i < slots.size()) ? (static_cast<float>(slots[i]) / 1000.0f) : def;
            };

            const std::string t = op.op_type;
            if (t == "CONV_2D") {
                options["padding"] = decode_padding(at_i(0, 0));
                options["stride_w"] = at_i(1, 1);
                options["stride_h"] = at_i(2, 1);
                options["fused_activation_function"] = decode_activation(at_i(3, 0));
            } else if (t == "DEPTHWISE_CONV_2D") {
                options["padding"] = decode_padding(at_i(0, 0));
                options["stride_w"] = at_i(1, 1);
                options["stride_h"] = at_i(2, 1);
                options["depth_multiplier"] = at_i(3, 1);
                options["fused_activation_function"] = decode_activation(at_i(4, 0));
            } else if (t == "MAX_POOL_2D" || t == "AVERAGE_POOL_2D") {
                options["padding"] = decode_padding(at_i(0, 0));
                options["stride_w"] = at_i(1, 2);
                options["stride_h"] = at_i(2, 2);
                options["filter_width"] = at_i(3, 2);
                options["filter_height"] = at_i(4, 2);
                options["fused_activation_function"] = decode_activation(at_i(5, 0));
            } else if (t == "CONCATENATION") {
                options["axis"] = at_i(0, 3);
                options["fused_activation_function"] = decode_activation(at_i(1, 0));
            } else if (t == "FULLY_CONNECTED") {
                options["fused_activation_function"] = decode_activation(at_i(0, 0));
                options["keep_num_dims"] = at_b(1, false);
                options["asymmetric_quantize_inputs"] = at_b(2, false);
            } else if (t == "SOFTMAX") {
                options["beta"] = at_f_q(0, 1.0f);
            } else if (t == "ADD" || t == "MUL" || t == "SUB") {
                options["fused_activation_function"] = decode_activation(at_i(0, 0));
            } else if (t == "RESHAPE") {
                int n = at_i(0, 0);
                if (n > 0) {
                    std::vector<int> shp;
                    shp.reserve(static_cast<size_t>(n));
                    for (int k = 0; k < n; ++k) {
                        shp.push_back(at_i(static_cast<size_t>(1 + k), 1));
                    }
                    options["new_shape"] = shp;
                }
            } else if (t == "RESIZE_BILINEAR") {
                options["align_corners"] = at_b(0, false);
                options["half_pixel_centers"] = at_b(1, false);
            } else if (t == "MEAN") {
                options["keep_dims"] = at_b(0, false);
            } else if (t == "GELU") {
                options["approximate"] = at_b(0, true);
            } else if (t == "BATCH_MATMUL") {
                options["adj_x"] = at_b(0, false);
                options["adj_y"] = at_b(1, false);
                options["asymmetric_quantize_inputs"] = at_b(2, false);
            } else if (t == "GATHER") {
                options["axis"] = at_i(0, 0);
                options["batch_dims"] = at_i(1, 0);
            } else if (t == "SPLIT") {
                options["num_splits"] = at_i(0, 3);
            } else if (t == "SQUEEZE") {
                int n = at_i(0, 0);
                if (n > 0) {
                    std::vector<int> dims;
                    dims.reserve(static_cast<size_t>(n));
                    for (int k = 0; k < n; ++k) {
                        dims.push_back(at_i(static_cast<size_t>(1 + k), 0));
                    }
                    options["squeeze_dims"] = dims;
                }
            } else if (t == "STRIDED_SLICE") {
                options["begin_mask"] = at_i(0, 0);
                options["end_mask"] = at_i(1, 0);
                options["ellipsis_mask"] = at_i(2, 0);
                options["new_axis_mask"] = at_i(3, 0);
                options["shrink_axis_mask"] = at_i(4, 0);
            } else if (t == "PACK") {
                options["values_count"] = at_i(0, 1);
                options["axis"] = at_i(1, 0);
            } else if (t == "RANGE" || t == "EXPAND_DIMS" || t == "TRANSPOSE") {
                // no-op
            } else if (t == "SHAPE") {
                options["out_type"] = decode_dtype(at_i(0, 1));
            } else if (t == "CAST") {
                options["in_data_type"] = decode_dtype(at_i(0, 0));
                options["out_data_type"] = decode_dtype(at_i(1, 0));
            }
        }

        op.builtin_options = options;
        op.mutating_variable_inputs =
                v_op.value("mutating_variable_inputs", std::vector<int>());
        // Note: v_forward_connections in new format is Base64 string array, not int array
        // It will be decrypted in de_virtualize_graph(), so we leave this field empty here
        op.v_forward_connections = {};

        result.push_back(std::move(op));
    }

    devirtualized_ops_ = result;
    return result;
}

std::vector<DevirtualizedParam> VirtualizedModelParser::de_virtualize_params() {
    if (!ok_) return {};

    // Extract operators array from v_infos_ object
    if (!v_infos_.contains("operators") || !v_infos_["operators"].is_array()) {
        Fail("Missing or invalid 'operators' field in v_infos.json");
        return {};
    }

    const auto& operators_array = v_infos_["operators"];
    std::vector<DevirtualizedParam> result;

    for (const auto& v_op : operators_array) {
        if (!v_op.contains("v_position_data") || v_op["v_position_data"].empty()) {
            continue;
        }

        std::vector<DevirtualizedParam> params;
        const auto& v_position_data_list = v_op["v_position_data"];
        const auto& v_shape_list = v_op["v_shape"];
        const int op_index = v_op["index"].get<int>();

        for (size_t i = 0; i < v_position_data_list.size(); ++i) {
            // 1) Decrypt shape block first to obtain dtype and param_slot for this parameter
            std::string v_shape_data = v_shape_list[i].get<std::string>();
            std::vector<uint8_t> shape_ciphertext = base64_decode(v_shape_data);

            // Construct 16-byte nonce for shape: 8-byte base + 8-byte combined counter
            // Combined counter = (op_index << 32) | shape_index
            std::array<uint8_t, 16> shape_nonce;
            std::memcpy(shape_nonce.data(), shape_nonce_base_.data(), 8);
            uint64_t shape_combined_counter = (static_cast<uint64_t>(op_index) << 32) | static_cast<uint64_t>(i);
            std::memcpy(shape_nonce.data() + 8, &shape_combined_counter, 8);

            // Deobfuscate the encryption key to get plaintext key (matches Python encryption key)
            std::array<uint8_t, 32> plaintext_shape_key = deobfuscate_key(shape_encryption_key_, xor_shape_offset_);

            // Decrypt shape using AES-256-CTR with plaintext key
            std::vector<uint8_t> shape_plaintext = aes_ctr_decrypt(shape_ciphertext, plaintext_shape_key, shape_nonce);
            if (shape_plaintext.size() < 12) {
                Fail("Failed to decrypt shape for op " + std::to_string(op_index) +
                    " param " + std::to_string(i));
                return {};
            }
            // Extract extended shape header and dims:
            // [num_dims(uint32)][dtype_code(uint32)][param_slot(int32)][dims(int32) × L]
            uint32_t num_dims;
            uint32_t dtype_code;
            int32_t param_slot;
            std::memcpy(&num_dims, shape_plaintext.data(), 4);
            std::memcpy(&dtype_code, shape_plaintext.data() + 4, 4);
            std::memcpy(&param_slot, shape_plaintext.data() + 8, 4);

            // Phase 3: Discard dummy entries early (before decrypting position block).
            // Dummy is defined strictly as: num_dims == 0 AND param_slot < 0.
            // Real scalar constants have num_dims == 0 but param_slot >= 0 and must be kept.
            if (num_dims == 0 && param_slot < 0) {
                continue;  // Skip this parameter entirely
            }

            if (shape_plaintext.size() < 12 + num_dims * 4) {
                Fail("Shape plaintext size mismatch for op " + std::to_string(op_index) +
                    " param " + std::to_string(i));
                return {};
            }

            std::vector<int> shape;
            shape.reserve(num_dims);
            for (uint32_t d = 0; d < num_dims; ++d) {
                int32_t dim_value;
                std::memcpy(&dim_value, shape_plaintext.data() + 12 + d * 4, 4);
                shape.push_back(static_cast<int>(dim_value));
            }

            // Map dtype_code to dtype string (lowercase to match builder expectations)
            std::string dtype;
            switch (dtype_code) {
                case 2: dtype = "int32"; break;
                case 3: dtype = "bool"; break;
                default: dtype = "float32"; break;
            }

            // 2) Decrypt parameter position block using param_slot from shape header
            std::string v_position_data = v_position_data_list[i].get<std::string>();
            std::vector<uint8_t> ciphertext = base64_decode(v_position_data);

            std::array<uint8_t, 16> nonce;
            std::memcpy(nonce.data(), param_nonce_base_.data(), 8);
            int input_slot = (param_slot >= 0) ? param_slot : 0;
            uint64_t combined_counter = (static_cast<uint64_t>(op_index) << 32) |
                                        static_cast<uint64_t>(input_slot);
            std::memcpy(nonce.data() + 8, &combined_counter, 8);

            std::array<uint8_t, 32> plaintext_param_key = deobfuscate_key(param_encryption_key_, xor_param_offset_);
            std::vector<uint8_t> plaintext = aes_ctr_decrypt(ciphertext, plaintext_param_key, nonce);
            if (plaintext.size() < 16) {
                Fail("Failed to decrypt parameter position for op " + std::to_string(op_index) +
                    " param " + std::to_string(i));
                return {};
            }

            uint64_t real_start_pos, real_length;
            std::memcpy(&real_start_pos, plaintext.data(), 8);
            std::memcpy(&real_length, plaintext.data() + 8, 8);

            if (real_start_pos < 0 || real_length < 0) {
                Fail("Param positions must be non-negative");
                return {};
            }

            const size_t total_bytes = use_mmap_ ? mmap_size_ : params_array_.size();
            const size_t element_capacity = total_bytes / sizeof(float);
            if (static_cast<size_t>(real_start_pos + real_length) > element_capacity) {
                Fail("Param size mismatch: index out of bounds");
                return {};
            }
            const size_t byte_offset =
                    static_cast<size_t>(real_start_pos) * sizeof(float);
            const size_t byte_length =
                    static_cast<size_t>(real_length) * sizeof(float);

            DevirtualizedParam param;
            param.index = static_cast<int>(i);  // Use local parameter index within the operator
            param.op_index = op_index;
            param.start_pos = static_cast<long long>(real_start_pos);
            param.length = static_cast<long long>(real_length);
            param.shape = shape;
            param.dtype = dtype;
            param.param_slot = static_cast<int>(param_slot);

            // Zero-copy optimization: store pointer to parameter data
            // When mmap is enabled, point to mmap region; otherwise point to params_array_
            // This avoids duplicating parameter data and reduces peak memory usage significantly
            if (use_mmap_) {
                // Point to memory-mapped file region (true zero-copy, no read() syscall)
                param.data_ptr = static_cast<const uint8_t*>(mmap_addr_) + byte_offset;
            } else {
                // Point to params_array_ (avoids duplicate vector copy but still has read() overhead)
                param.data_ptr = params_array_.data() + byte_offset;
            }
            param.data_size_bytes = byte_length;

            // Legacy fields remain empty (for backward compatibility with code that checks .empty())
            // New code should use data_ptr and data_size_bytes instead

            // Validate parameter size matches expected shape
            const size_t data_size = static_cast<size_t>(real_length);
            if (!shape.empty()) {
                size_t expected_size = 1;
                for (int dim : shape) {
                    expected_size *= static_cast<size_t>(dim);
                }
                if (data_size != expected_size) {
                    Fail("Param size mismatch: expected " + std::to_string(expected_size) +
                            ", actual " + std::to_string(data_size));
                    return {};
                }
            }

            params.push_back(std::move(param));
        }

        // Reorder parameters for specific operator types (op_index already defined above)
        auto op_it = std::find_if(devirtualized_ops_.begin(), devirtualized_ops_.end(),
            [op_index](const DevirtualizedOp& op) { return op.index == op_index; });

        if (op_it != devirtualized_ops_.end()) {
            const std::string& op_type = op_it->op_type;
            if (op_type == "CONV_2D" || op_type == "DEPTHWISE_CONV_2D" ||
                op_type == "FULLY_CONNECTED") {
                // Use universal sorting strategy: higher dimension first (weights before biases)
                std::sort(params.begin(), params.end(),
                    [](const DevirtualizedParam& a, const DevirtualizedParam& b) {
                        return a.shape.size() > b.shape.size();
                    });
            }
        }

        for (auto& param : params) {
            result.push_back(std::move(param));
        }
    }

    devirtualized_params_ = result;
    return result;
}

std::vector<DevirtualizedGraph> VirtualizedModelParser::de_virtualize_graph() {
    if (!ok_) return {};

    // Extract operators array from v_infos_ object
    if (!v_infos_.contains("operators") || !v_infos_["operators"].is_array()) {
        Fail("Missing or invalid 'operators' field in v_infos.json");
        return {};
    }

    const auto& operators_array = v_infos_["operators"];
    std::vector<DevirtualizedGraph> result;
    result.reserve(operators_array.size());

    for (size_t op_idx = 0; op_idx < operators_array.size(); ++op_idx) {
        const auto& v_op = operators_array[op_idx];

        if (!v_op.contains("v_forward_connections")) {
            continue;
        }

        DevirtualizedGraph node;
        node.index = v_op["index"].get<int>();

        std::vector<int> forward_connections;
        const auto& v_forward_connections_list = v_op["v_forward_connections"];

        for (size_t conn_idx = 0; conn_idx < v_forward_connections_list.size(); ++conn_idx) {
            // Decrypt connection ID from Base64-encoded ciphertext
            std::string v_conn_data = v_forward_connections_list[conn_idx].get<std::string>();
            std::vector<uint8_t> ciphertext = base64_decode(v_conn_data);

            // Construct 16-byte nonce: 8-byte base + 8-byte combined counter (using semantic op index)
            // Combine op index and conn_idx into single 64-bit counter (matches Python: (op['index'] << 32) | conn_idx)
            std::array<uint8_t, 16> nonce;
            std::memcpy(nonce.data(), graph_nonce_base_.data(), 8);
            uint64_t combined_counter = (static_cast<uint64_t>(node.index) << 32) | static_cast<uint64_t>(conn_idx);
            std::memcpy(nonce.data() + 8, &combined_counter, 8);

            // Deobfuscate the encryption key to get plaintext key (matches Python encryption key)
            std::array<uint8_t, 32> plaintext_graph_key = deobfuscate_key(graph_encryption_key_, xor_graph_offset_);

            // Decrypt using AES-256-CTR with plaintext key
            std::vector<uint8_t> plaintext = aes_ctr_decrypt(ciphertext, plaintext_graph_key, nonce);
            if (plaintext.size() < 4) {
                Fail("Failed to decrypt graph connection for op " + std::to_string(node.index) +
                    " connection " + std::to_string(conn_idx));
                return {};
            }

            // Extract 4-byte unsigned integer (little-endian) - this is the virtualized conn_id
            uint32_t v_conn_id;
            std::memcpy(&v_conn_id, plaintext.data(), 4);

            // Apply modular arithmetic restoration: real_conn_id = v_conn_id % conn_modulus
            uint32_t real_conn_id = v_conn_id % conn_modulus_;

            forward_connections.push_back(static_cast<int>(real_conn_id));
        }

        node.forward_connections = std::move(forward_connections);
        node.forward_branches =
                v_op.value("forward_branches", std::vector<int>());

        result.push_back(std::move(node));
    }

    devirtualized_graph_ = result;
    return result;
}

std::pair<std::vector<int>, std::vector<int>>
VirtualizedModelParser::identify_input_output_ops() {
    if (!ok_) return {};

    if (input_ops_.empty() || output_ops_.empty()) {
        _infer_input_output_ops();
    }
    return {input_ops_, output_ops_};
}

void VirtualizedModelParser::_infer_input_output_ops() {
    if (!ok_) return;

    if (devirtualized_graph_.empty()) {
        de_virtualize_graph();
        if (!ok_) return;
    }

    std::set<int> dependencies;
    for (const auto& node : devirtualized_graph_) {
        dependencies.insert(node.forward_connections.begin(),
                                                node.forward_connections.end());
    }

    input_ops_.clear();
    for (const auto& node : devirtualized_graph_) {
        if (!dependencies.count(node.index)) {
            input_ops_.push_back(node.index);
        }
    }

    output_ops_.clear();
    for (const auto& node : devirtualized_graph_) {
        if (node.forward_connections.empty()) {
            output_ops_.push_back(node.index);
        }
    }
}

std::vector<DevirtualizedParam> VirtualizedModelParser::get_op_params(int op_index) {
    if (!ok_) return {};

    if (devirtualized_params_.empty()) {
        de_virtualize_params();
        if (!ok_) return {};
    }

    std::vector<DevirtualizedParam> params;
    for (const auto& param : devirtualized_params_) {
        if (param.op_index == op_index) {
            params.push_back(param);
        }
    }
    return params;
}

std::vector<const DevirtualizedParam*> VirtualizedModelParser::get_op_params_ptrs(int op_index) const {
    if (!ok_) return {};

    std::vector<const DevirtualizedParam*> params_ptrs;
    for (const auto& param : devirtualized_params_) {
        if (param.op_index == op_index) {
            params_ptrs.push_back(&param);
        }
    }
    return params_ptrs;
}

DevirtualizedOp VirtualizedModelParser::get_op_info(int op_index) {
    if (!ok_) return DevirtualizedOp();

    if (devirtualized_ops_.empty()) {
        de_virtualize_op_types();
        if (!ok_) return DevirtualizedOp();
    }

    for (const auto& op : devirtualized_ops_) {
        if (op.index == op_index) {
            return op;
        }
    }

    Fail("Cannot find operator index: " + std::to_string(op_index));
    return DevirtualizedOp();
}

std::vector<int> VirtualizedModelParser::get_op_forward_connection(int op_index) {
    if (!ok_) return {};

    if (devirtualized_graph_.empty()) {
        de_virtualize_graph();
        if (!ok_) return {};
    }

    for (const auto& node : devirtualized_graph_) {
        if (node.index == op_index) {
            return node.forward_connections;
        }
    }

    Fail("Cannot find operator index in devirtualized_graph: " +
            std::to_string(op_index));
    return {};
}

std::vector<int> VirtualizedModelParser::get_op_forward_branch(int op_index) {
    if (!ok_) return {};

    if (devirtualized_graph_.empty()) {
        de_virtualize_graph();
        if (!ok_) return {};
    }

    for (const auto& node : devirtualized_graph_) {
        if (node.index == op_index) {
            return node.forward_branches;
        }
    }

    Fail("Cannot find operator index in devirtualized_graph: " +
            std::to_string(op_index));
    return {};
}

std::vector<int> VirtualizedModelParser::get_execution_order() {
    if (!ok_) return {};

    if (devirtualized_graph_.empty()) {
        de_virtualize_graph();
        if (!ok_) return {};
    }

    std::map<int, std::vector<int>> adjacency;
    std::map<int, int> in_degree;

    for (const auto& node : devirtualized_graph_) {
        adjacency[node.index] = {};
        in_degree[node.index] = 0;
    }

    for (const auto& node : devirtualized_graph_) {
        for (int next : node.forward_connections) {
            adjacency[node.index].push_back(next);
            ++in_degree[next];
        }
    }

    std::queue<int> zero_in_degree;
    for (const auto& pair : in_degree) {
        if (pair.second == 0) {
            zero_in_degree.push(pair.first);
        }
    }

    std::vector<int> order;
    while (!zero_in_degree.empty()) {
        const int current = zero_in_degree.front();
        zero_in_degree.pop();
        order.push_back(current);

        for (int next : adjacency[current]) {
            const int remaining = --in_degree[next];
            if (remaining == 0) {
                zero_in_degree.push(next);
            }
        }
    }

    std::reverse(order.begin(), order.end());
    return order;
}

bool VirtualizedModelParser::validate_model_integrity() {
    if (!ok_) return false;

    const auto ops = de_virtualize_op_types();
    if (!ok_) return false;
    const auto params = de_virtualize_params();
    if (!ok_) return false;
    const auto graph = de_virtualize_graph();
    if (!ok_) return false;
    const auto io_pair = identify_input_output_ops();
    if (!ok_) return false;
    const auto& input_ops = io_pair.first;
    const auto& output_ops = io_pair.second;

    if (ops.size() != graph.size()) {
        return false;
    }

    std::set<int> param_ops_set;
    for (const auto& param : params) {
        param_ops_set.insert(param.op_index);
    }

    std::set<int> mandatory_ops;
    for (const auto& op : ops) {
        if (MANDATORY_PARAM_OPS.count(op.op_type)) {
            mandatory_ops.insert(op.index);
        }
    }

    std::set<int> optional_with_params;
    const auto& operators_array = v_infos_["operators"];
    for (const auto& v : operators_array) {
        if (v.contains("v_position_data") && !v["v_position_data"].empty()) {
            optional_with_params.insert(v["index"].get<int>());
        }
    }

    std::set<int> expected_param_ops = mandatory_ops;
    expected_param_ops.insert(optional_with_params.begin(), optional_with_params.end());

    std::set<int> missing_mandatory;
    std::set_difference(mandatory_ops.begin(), mandatory_ops.end(),
                                            param_ops_set.begin(), param_ops_set.end(),
                                            std::inserter(missing_mandatory, missing_mandatory.begin()));

    std::set<int> extra_ops;
    std::set_difference(param_ops_set.begin(), param_ops_set.end(),
                                            expected_param_ops.begin(), expected_param_ops.end(),
                                            std::inserter(extra_ops, extra_ops.begin()));

    if (!missing_mandatory.empty() || !extra_ops.empty()) {
        return false;
    }

    if (input_ops.empty() || output_ops.empty()) {
        return false;
    }

    return true;
}

void VirtualizedModelParser::export_model_summary(const std::string& output_path) {
    if (!ok_) return;

    if (devirtualized_ops_.empty()) {
        de_virtualize_op_types();
        if (!ok_) return;
    }
    if (devirtualized_params_.empty()) {
        de_virtualize_params();
        if (!ok_) return;
    }
    if (devirtualized_graph_.empty()) {
        de_virtualize_graph();
        if (!ok_) return;
    }

    const auto io_pair = identify_input_output_ops();
    if (!ok_) return;
    const auto& input_ops = io_pair.first;
    const auto& output_ops = io_pair.second;

    nlohmann::json summary;
    summary["model_info"]["total_ops"] = devirtualized_ops_.size();
    summary["model_info"]["total_params"] = devirtualized_params_.size();
    summary["model_info"]["total_nodes"] = devirtualized_graph_.size();
    summary["model_info"]["input_ops"] = input_ops;
    summary["model_info"]["output_ops"] = output_ops;
    summary["model_info"]["execution_order"] = get_execution_order();

    for (const auto& op : devirtualized_ops_) {
        const auto op_params = get_op_params(op.index);
        nlohmann::json op_summary;
        op_summary["index"] = op.index;
        op_summary["type"] = op.op_type;
        op_summary["param_count"] = op_params.size();
        op_summary["builtin_options"] = op.builtin_options;

        // Get forward connections from devirtualized graph (already decrypted)
        std::vector<int> forward_connections;
        for (const auto& node : devirtualized_graph_) {
            if (node.index == op.index) {
                forward_connections = node.forward_connections;
                break;
            }
        }
        op_summary["forward_connections"] = forward_connections;
        op_summary["forward_branches"] = get_op_forward_branch(op.index);
        op_summary["mutating_variable_inputs"] = op.mutating_variable_inputs;

        summary["ops_summary"].push_back(op_summary);
    }

    for (const auto& param : devirtualized_params_) {
        nlohmann::json param_summary;
        param_summary["index"] = param.index;
        param_summary["op_index"] = param.op_index;
        param_summary["shape"] = param.shape;
        param_summary["dtype"] = param.dtype;
        param_summary["start_pos"] = param.start_pos;
        param_summary["length"] = param.length;

        summary["params_summary"].push_back(param_summary);
    }

    std::ofstream file(output_path);
    file << summary.dump(2);
}

// Memory mapping implementation (platform-specific)
// Maps parameter file into virtual address space for zero-copy access
void VirtualizedModelParser::_mmap_params_file() {
#ifdef _WIN32
    // Windows implementation using CreateFile + CreateFileMapping + MapViewOfFile

    // Open file with read access
    file_handle_ = CreateFileA(
        params_path_.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        Fail("Failed to open parameter file for mmap: " + params_path_);
        return;
    }

    // Get file size
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file_handle_, &file_size)) {
        Fail("Failed to get file size for mmap: " + params_path_);
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return;
    }

    mmap_size_ = static_cast<size_t>(file_size.QuadPart);

    // Validate file size alignment
    if (mmap_size_ % sizeof(float) != 0) {
        Fail("Params file size must be aligned to 4-byte entries for " + params_path_);
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return;
    }

    // Create file mapping object
    mapping_handle_ = CreateFileMappingA(
        file_handle_,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );

    if (mapping_handle_ == NULL) {
        Fail("Failed to create file mapping for: " + params_path_);
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return;
    }

    // Map view of file into address space
    mmap_addr_ = MapViewOfFile(
        mapping_handle_,
        FILE_MAP_READ,
        0,
        0,
        0
    );

    if (mmap_addr_ == NULL) {
        Fail("Failed to map view of file: " + params_path_);
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        mapping_handle_ = NULL;
        file_handle_ = INVALID_HANDLE_VALUE;
        return;
    }

#else
    // POSIX implementation (Linux/macOS) using open + fstat + mmap

    // Open file with read-only access
    file_descriptor_ = open(params_path_.c_str(), O_RDONLY);
    if (file_descriptor_ == -1) {
        Fail("Failed to open parameter file for mmap: " + params_path_);
        return;
    }

    // Get file size using fstat
    struct stat file_stat;
    if (fstat(file_descriptor_, &file_stat) == -1) {
        Fail("Failed to get file size for mmap: " + params_path_);
        close(file_descriptor_);
        file_descriptor_ = -1;
        return;
    }

    mmap_size_ = static_cast<size_t>(file_stat.st_size);

    // Validate file size alignment
    if (mmap_size_ % sizeof(float) != 0) {
        Fail("Params file size must be aligned to 4-byte entries for " + params_path_);
        close(file_descriptor_);
        file_descriptor_ = -1;
        return;
    }

    // Map file into memory (read-only, private mapping)
    // PROT_READ: Pages may be read
    // MAP_PRIVATE: Changes are not written back to file (copy-on-write)
    mmap_addr_ = mmap(
        nullptr,              // Let kernel choose address
        mmap_size_,           // Length of mapping
        PROT_READ,            // Memory protection flags
        MAP_PRIVATE,          // Mapping flags
        file_descriptor_,     // File descriptor
        0                     // Offset in file
    );

    if (mmap_addr_ == MAP_FAILED) {
        Fail("Failed to mmap parameter file: " + params_path_);
        close(file_descriptor_);
        file_descriptor_ = -1;
        mmap_addr_ = nullptr;
        return;
    }

    // Advise kernel about memory access pattern (sequential read)
    // This allows kernel to optimize page-in behavior
    madvise(mmap_addr_, mmap_size_, MADV_SEQUENTIAL);
#endif
}

// Unmap parameter file and release resources
void VirtualizedModelParser::_munmap_params_file() {
    if (mmap_addr_ == nullptr) {
        return;  // Nothing to unmap
    }

#ifdef _WIN32
    // Windows cleanup: UnmapViewOfFile + CloseHandle (mapping + file)
    if (mmap_addr_ != NULL) {
        UnmapViewOfFile(mmap_addr_);
        mmap_addr_ = nullptr;
    }
    if (mapping_handle_ != NULL) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = NULL;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    // POSIX cleanup: munmap + close
    if (mmap_addr_ != nullptr && mmap_addr_ != MAP_FAILED) {
        munmap(mmap_addr_, mmap_size_);
        mmap_addr_ = nullptr;
    }
    if (file_descriptor_ != -1) {
        close(file_descriptor_);
        file_descriptor_ = -1;
    }
#endif

    mmap_size_ = 0;
}

}    // namespace parser
}    // namespace tflite
