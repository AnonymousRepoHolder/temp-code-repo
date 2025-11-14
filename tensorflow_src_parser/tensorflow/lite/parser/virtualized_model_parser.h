#ifndef TENSORFLOW_LITE_PARSER_VIRTUALIZED_MODEL_PARSER_H_
#define TENSORFLOW_LITE_PARSER_VIRTUALIZED_MODEL_PARSER_H_
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The NeuralVirtualizer Authors

#include <array>
#include <map>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

// Platform-specific mmap headers
// mmap provides zero-copy file access for parameter data
// Windows: Use memory mapping API via windows.h
// POSIX (Linux/macOS): Use POSIX mmap via sys/mman.h
#ifdef _WIN32
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace tflite {
namespace parser {

extern const std::map<int, std::string> op_type_mapping;
extern const std::set<std::string> PARAM_OPS;
extern const std::set<std::string> MANDATORY_PARAM_OPS;

std::string get_op_type_from_deprecated_builtin_code(int deprecated_builtin_code);
nlohmann::json convert_dtype_for_json(const nlohmann::json& obj);

struct DevirtualizedOp {
  int index;
  std::string op_type;
  nlohmann::json builtin_options;
  std::vector<int> mutating_variable_inputs;
  std::vector<int> v_forward_connections;
};

struct DevirtualizedParam {
  int index;
  int op_index;
  long long start_pos;
  long long length;
  std::vector<int> shape;
  std::string dtype;
  int param_slot;  // Restored from encrypted shape header (Phase 1)

  // Zero-copy optimization: store pointer to params_array_ instead of copying data
  // This reduces peak memory usage by avoiding duplicate copies of parameter data
  const uint8_t* data_ptr;       // Pointer to raw parameter data in params_array_
  size_t data_size_bytes;        // Size of parameter data in bytes

  // Legacy fields kept for backward compatibility (deprecated, will be empty)
  std::vector<float> data;
  std::vector<int32_t> data_int32;
  std::vector<float> value;
};

struct DevirtualizedGraph {
  int index;
  std::vector<int> forward_connections;
  std::vector<int> forward_branches;
};

class VirtualizedModelParser {
 private:
  std::string v_infos_path_;
  std::string params_path_;

  // Replace modulus with AES-256-CTR encryption keys and nonce bases
  // Each domain (op/param/graph/shape) has independent key-nonce pair for crypto best practices
  std::array<uint8_t, 32> op_encryption_key_;      // Operator type encryption key (AES-256)
  std::array<uint8_t, 32> param_encryption_key_;   // Parameter position encryption key (AES-256)
  std::array<uint8_t, 32> graph_encryption_key_;   // Graph connection encryption key (AES-256)
  std::array<uint8_t, 32> shape_encryption_key_;   // Shape information encryption key (AES-256)

  std::array<uint8_t, 8> op_nonce_base_;           // Operator nonce base (8 bytes)
  std::array<uint8_t, 8> param_nonce_base_;        // Parameter nonce base (8 bytes)
  std::array<uint8_t, 8> graph_nonce_base_;        // Graph nonce base (8 bytes)
  std::array<uint8_t, 8> shape_nonce_base_;        // Shape nonce base (8 bytes)

  // XOR-based key obfuscation parameters
  // Stored keys are obfuscated using: real_key = obfuscated_key XOR master_mask XOR domain_offset
  // This prevents direct key extraction from binary through static analysis
  std::array<uint8_t, 32> xor_master_mask_;        // Master XOR mask (32 bytes, shared across all domains)
  uint8_t xor_op_offset_;                          // Operator domain offset (1 byte)
  uint8_t xor_param_offset_;                       // Parameter domain offset (1 byte)
  uint8_t xor_graph_offset_;                       // Graph domain offset (1 byte)
  uint8_t xor_shape_offset_;                       // Shape domain offset (1 byte)

  // Modular arithmetic parameters (Zhou Mingyi approach)
  // Formula: virtualized_value = random_multiplier × modulus + real_value
  // Restoration: real_value = virtualized_value % modulus
  uint32_t op_modulus_;                            // Operator modulus (default: 251, covers TFLite op types 0-189)
  uint32_t conn_modulus_;                          // Connection modulus (default: 50021, covers <5000 layers)

  // Model metadata (decrypted from encrypted_metadata field in v_infos.json)
  std::vector<std::vector<int>> input_shapes_;              // Input tensor shapes
  std::vector<std::string> input_dtypes_;                   // Input tensor data types
  std::vector<std::vector<int>> input_shape_signatures_;    // Input shape signatures (for dynamic models)

  nlohmann::json v_infos_;
  std::vector<uint8_t> params_array_;

  // Memory-mapped file management (zero-copy solution for parameter data)
  // When mmap is enabled, params_array_ remains empty and data_ptr points to mmap region
  bool use_mmap_;                    // Whether to use mmap for parameter file access
  void* mmap_addr_;                  // Base address of memory-mapped region (nullptr if not mapped)
  size_t mmap_size_;                 // Size of memory-mapped region in bytes
#ifdef _WIN32
  HANDLE file_handle_;               // Windows file handle for memory mapping
  HANDLE mapping_handle_;            // Windows mapping object handle
#else
  int file_descriptor_;              // POSIX file descriptor for memory mapping
#endif

  std::vector<DevirtualizedOp> devirtualized_ops_;
  std::vector<DevirtualizedParam> devirtualized_params_;
  std::vector<DevirtualizedGraph> devirtualized_graph_;

  std::vector<int> input_ops_;
  std::vector<int> output_ops_;

  bool ok_ = true;
  std::string error_message_;

  void Fail(const std::string& message);
  void _load_virtualized_data();
  void _load_model_metadata();
  void _infer_input_output_ops();

  // Memory mapping helper functions (platform-agnostic wrappers)
  // These functions provide unified interface for Windows and POSIX mmap operations
  void _mmap_params_file();        // Map parameter file to memory using platform-specific API
  void _munmap_params_file();      // Unmap parameter file and release resources

  // AES-256-CTR decryption helper function
  // Decrypts ciphertext using provided key and nonce
  // Returns decrypted plaintext as vector of bytes
  std::vector<uint8_t> aes_ctr_decrypt(
      const std::vector<uint8_t>& ciphertext,
      const std::array<uint8_t, 32>& key,
      const std::array<uint8_t, 16>& nonce);

  // XOR-based key deobfuscation helper function
  // Applies XOR deobfuscation to recover real encryption key from obfuscated key
  // Formula: real_key[i] = obfuscated_key[i] XOR master_mask[i] XOR domain_offset
  std::array<uint8_t, 32> deobfuscate_key(
      const std::array<uint8_t, 32>& obfuscated_key,
      uint8_t domain_offset) const;

public:
  VirtualizedModelParser(const std::string& v_infos_path = "v_infos.json",
                        const std::string& params_path = "params.bin",
                        bool use_mmap = true);  // Enable mmap by default for zero-copy optimization
  ~VirtualizedModelParser();  // Destructor now needs to clean up mmap resources

  std::vector<DevirtualizedOp> de_virtualize_op_types();
  std::vector<DevirtualizedParam> de_virtualize_params();
  std::vector<DevirtualizedGraph> de_virtualize_graph();
  std::pair<std::vector<int>, std::vector<int>> identify_input_output_ops();
  std::vector<int> get_execution_order();
  std::vector<DevirtualizedParam> get_op_params(int op_index);
  // Zero-copy version: return a pointer to the existing parameters to avoid deep copies
  std::vector<const DevirtualizedParam*> get_op_params_ptrs(int op_index) const;
  DevirtualizedOp get_op_info(int op_index);
  std::vector<int> get_op_forward_connection(int op_index);
  std::vector<int> get_op_forward_branch(int op_index);
  bool validate_model_integrity();
  void export_model_summary(const std::string& output_path = "model_summary.json");

  // Zero-copy optimization: release params_array_ after FlatBuffer construction
  // This should be called only after all parameter data pointers are no longer needed
  // NOTE: This function does nothing when mmap is enabled, as mmap memory is managed separately
  void release_params_array() {
    if (!use_mmap_) {
      params_array_.clear();
      params_array_.shrink_to_fit();
    }
    // When mmap is enabled, memory is released in destructor via _munmap_params_file()
  }

  // Phase 2 optimization: Check if mmap is being used for parameter file access
  // This is needed to determine if MADV_DONTNEED can be applied for page-level memory release
  bool is_using_mmap() const { return use_mmap_; }

  // Phase 2 optimization: Get mmap base address for calculating page-aligned offsets
  // Returns nullptr if mmap is not enabled
  void* get_mmap_base_address() const { return mmap_addr_; }

  // Phase 2 optimization: Get mmap region size for boundary checking
  size_t get_mmap_size() const { return mmap_size_; }

  bool ok() const { return ok_; }
  const std::string& error_message() const { return error_message_; }

  const std::vector<DevirtualizedOp>& get_devirtualized_ops() const { return devirtualized_ops_; }
  const std::vector<DevirtualizedParam>& get_devirtualized_params() const { return devirtualized_params_; }
  const std::vector<DevirtualizedGraph>& get_devirtualized_graph() const { return devirtualized_graph_; }
  const nlohmann::json& get_v_infos() const { return v_infos_; }
  const std::vector<uint8_t>& get_params_array() const { return params_array_; }
  const std::vector<int>& get_input_ops() const { return input_ops_; }
  const std::vector<int>& get_output_ops() const { return output_ops_; }

  // Getter methods for model metadata (decrypted from v_infos.json)
  const std::vector<std::vector<int>>& get_input_shapes() const { return input_shapes_; }
  const std::vector<std::string>& get_input_dtypes() const { return input_dtypes_; }
  const std::vector<std::vector<int>>& get_input_shape_signatures() const { return input_shape_signatures_; }
};

}  // namespace parser
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PARSER_VIRTUALIZED_MODEL_PARSER_H_
