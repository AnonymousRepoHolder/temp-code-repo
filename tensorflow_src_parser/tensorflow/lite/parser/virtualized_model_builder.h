#ifndef TENSORFLOW_LITE_PARSER_VIRTUALIZED_MODEL_BUILDER_H_
#define TENSORFLOW_LITE_PARSER_VIRTUALIZED_MODEL_BUILDER_H_
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The NeuralVirtualizer Authors

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/parser/virtualized_model_parser.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace parser {

struct TensorInfo {
  int index;
  std::string name;
  std::vector<int> shape;
  std::string dtype;
};

class VirtualizedModelBuilder {
public:
  // BuildOutput controls how the final buffer is owned.
  // kVector: legacy path that returns std::vector<uint8_t> (involves a copy).
  // kDetached: zero-copy path that returns flatbuffers::DetachedBuffer.
  enum class BuildOutput { kVector, kDetached };

  VirtualizedModelBuilder(const std::string& v_infos_path = "v_infos.json",
                          const std::string& params_path = "params.bin",
                          BuildOutput output = BuildOutput::kVector);
  ~VirtualizedModelBuilder() = default;

  const std::vector<uint8_t>& get_model_buffer() const { return model_buffer_; }
  
  // Move the model buffer to avoid copying for memory optimization.
  std::vector<uint8_t> release_model_buffer() { return std::move(model_buffer_); }
  // Move the detached buffer to enable zero-copy handoff to TFLite Allocation.
  flatbuffers::DetachedBuffer release_model_detached() { return std::move(detached_buffer_); }
  std::map<std::string, nlohmann::json> model_info_test();

  const std::map<std::string, int>& get_tensor_indices() const {
    return tensor_indices_;
  }
  const std::map<std::string, std::vector<int>>& get_tensor_shapes() const {
    return tensor_shapes_;
  }
  const std::vector<int>& get_execution_order() const { return execution_order_; }
  const std::vector<int>& get_input_ops() const { return input_ops_; }
  const std::vector<int>& get_output_ops() const { return output_ops_; }
  VirtualizedModelParser* get_parser() const { return parser_.get(); }

  bool ok() const { return ok_; }
  const std::string& error_message() const { return error_message_; }

private:
  std::unique_ptr<VirtualizedModelParser> parser_;

  nlohmann::json v_infos_;
  // params_array_ and devirtualized_params_ have been removed; fetch them directly from the parser.
  std::vector<DevirtualizedOp> devirtualized_ops_;
  std::vector<DevirtualizedGraph> devirtualized_graph_;
  std::vector<int> execution_order_;
  std::vector<int> input_ops_;
  std::vector<int> output_ops_;

  std::map<std::string, int> tensor_indices_;
  std::map<std::string, std::vector<int>> tensor_shapes_;
  std::map<std::string, int> opcode_map_;
  std::vector<int> input_tensor_indices_;
  std::vector<int> output_tensor_indices_;

  std::vector<uint8_t> model_buffer_;
  flatbuffers::DetachedBuffer detached_buffer_;

  bool ok_ = true;
  std::string error_message_;

  bool CheckParserReady();
  void Fail(const std::string& message);

  std::map<std::string, int> _assign_tensor_indices();
  std::map<std::string, std::vector<int>> _infer_tensor_shapes();
  std::pair<std::vector<int>, std::vector<int>> _determine_operator_tensors(
      const DevirtualizedOp& op_info);
  std::vector<int> _apply_operator_specific_input_logic(
      const DevirtualizedOp& op_info, const std::vector<int>& inputs,
      const std::map<int, int>& slot_to_param,
      const std::map<int, int>& slot_to_data,
      const std::vector<std::pair<int, int>>& data_sources_meta);
  std::vector<int> _build_operator_outputs(const DevirtualizedOp& op_info);
  std::map<std::string, int> _build_opcode_map();
  std::vector<int> _get_input_tensor_indices();
  std::vector<int> _get_output_tensor_indices();

  std::vector<uint8_t> _build_tflite_model_buffer();
  flatbuffers::DetachedBuffer _build_tflite_model_detached();
  std::pair<std::vector<flatbuffers::Offset<tflite::Buffer>>, std::map<std::string, int>>
  _build_buffers_and_param_tensor_map(flatbuffers::FlatBufferBuilder& builder);

  // Phase 2 optimization: Chunked copy with immediate mmap page release
  // This function replaces direct CreateVector calls to reduce peak memory usage
  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> _copy_param_data_chunked(
      flatbuffers::FlatBufferBuilder& builder,
      const uint8_t* src_data,
      size_t src_size);

  // Phase 2 optimization: Release mmap pages immediately after copying
  // Uses MADV_DONTNEED (Linux) or DiscardVirtualMemory (Windows) to reduce RSS
  void _release_mmap_pages(const uint8_t* addr, size_t size);

  std::vector<flatbuffers::Offset<tflite::Tensor>> _build_tensors(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<flatbuffers::Offset<tflite::Buffer>>& buffer_offsets,
      const std::map<std::string, int>& param_tensor_map);
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> _build_operator_codes(
      flatbuffers::FlatBufferBuilder& builder);
  std::vector<flatbuffers::Offset<tflite::Operator>> _build_operators(
      flatbuffers::FlatBufferBuilder& builder);
  flatbuffers::Offset<tflite::SubGraph> _build_subgraph(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<flatbuffers::Offset<tflite::Tensor>>& tensor_offsets,
      const std::vector<flatbuffers::Offset<tflite::Operator>>& operator_offsets);
  flatbuffers::Offset<tflite::Model> _build_model(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<flatbuffers::Offset<tflite::OperatorCode>>& opcode_offsets,
      const std::vector<flatbuffers::Offset<tflite::SubGraph>>& subgraph_offsets,
      const std::vector<flatbuffers::Offset<tflite::Buffer>>& buffer_offsets);

  void _release_parsing_intermediates();
  void _free_build_artifacts();
};

}  // namespace parser
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PARSER_VIRTUALIZED_MODEL_BUILDER_H_
