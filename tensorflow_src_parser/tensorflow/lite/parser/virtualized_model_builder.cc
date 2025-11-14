#include "tensorflow/lite/parser/virtualized_model_builder.h"
#include <algorithm>
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The NeuralVirtualizer Authors
#include <cmath>
#include <numeric>
#include <set>
#include <queue>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cctype>

#ifdef __linux__
#include <unistd.h>    // sysconf
#include <sys/mman.h>  // madvise
#endif

namespace tflite {
namespace parser {
// Helper conversions for parameter payloads
namespace {
std::vector<int> ParamDataToIntVector(const DevirtualizedParam& param) {
    // Zero-copy optimization: use data_ptr instead of legacy data/data_int32 vectors
    if (param.data_ptr != nullptr && param.data_size_bytes > 0) {
        std::vector<int> values;

        // Determine element count based on dtype
        if (param.dtype == "int32") {
            const size_t element_count = param.data_size_bytes / sizeof(int32_t);
            values.reserve(element_count);
            const int32_t* int_data = reinterpret_cast<const int32_t*>(param.data_ptr);
            for (size_t i = 0; i < element_count; ++i) {
                values.push_back(static_cast<int>(int_data[i]));
            }
        } else {
            // Assume float32 dtype
            const size_t element_count = param.data_size_bytes / sizeof(float);
            values.reserve(element_count);
            const float* float_data = reinterpret_cast<const float*>(param.data_ptr);
            for (size_t i = 0; i < element_count; ++i) {
                values.push_back(static_cast<int>(std::lround(float_data[i])));
            }
        }
        return values;
    }

    // Fallback to legacy fields for backward compatibility (should be empty in new code)
    if (!param.data.empty()) {
        std::vector<int> values;
        values.reserve(param.data.size());
        for (float value : param.data) {
            values.push_back(static_cast<int>(std::lround(value)));
        }
        return values;
    }
    if (!param.data_int32.empty()) {
        return std::vector<int>(param.data_int32.begin(), param.data_int32.end());
    }
    return {};
}

// Overloaded version that accepts a pointer.
std::vector<int> ParamDataToIntVector(const DevirtualizedParam* param) {
    if (!param) return {};
    return ParamDataToIntVector(*param);
}

}  // namespace


// Constructor for VirtualizedModelBuilder
VirtualizedModelBuilder::VirtualizedModelBuilder(const std::string& v_infos_path,
                                                const std::string& params_path,
                                                BuildOutput output) {
    // Parse virtualized model information
    parser_ = std::make_unique<VirtualizedModelParser>(v_infos_path, params_path);
    if (!CheckParserReady()) {
        return;
    }
    
    // Get parsed data
    v_infos_ = parser_->get_v_infos();
    devirtualized_ops_ = parser_->get_devirtualized_ops();
    devirtualized_graph_ = parser_->get_devirtualized_graph();
    execution_order_ = parser_->get_execution_order();
    if (!CheckParserReady()) {
        return;
    }
    
    auto [input_ops, output_ops] = parser_->identify_input_output_ops();
    if (!CheckParserReady()) {
        return;
    }
    input_ops_ = input_ops;
    output_ops_ = output_ops;
    
    // Infer tensors, operators, connection structure
    tensor_indices_ = _assign_tensor_indices();
    if (!CheckParserReady()) {
        return;
    }
    tensor_shapes_ = _infer_tensor_shapes();
    if (!CheckParserReady()) {
        return;
    }
    
    opcode_map_ = _build_opcode_map();
    if (!CheckParserReady()) {
        return;
    }
    input_tensor_indices_ = _get_input_tensor_indices();
    if (!CheckParserReady()) {
        return;
    }
    output_tensor_indices_ = _get_output_tensor_indices();
    if (!CheckParserReady()) {
        return;
    }
    
    // Build TFLite model buffer according to desired ownership
    if (output == BuildOutput::kDetached) {
        detached_buffer_ = _build_tflite_model_detached();
        if (!CheckParserReady()) {
            return;
        }
    } else {
        model_buffer_ = _build_tflite_model_buffer();
        if (!CheckParserReady()) {
            return;
        }
    }
    
    // Release large memory blocks after building
    _free_build_artifacts();
}

bool VirtualizedModelBuilder::CheckParserReady() {
    if (!ok_) {
        return false;
    }
    if (!parser_ || !parser_->ok()) {
        Fail(parser_ ? parser_->error_message() : std::string("VirtualizedModelParser is not initialized."));
        return false;
    }
    return true;
}

void VirtualizedModelBuilder::Fail(const std::string& message) {
    if (ok_) {
        ok_ = false;
        error_message_ = message;
    }
}
// Assign tensor indices
std::map<std::string, int> VirtualizedModelBuilder::_assign_tensor_indices() {
    if (!CheckParserReady()) {
        return {};
    }

    std::map<std::string, int> tensor_indices;
    int current_index = 0;
    
    // Input tensors
    for (size_t i = 0; i < input_ops_.size(); ++i) {
        tensor_indices["input_" + std::to_string(i)] = current_index++;
    }
    
    // Intermediate tensors (operator outputs)
    for (int op_index : execution_order_) {
        // Handle multi-output operators separately (e.g., SPLIT)
        auto op_info = parser_->get_op_info(op_index);
        if (!CheckParserReady()) {
            return {};
        }
        if (op_info.op_type == "SPLIT") {
            auto options = op_info.builtin_options;
            int num_splits = 3; // Default 3
            if (options.contains("num_splits")) {
                num_splits = options["num_splits"].get<int>();
            }
            
            tensor_indices["op_" + std::to_string(op_index) + "_output"] = current_index++;
            for (int j = 1; j < num_splits; ++j) {
                tensor_indices["op_" + std::to_string(op_index) + "_output_" + std::to_string(j)] = current_index++;
            }
        } else {
            tensor_indices["op_" + std::to_string(op_index) + "_output"] = current_index++;
        }
    }
    
    // Parameter tensors
    const auto& operators_array = v_infos_["operators"];
    for (const auto& v_op : operators_array) {
        int op_index = v_op["index"];
        auto v_position_data = v_op.value("v_position_data", std::vector<std::string>());
        for (size_t i = 0; i < v_position_data.size(); ++i) {
            tensor_indices["op_" + std::to_string(op_index) + "_param_" + std::to_string(i)] = current_index++;
        }
    }
    
    return tensor_indices;
}

// Infer tensor shapes
std::map<std::string, std::vector<int>> VirtualizedModelBuilder::_infer_tensor_shapes() {
    if (!CheckParserReady()) {
        return {};
    }

    std::map<std::string, std::vector<int>> tensor_shapes;

    // Get input metadata from parser (decrypted from v_infos.json)
    auto input_shapes = parser_->get_input_shapes();
    auto input_dtypes = parser_->get_input_dtypes();
    auto input_shape_signatures = parser_->get_input_shape_signatures();

    for (size_t i = 0; i < input_ops_.size(); ++i) {
        if (i < input_shapes.size()) {
            tensor_shapes["input_" + std::to_string(i)] = input_shapes[i];
        } else {
            tensor_shapes["input_" + std::to_string(i)] = input_shapes[0];
        }
    }
    
    // Intermediate tensor shape inference
    for (int op_index : execution_order_) {
        auto op_info_iter = std::find_if(devirtualized_ops_.begin(), devirtualized_ops_.end(),
            [op_index](const DevirtualizedOp& op) { return op.index == op_index; });
        if (op_info_iter == devirtualized_ops_.end()) continue;
        const DevirtualizedOp& op_info = *op_info_iter;
        std::string op_type = op_info.op_type;
        
        // Select previous input shape for unary/default inference
        std::vector<int> prev_shape;
        std::vector<std::vector<int>> candidate_prev_shapes;
        
        // Priority: if this operator directly consumes subgraph input, use corresponding input_i shape
        if (std::find(input_ops_.begin(), input_ops_.end(), op_index) != input_ops_.end()) {
            for (size_t i = 0; i < input_ops_.size(); ++i) {
                if (input_ops_[i] == op_index && i < input_shapes.size()) {
                    candidate_prev_shapes.push_back(input_shapes[i]);
                    break;
                }
            }
        }
        
        // Secondary: select shape with maximum rank from forward connection outputs
        auto forward_connections = parser_->get_op_forward_connection(op_index);
        auto forward_branches = parser_->get_op_forward_branch(op_index);
        
        for (size_t j = 0; j < forward_connections.size(); ++j) {
            int prev_op = forward_connections[j];
            std::string name;
            if (j >= forward_branches.size() || forward_branches[j] == 0) {
                name = "op_" + std::to_string(prev_op) + "_output";
            } else {
                name = "op_" + std::to_string(prev_op) + "_output_" + std::to_string(forward_branches[j]);
            }
            
            auto it = tensor_shapes.find(name);
            if (it != tensor_shapes.end()) {
                candidate_prev_shapes.push_back(it->second);
            }
        }
        
        if (!candidate_prev_shapes.empty()) {
            // Select maximum rank
            prev_shape = *std::max_element(candidate_prev_shapes.begin(), candidate_prev_shapes.end(),
                [](const std::vector<int>& a, const std::vector<int>& b) {
                    return a.size() < b.size();
                });
        } else {
            prev_shape = input_shapes[0]; // Fallback
        }
        
        std::vector<int> output_shape = prev_shape; // Default: keep unchanged
        
        // Infer output shape based on operator type - specific implementation logic for each operator
        if (op_type == "CONV_2D" || op_type == "DEPTHWISE_CONV_2D") {
            // Convolution output shape inference
            auto options = op_info.builtin_options;
            int stride_h = options.value("stride_h", 1);
            int stride_w = options.value("stride_w", 1);
            std::string padding = options.value("padding", "SAME");
            int depth_multiplier = options.value("depth_multiplier", 1);
            (void)depth_multiplier; // Suppress unused variable warning, depth_multiplier is used in DEPTHWISE_CONV_2D logic
            
            // Get convolution kernel shape
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
            if (!op_params_ptrs.empty()) {
                // Sort by shape to find weight parameters
                auto params_sorted = op_params_ptrs;
                std::sort(params_sorted.begin(), params_sorted.end(),
                    [](const DevirtualizedParam* a, const DevirtualizedParam* b) {
                        return a->shape.size() > b->shape.size();
                    });
                
                auto& kernel_shape = params_sorted[0]->shape; // Take highest dimension as weight
                if (kernel_shape.size() >= 4) {
                    int out_channels;
                    if (op_type == "CONV_2D") {
                        // CONV_2D: [out_channels, height, width, in_channels]
                        out_channels = kernel_shape[0];
                    } else { // DEPTHWISE_CONV_2D
                        // DEPTHWISE_CONV_2D: [1, height, width, channels * depth_multiplier]
                        out_channels = kernel_shape[3];
                    }
                    
                    int kh = kernel_shape[1], kw = kernel_shape[2];
                    
                    if (prev_shape.size() >= 4) {
                        int h = prev_shape[1], w = prev_shape[2];
                        int out_h, out_w;
                        
                        if (padding == "SAME") {
                            out_h = static_cast<int>(std::ceil(static_cast<float>(h) / stride_h));
                            out_w = static_cast<int>(std::ceil(static_cast<float>(w) / stride_w));
                        } else { // VALID
                            out_h = static_cast<int>(std::ceil(static_cast<float>(h - kh + 1) / stride_h));
                            out_w = static_cast<int>(std::ceil(static_cast<float>(w - kw + 1) / stride_w));
                        }
                        
                        output_shape = {prev_shape[0], out_h, out_w, out_channels};
                    }
                }
            }
        } else if (op_type == "RELU" || op_type == "RELU6" || op_type == "LOGISTIC" || 
                op_type == "TANH" || op_type == "SOFTMAX" || op_type == "GELU" || 
                op_type == "RSQRT") {
            // Activation/element-wise operations, output shape same as input
            output_shape = prev_shape;
        } else if (op_type == "SQUEEZE") {
            auto options = op_info.builtin_options;
            std::vector<int> in_shape = prev_shape;
            
            if (!in_shape.empty()) {
                std::vector<int> dims;
                if (options.contains("squeeze_dims") && options["squeeze_dims"].is_array()) {
                    for (const auto& d : options["squeeze_dims"]) {
                        dims.push_back(d.get<int>());
                    }
                }
                
                if (!dims.empty()) {
                    // Normalize negative indices
                    int rank = static_cast<int>(in_shape.size());
                    std::vector<int> normalized;
                    for (int d : dims) {
                        if (d < 0) d += rank;
                        if (d >= 0 && d < rank) {
                            normalized.push_back(d);
                        }
                    }
                    
                    std::vector<int> keep;
                    for (size_t i = 0; i < in_shape.size(); ++i) {
                        if (std::find(normalized.begin(), normalized.end(), static_cast<int>(i)) != normalized.end() && 
                            in_shape[i] == 1) {
                            continue;
                        }
                        keep.push_back(in_shape[i]);
                    }
                    output_shape = keep.empty() ? std::vector<int>{1} : keep;
                } else {
                    // Remove all dimensions with size 1
                    std::vector<int> keep;
                    for (int v : in_shape) {
                        if (v != 1) {
                            keep.push_back(v);
                        }
                    }
                    output_shape = keep.empty() ? std::vector<int>{1} : keep;
                }
            }
        } else if (op_type == "ADD" || op_type == "MUL" || op_type == "SUB") {
            // Element-wise operations, output shape same as input
            auto graph_entry = std::find_if(devirtualized_graph_.begin(), devirtualized_graph_.end(),
                [op_index](const DevirtualizedGraph& g) { return g.index == op_index; });
            if (graph_entry != devirtualized_graph_.end()) {
                auto input_ops = graph_entry->forward_connections;
                auto input_branches = graph_entry->forward_branches;
                std::vector<std::vector<int>> input_tensor_shapes;
                for (size_t j = 0; j < input_ops.size(); ++j) {
                    if (input_branches[j] == 0) {
                        std::string name = "op_" + std::to_string(input_ops[j]) + "_output";
                        auto it = tensor_shapes.find(name);
                        if (it != tensor_shapes.end()) {
                            input_tensor_shapes.push_back(it->second);
                        }
                    } else {
                        std::string name = "op_" + std::to_string(input_ops[j]) + "_output_" + std::to_string(input_branches[j]);
                        auto it = tensor_shapes.find(name);
                        if (it != tensor_shapes.end()) {
                            input_tensor_shapes.push_back(it->second);
                        }
                    }
                }
                // Verify all input shapes are the same
                std::set<std::string> shape_strings;
                for (const auto& shape : input_tensor_shapes) {
                    std::string shape_str;
                    for (size_t k = 0; k < shape.size(); ++k) {
                        if (k > 0) shape_str += ",";
                        shape_str += std::to_string(shape[k]);
                    }
                    shape_strings.insert(shape_str);
                }
                if (shape_strings.size() == 1 && !input_tensor_shapes.empty()) {
                    output_shape = input_tensor_shapes[0];
                } else {
                    // If shapes differ, take the first input's shape
                    output_shape = input_tensor_shapes.empty() ? prev_shape : input_tensor_shapes[0];
                }
            } else {
                output_shape = prev_shape;
            }
        } else if (op_type == "CONCATENATION") {
            // Concatenation operation
            auto options = op_info.builtin_options;
            int axis = options.value("axis", 3);
            
            // Get all input tensor shapes
            std::vector<std::vector<int>> input_tensor_shapes;
            for (size_t j = 0; j < forward_connections.size(); ++j) {
                int prev_op = forward_connections[j];
                std::string name;
                if (j >= forward_branches.size() || forward_branches[j] == 0) {
                    name = "op_" + std::to_string(prev_op) + "_output";
                } else {
                    name = "op_" + std::to_string(prev_op) + "_output_" + std::to_string(forward_branches[j]);
                }
                
                auto it = tensor_shapes.find(name);
                if (it != tensor_shapes.end()) {
                    input_tensor_shapes.push_back(it->second);
                }
            }
            
            if (!input_tensor_shapes.empty()) {
                output_shape = input_tensor_shapes[0]; // Copy the first input's shape
                if (axis >= 0 && axis < static_cast<int>(output_shape.size())) {
                    int sum_axis = 0;
                    for (const auto& shape : input_tensor_shapes) {
                        if (axis < static_cast<int>(shape.size())) {
                            sum_axis += shape[axis];
                        }
                    }
                    output_shape[axis] = sum_axis;
                }
            }
        } else if (op_type == "MAX_POOL_2D" || op_type == "AVERAGE_POOL_2D") {
            // Pooling operation
            auto options = op_info.builtin_options;
            int stride_h = options.value("stride_h", 2);
            int stride_w = options.value("stride_w", 2);
            int filter_height = options.value("filter_height", 2);
            int filter_width = options.value("filter_width", 2);
            std::string padding = options.value("padding", "SAME");
            
            if (prev_shape.size() >= 4) {
                int h = prev_shape[1], w = prev_shape[2];
                int out_h, out_w;
                
                if (padding == "SAME") {
                    out_h = static_cast<int>(std::ceil(static_cast<float>(h) / stride_h));
                    out_w = static_cast<int>(std::ceil(static_cast<float>(w) / stride_w));
                } else { // VALID
                    out_h = static_cast<int>(std::ceil(static_cast<float>(h - filter_height + 1) / stride_h));
                    out_w = static_cast<int>(std::ceil(static_cast<float>(w - filter_width + 1) / stride_w));
                }
                
                output_shape = {prev_shape[0], out_h, out_w, prev_shape[3]};
            }
        } else if (op_type == "RESHAPE") {
            // Reshape operation
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
            int input_size = 1;
            for (int dim : prev_shape) {
                input_size *= dim;
            }
            
            std::vector<int> target_shape;
            if (!op_params_ptrs.empty()) {
                auto int_values = ParamDataToIntVector(op_params_ptrs[0]);
                if (!int_values.empty()) {
                    target_shape = std::move(int_values);
                }
            }
            if (target_shape.empty()) {
                auto options = op_info.builtin_options;
                if (options.contains("new_shape") && options["new_shape"].is_array()) {
                    for (const auto& x : options["new_shape"]) {
                        target_shape.push_back(x.get<int>());
                    }
                } else {
                    target_shape = {1, -1};
                }
            }
            // Parse -1 dimension
            if (std::find(target_shape.begin(), target_shape.end(), -1) != target_shape.end()) {
                int known_size = 1;
                for (int dim : target_shape) {
                    if (dim != -1) {
                        known_size *= dim;
                    }
                }
                int unknown_size = known_size > 0 ? input_size / known_size : 1;
                
                for (int& dim : target_shape) {
                    if (dim == -1) {
                        dim = unknown_size;
                        break;
                    }
                }
            }
            
            output_shape = target_shape;
        } else if (op_type == "FULLY_CONNECTED") {
            // Fully connected layer
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
            auto options = op_info.builtin_options;
            bool keep_num_dims = options.value("keep_num_dims", false);
            
            if (!op_params_ptrs.empty() && !op_params_ptrs[0]->shape.empty()) {
                int units = op_params_ptrs[0]->shape[0];
                
                if (keep_num_dims) {
                    // Keep input dimensions, only change the last one
                    output_shape = std::vector<int>(prev_shape.begin(), prev_shape.end() - 1);
                    output_shape.push_back(units);
                } else {
                    // Standard fully connected: flatten input
                    output_shape = {prev_shape[0], units};
                }
            }
        } else if (op_type == "RESIZE_BILINEAR") {
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
            if (!op_params_ptrs.empty()) {
                auto size_values = ParamDataToIntVector(op_params_ptrs[0]);
                if (size_values.size() >= 2) {
                    int new_height = size_values[0];
                    int new_width = size_values[1];
                    if (new_height > 0 && new_width > 0) {
                        output_shape = {prev_shape[0], new_height, new_width, prev_shape[3]};
                    }
                }
            }
        } else if (op_type == "MEAN") {
            // Global average pooling
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
            if (!op_params_ptrs.empty()) {
                auto reduction_indices = ParamDataToIntVector(op_params_ptrs[0]);
                output_shape = prev_shape;
                for (int axis : reduction_indices) {
                    if (axis >= 0 && axis < static_cast<int>(output_shape.size())) {
                        output_shape[axis] = 1;
                    }
                }
            } else {
                // Default global average pooling
                output_shape = {prev_shape[0], 1, 1, prev_shape[3]};
            }
        } else if (op_type == "BATCH_MATMUL") {
            // Batch matrix multiplication, considering adj_x/adj_y, supports one side as constant parameter
            auto forward_connections = parser_->get_op_forward_connection(op_index);
            auto forward_branches = parser_->get_op_forward_branch(op_index);
            
            std::vector<std::vector<int>> data_input_shapes;
            for (size_t j = 0; j < forward_connections.size(); ++j) {
                int prev_op = forward_connections[j];
                int branch = (j < forward_branches.size()) ? forward_branches[j] : 0;
                
                std::string input_name = (branch == 0) ? 
                    ("op_" + std::to_string(prev_op) + "_output") :
                    ("op_" + std::to_string(prev_op) + "_output_" + std::to_string(branch));
                
                auto it = tensor_shapes.find(input_name);
                if (it != tensor_shapes.end()) {
                    data_input_shapes.push_back(it->second);
                } else {
                    data_input_shapes.push_back(prev_shape);
                }
            }
            
            // Parameter side shape (if exists): derive from restored param_slot
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
            std::map<int, std::vector<int>> param_shape_by_slot;
            for (size_t i = 0; i < op_params_ptrs.size(); ++i) {
                int slot = op_params_ptrs[i]->param_slot;
                if (slot == 0 || slot == 1) {
                    param_shape_by_slot[slot] = op_params_ptrs[i]->shape;
                }
            }
            
            // A, B shapes: prioritize parameter slots, otherwise use forward data input
            std::vector<int> a_shape = param_shape_by_slot.count(0) ? 
                param_shape_by_slot[0] : 
                (data_input_shapes.size() >= 1 ? data_input_shapes[0] : prev_shape);
            std::vector<int> b_shape = param_shape_by_slot.count(1) ? 
                param_shape_by_slot[1] : 
                (data_input_shapes.size() >= 2 ? data_input_shapes[1] : prev_shape);
                
            if (a_shape.size() >= 2 && b_shape.size() >= 2) {
                nlohmann::json options = op_info.builtin_options;
                bool adj_x = options.value("adj_x", false);
                bool adj_y = options.value("adj_y", false);
                
                std::vector<int> batch_dims(a_shape.begin(), a_shape.end() - 2);
                int m = adj_x ? a_shape[a_shape.size()-1] : a_shape[a_shape.size()-2];
                int n = adj_y ? b_shape[b_shape.size()-2] : b_shape[b_shape.size()-1];
                
                output_shape = batch_dims;
                output_shape.push_back(m);
                output_shape.push_back(n);
            } else {
                output_shape = a_shape;
            }
        } else if (op_type == "TRANSPOSE") {
            // Transpose operation, rearrange dimensions according to perm parameter
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
            std::vector<int> input_shape = prev_shape;
            int in_rank = static_cast<int>(input_shape.size());
            
            if (!op_params_ptrs.empty()) {
                std::vector<int> perm_list = ParamDataToIntVector(op_params_ptrs[0]);
                if (!perm_list.empty()) {
                    std::vector<int> normalized;
                    bool valid = (perm_list.size() == input_shape.size()) && in_rank > 0;
                    for (int p : perm_list) {
                        int p_val = p;
                        if (p_val < 0) {
                            p_val += in_rank;
                        }
                        if (p_val < 0 || p_val >= in_rank) {
                            valid = false;
                            break;
                        }
                        normalized.push_back(p_val);
                    }
                    if (valid) {
                        std::vector<int> new_shape(input_shape.size());
                        for (size_t idx = 0; idx < normalized.size(); ++idx) {
                            new_shape[idx] = input_shape[normalized[idx]];
                        }
                        output_shape = new_shape;
                    }
                }
            }
        } else if (op_type == "GATHER") {
            // Gather operation
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
            if (!op_params_ptrs.empty()) {
                auto indices_shape = prev_shape;
                auto params_shape = op_params_ptrs[0]->shape;
                auto options = op_info.builtin_options;
                int axis = options.value("axis", 0);
                
                // Gather output shape: params.shape[:axis] + indices.shape + params.shape[axis+1:]
                std::vector<int> result_shape;
                for (int i = 0; i < axis && i < static_cast<int>(params_shape.size()); ++i) {
                    result_shape.push_back(params_shape[i]);
                }
                for (int dim : indices_shape) {
                    result_shape.push_back(dim);
                }
                for (int i = axis + 1; i < static_cast<int>(params_shape.size()); ++i) {
                    result_shape.push_back(params_shape[i]);
                }
                output_shape = result_shape;
            }
        } else if (op_type == "SPLIT") {
            // Split: inputs = [axis(or split_dim), value]
            nlohmann::json options = op_info.builtin_options;
            int num_splits = options.value("num_splits", 3);
            
            // Parse axis (from parameter tensor, slot=0)
            int axis = 0;
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
            if (!op_params_ptrs.empty()) {
                auto axis_values = ParamDataToIntVector(op_params_ptrs[0]);
                if (!axis_values.empty()) {
                    axis = axis_values[0];
                }
            }
            
            // Select value shape: choose non-scalar/highest rank from forward connections
            auto forward_connections = parser_->get_op_forward_connection(op_index);
            auto forward_branches = parser_->get_op_forward_branch(op_index);
            
            std::vector<std::vector<int>> candidate_shapes;
            for (size_t j = 0; j < forward_connections.size(); ++j) {
                int prod = forward_connections[j];
                int branch = (j < forward_branches.size()) ? forward_branches[j] : 0;
                std::string name = (branch == 0) ? 
                    ("op_" + std::to_string(prod) + "_output") :
                    ("op_" + std::to_string(prod) + "_output_" + std::to_string(branch));
                    
                auto it = tensor_shapes.find(name);
                if (it != tensor_shapes.end()) {
                    candidate_shapes.push_back(it->second);
                } else {
                    candidate_shapes.push_back(prev_shape);
                }
            }
            
            // Default take prev_shape, replace if higher rank non-scalar found
            std::vector<int> value_shape = prev_shape;
            int best_rank = static_cast<int>(value_shape.size());
            for (const auto& shp : candidate_shapes) {
                if (static_cast<int>(shp.size()) > best_rank) {
                    value_shape = shp;
                    best_rank = static_cast<int>(shp.size());
                }
            }
            
            // Normalize axis and apply boundary protection
            int rank = static_cast<int>(value_shape.size());
            if (rank == 0) {
                // Unable to infer data rank, placeholder: keep shape unchanged
                output_shape = value_shape;
            } else {
                if (axis < 0) {
                    axis += rank;
                }
                if (axis < 0 || axis >= rank) {
                    // When axis out of bounds, don't modify dimensions, leave to runtime
                    output_shape = value_shape;
                } else {
                    std::vector<int> out_shape = value_shape;
                    if (axis >= 0 && axis < static_cast<int>(out_shape.size())) {
                        int dim_size = out_shape[axis];
                        if (dim_size > 0 && dim_size % num_splits == 0) {
                            out_shape[axis] = dim_size / num_splits;
                        }
                        // else: when dimension unknown or not divisible, keep unchanged
                    }
                    output_shape = out_shape;
                }
            }
        } else if (op_type == "STRIDED_SLICE") {
            // Commonly used for shape vector slicing, result often scalar
            output_shape = {};
        } else if (op_type == "GREATER_EQUAL" || op_type == "SQUARED_DIFFERENCE") {
            // Simplify to first input shape, can upgrade to broadcasting later
            auto graph_it = std::find_if(devirtualized_graph_.begin(), devirtualized_graph_.end(),
                                    [op_index](const DevirtualizedGraph& g) { return g.index == op_index; });
            
            if (graph_it != devirtualized_graph_.end()) {
                const std::vector<int>& input_ops = graph_it->forward_connections;
                const std::vector<int>& input_branches = graph_it->forward_branches;
                
                if (!input_ops.empty()) {
                    std::string first_input_name;
                    if (input_branches.empty() || input_branches[0] == 0) {
                        first_input_name = "op_" + std::to_string(input_ops[0]) + "_output";
                    } else {
                        first_input_name = "op_" + std::to_string(input_ops[0]) + "_output_" + std::to_string(input_branches[0]);
                    }
                    
                    auto it = tensor_shapes.find(first_input_name);
                    if (it != tensor_shapes.end()) {
                        output_shape = it->second;
                    } else {
                        output_shape = prev_shape;
                    }
                } else {
                    output_shape = prev_shape;
                }
            } else {
                output_shape = prev_shape;
            }
        } else if (op_type == "SHAPE") {
            // Output 1D shape vector
            int rank = static_cast<int>(prev_shape.size());
            output_shape = {rank};
        } else if (op_type == "PACK") {
            auto options = op_info.builtin_options;
            int values_count = options.value("values_count", 1);
            int axis = options.value("axis", 0);
            if (axis == 0) {
                output_shape = {values_count};
            } else {
                output_shape = prev_shape;
            }
        } else if (op_type == "EXPAND_DIMS") {
            // Insert 1 at axis
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
            int axis = 0;
            if (!op_params_ptrs.empty()) {
                auto axis_values = ParamDataToIntVector(op_params_ptrs[0]);
                axis = axis_values.empty() ? 0 : axis_values[0];
            }
            std::vector<int> in_shape = prev_shape;
            if (axis < 0) {
                axis += static_cast<int>(in_shape.size()) + 1;
            }
            axis = std::max(0, std::min(axis, static_cast<int>(in_shape.size())));
            std::vector<int> out_shape = in_shape;
            out_shape.insert(out_shape.begin() + axis, 1);
            output_shape = out_shape;
        } else if (op_type == "CAST") {
            output_shape = prev_shape;
        } else if (op_type == "RANGE") {
            // Length decided at runtime, placeholder
            output_shape = {1};
        } else {
            // Default: same as previous output
            output_shape = prev_shape;
        }
        // Set main output tensor shape
        tensor_shapes["op_" + std::to_string(op_index) + "_output"] = output_shape;
        
        // Handle other output tensors for multi-output operators (e.g., SPLIT)
        if (op_type == "SPLIT") {
            auto options = op_info.builtin_options;
            int num_splits = 3;
            if (options.contains("num_splits")) {
                num_splits = options["num_splits"].get<int>();
            }
            // Set same shape for other outputs of SPLIT
            for (int j = 1; j < num_splits; ++j) {
                tensor_shapes["op_" + std::to_string(op_index) + "_output_" + std::to_string(j)] = output_shape;
            }
        }
    }
    
    // Parameter tensors
    const auto& operators_array = v_infos_["operators"];
    for (const auto& op : operators_array) {
        int op_index = op["index"];
        
        // Get all parameters for this operator (using pointers to avoid copy)
        auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
        
        if (!op_params_ptrs.empty()) {
            // Get operator type to determine parameter order
            auto op_info_iter = std::find_if(devirtualized_ops_.begin(), devirtualized_ops_.end(),
                [op_index](const DevirtualizedOp& devop) { return devop.index == op_index; });
            
            std::vector<const DevirtualizedParam*> params_sorted;
            if (op_info_iter != devirtualized_ops_.end() && 
                (op_info_iter->op_type == "CONV_2D" || op_info_iter->op_type == "DEPTHWISE_CONV_2D" || 
                op_info_iter->op_type == "FULLY_CONNECTED")) {
                // For convolution and fully connected operators, sort by shape: multi-dimensional (weights) first, one-dimensional (bias) second
                params_sorted = op_params_ptrs;
                std::sort(params_sorted.begin(), params_sorted.end(), 
                    [](const DevirtualizedParam* a, const DevirtualizedParam* b) {
                        return a->shape.size() > b->shape.size();
                    });
            } else {
                // Other operators maintain original order
                params_sorted = op_params_ptrs;
            }
            
            for (size_t i = 0; i < params_sorted.size(); ++i) {
                std::string tensor_name = "op_" + std::to_string(op_index) + "_param_" + std::to_string(i);
                tensor_shapes[tensor_name] = params_sorted[i]->shape;
            }
        }
    }
    
    return tensor_shapes;
}


// Build opcode mapping
std::map<std::string, int> VirtualizedModelBuilder::_build_opcode_map() {
    std::vector<std::string> op_types;
    for (const auto& op : devirtualized_ops_) {
        if (std::find(op_types.begin(), op_types.end(), op.op_type) == op_types.end()) {
            op_types.push_back(op.op_type);
        }
    }
    
    std::map<std::string, int> opcode_map;
    for (size_t i = 0; i < op_types.size(); ++i) {
        opcode_map[op_types[i]] = static_cast<int>(i);
    }
    
    return opcode_map;
}

// Get input tensor indices
std::vector<int> VirtualizedModelBuilder::_get_input_tensor_indices() {
    std::vector<int> indices;
    for (size_t i = 0; i < input_ops_.size(); ++i) {
        std::string tensor_name = "input_" + std::to_string(i);
        auto it = tensor_indices_.find(tensor_name);
        if (it != tensor_indices_.end()) {
            indices.push_back(it->second);
        } else {
            Fail("Tensor not found: " + tensor_name);
            return {};
        }
    }
    return indices;
}

// Get output tensor indices
std::vector<int> VirtualizedModelBuilder::_get_output_tensor_indices() {
    std::vector<int> indices;
    for (int op_index : output_ops_) {
        std::string prefix = "op_" + std::to_string(op_index) + "_output";
        
        // Find all matching output tensors
        std::vector<std::pair<std::string, int>> outputs;
        for (const auto& pair : tensor_indices_) {
            if (pair.first == prefix || pair.first.substr(0, prefix.length() + 1) == prefix + "_") {
                outputs.push_back(pair);
            }
        }
        
        // Sort by name to ensure order
        std::sort(outputs.begin(), outputs.end(), 
            [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                if (a.first.length() != b.first.length()) {
                    return a.first.length() < b.first.length();
                }
                return a.first < b.first;
            });
        
        for (const auto& output : outputs) {
            indices.push_back(output.second);
        }
    }
    return indices;
}

// Release parsing intermediates to reduce peak memory during FlatBuffer construction
// This function is called after _build_operators completes, when v_infos_ is no longer needed
// but before building subgraph/model structures
void VirtualizedModelBuilder::_release_parsing_intermediates() {
    // Release parser's params_array (if not using mmap)
    // This was already being done, we're keeping it here for clarity
    if (parser_) {
        parser_->release_params_array();
    }

    // Release v_infos_ JSON structure
    // After _build_operators completes, v_infos_["operators"] is no longer needed
    // Note: v_infos_ is used by _determine_operator_tensors, which is called by:
    // - _build_tensors (for dtype inference)
    // - _build_operators (for input construction)
    // Therefore, we can only release it after _build_operators completes
    v_infos_ = nlohmann::json();

    // Release devirtualized_graph_
    // After _infer_tensor_shapes completes (in constructor), graph structure is no longer needed
    devirtualized_graph_.clear();
    devirtualized_graph_.shrink_to_fit();

    // Release devirtualized_ops_
    // SAFE: This function is invoked only after _build_operator_codes() and
    // _build_operators() have completed in the calling build path. Subsequent
    // steps (_build_subgraph/_build_model) consume only FlatBuffers offsets,
    // not the high-level op descriptors.
    devirtualized_ops_.clear();
    devirtualized_ops_.shrink_to_fit();
}

// Release build artifacts to reduce memory usage
void VirtualizedModelBuilder::_free_build_artifacts() {
    // Note: params_array_ has already been released in _build_tflite_model_buffer()
    // for optimal memory usage during FlatBuffer construction

    // Release parser resources (devirtualized_ops_, devirtualized_params_, v_infos_, etc.)
    parser_.reset();
}

// Build TFLite model buffer
std::vector<uint8_t> VirtualizedModelBuilder::_build_tflite_model_buffer() {

    // Pre-allocation optimization (Solution 5):
    // Calculate estimated buffer size to avoid excessive reallocations during FlatBuffer construction
    // This significantly reduces memory allocation overhead and speeds up model building

    // Step 1: Calculate total parameter data size
    size_t total_param_size = 0;
    const auto& params = parser_->get_devirtualized_params();
    for (const auto& param : params) {
        total_param_size += param.data_size_bytes;
    }

    // Step 2: Estimate metadata overhead
    // Each tensor requires approximately 200 bytes for:
    // - Tensor descriptor (name, shape, dtype, quantization info, buffer index)
    // - Offset tables and alignment padding
    size_t metadata_per_tensor = 200;
    size_t metadata_overhead = tensor_indices_.size() * metadata_per_tensor;

    // Additional overhead for operators, subgraph, and model structure
    size_t operator_overhead = devirtualized_ops_.size() * 150;  // ~150 bytes per operator
    size_t structural_overhead = 10 * 1024;  // 10KB for subgraph, model, and opcode tables

    // Step 3: Calculate total estimated size with 5% safety margin
    // Reduced from 10% to 5% for Phase 1 memory optimization
    // The margin accounts for alignment padding and minor size variations
    // If reallocation occurs, a warning will be logged (see check below)
    size_t estimated_size = total_param_size + metadata_overhead + operator_overhead + structural_overhead;
    size_t size_with_margin = estimated_size + estimated_size / 20;  // 5% margin

    // Create FlatBufferBuilder with pre-allocated capacity
    // This avoids ~20 reallocations (exponential growth from 1KB to ~600MB)
    // Expected speedup: 30-45% reduction in build time
    flatbuffers::FlatBufferBuilder builder(size_with_margin);

    auto [buffer_offsets, param_tensor_map] = _build_buffers_and_param_tensor_map(builder);

    auto tensor_offsets = _build_tensors(builder, buffer_offsets, param_tensor_map);

    auto opcode_offsets = _build_operator_codes(builder);

    auto operator_offsets = _build_operators(builder);

    // Phase 1 optimization: Release parsing intermediates early
    // At this point, all operators have been built, and v_infos_ is no longer needed
    // This reduces peak memory usage during subsequent subgraph/model construction
    _release_parsing_intermediates();

    auto subgraph_offset = _build_subgraph(builder, tensor_offsets, operator_offsets);
    std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraph_offsets = {subgraph_offset};

    auto model_offset = _build_model(builder, opcode_offsets, subgraph_offsets, buffer_offsets);

    builder.Finish(model_offset, "TFL3");

    int buffer_size = builder.GetSize();

    if (buffer_size > size_with_margin) {
        std::cerr << "[WARNING] FlatBuffer exceeded pre-allocation by "
                  << ((buffer_size - size_with_margin) / (1024.0 * 1024.0))
                  << " MB - reallocation occurred!" << std::endl;
    }

    // Zero-copy optimization: Use Release() to transfer ownership instead of copying
    // Release() returns a DetachedBuffer which manages the buffer ownership
    auto detached = builder.Release();

    // Convert DetachedBuffer to vector - performs a single copy from the finalized buffer.
    // The builder's internal storage is already released before this copy happens.
    std::vector<uint8_t> result(detached.data(), detached.data() + detached.size());

    return result;
}

// Build TFLite model buffer and return as DetachedBuffer to enable zero-copy handoff.
flatbuffers::DetachedBuffer VirtualizedModelBuilder::_build_tflite_model_detached() {
    // Pre-allocation and construction logic is identical to _build_tflite_model_buffer,
    // except we return the DetachedBuffer directly without copying into std::vector.

    // Step 1: Calculate total parameter data size
    size_t total_param_size = 0;
    const auto& params = parser_->get_devirtualized_params();
    for (const auto& param : params) {
        total_param_size += param.data_size_bytes;
    }

    // Step 2: Estimate overhead
    size_t metadata_per_tensor = 200;
    size_t metadata_overhead = tensor_indices_.size() * metadata_per_tensor;
    size_t operator_overhead = devirtualized_ops_.size() * 150;
    size_t structural_overhead = 10 * 1024;
    size_t estimated_size = total_param_size + metadata_overhead + operator_overhead + structural_overhead;
    size_t size_with_margin = estimated_size + estimated_size / 20;  // 5%

    flatbuffers::FlatBufferBuilder builder(size_with_margin);

    auto [buffer_offsets, param_tensor_map] = _build_buffers_and_param_tensor_map(builder);
    auto tensor_offsets = _build_tensors(builder, buffer_offsets, param_tensor_map);
    auto opcode_offsets = _build_operator_codes(builder);
    auto operator_offsets = _build_operators(builder);
    _release_parsing_intermediates();
    auto subgraph_offset = _build_subgraph(builder, tensor_offsets, operator_offsets);
    std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraph_offsets = {subgraph_offset};
    auto model_offset = _build_model(builder, opcode_offsets, subgraph_offsets, buffer_offsets);
    builder.Finish(model_offset, "TFL3");

    int buffer_size = builder.GetSize();
    if (buffer_size > static_cast<int>(size_with_margin)) {
        std::cerr << "[WARNING] FlatBuffer exceeded pre-allocation by "
                  << ((buffer_size - size_with_margin) / (1024.0 * 1024.0))
                  << " MB - reallocation occurred!" << std::endl;
    }

    // Return DetachedBuffer directly. The caller is responsible for lifetime.
    return builder.Release();
}

// Build buffers and parameter tensor mapping
// Phase 2 optimization: Sort parameters by physical address (data_ptr) to enable
// sequential mmap access with immediate page release, reducing peak memory usage
std::pair<std::vector<flatbuffers::Offset<tflite::Buffer>>, std::map<std::string, int>>
VirtualizedModelBuilder::_build_buffers_and_param_tensor_map(flatbuffers::FlatBufferBuilder& builder) {

    std::map<std::string, int> param_tensor_map;

    // Create empty buffer at index 0
    auto empty_buffer = tflite::CreateBuffer(builder);

    // Phase 2.1: Two-phase approach to maintain correct buffer indexing
    // while copying parameters in physical address order

    // Step 1: Collect all parameter copy tasks with their original buffer indices
    struct ParamCopyTask {
        const DevirtualizedParam* param;
        int original_buffer_index;
        std::string tensor_name;
    };

    std::vector<ParamCopyTask> tasks;
    int buffer_cursor = 1;  // Start from 1, as 0 is the empty buffer

    const auto& operators_array = v_infos_["operators"];
    for (const auto& op : operators_array) {
        int op_index = op["index"];

        // Get all parameters for this operator
        auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);

        if (!op_params_ptrs.empty()) {
            for (size_t i = 0; i < op_params_ptrs.size(); ++i) {
                const auto* op_param = op_params_ptrs[i];
                std::string tensor_name = "op_" + std::to_string(op_index) + "_param_" + std::to_string(i);

                tasks.push_back({
                    op_param,
                    buffer_cursor,
                    tensor_name
                });
                buffer_cursor++;
            }
        }
    }

    // Step 2: Sort tasks by physical address (data_ptr) in params.bin
    // This ensures sequential mmap access, allowing immediate page release after each copy
    // Critical: Python's random.shuffle(v_op_types) means operators_array order != params.bin order
    std::sort(tasks.begin(), tasks.end(),
        [](const ParamCopyTask& a, const ParamCopyTask& b) {
            return a.param->data_ptr < b.param->data_ptr;
        });

    // Step 3: Deduplicate buffers by underlying (data_ptr, size) to enable Buffer sharing.
    // Multiple parameter tensors may reference the exact same bytes (e.g., shared embeddings
    // that appear as different tensors with different shapes). We assign a single Buffer index
    // per unique (data_ptr, size) and let all referencing tensors share it.

    // Map: (data_ptr, size) -> assigned Buffer index (>=1)
    std::map<std::pair<const uint8_t*, size_t>, int> key_to_index;
    // Preserve copy order for sequential mmap access (increasing physical address)
    struct UniqueItem { const uint8_t* ptr; size_t size; int index; };
    std::vector<UniqueItem> uniques;

    int next_index = 1;  // Buffer index starts at 1 (0 is reserved for empty buffer)
    for (const auto& task : tasks) {
        const uint8_t* byte_data = task.param->data_ptr;
        size_t byte_size = task.param->data_size_bytes;
        std::pair<const uint8_t*, size_t> key{byte_data, byte_size};

        auto it = key_to_index.find(key);
        if (it == key_to_index.end()) {
            key_to_index.emplace(key, next_index);
            uniques.push_back(UniqueItem{byte_data, byte_size, next_index});
            ++next_index;
        }
        // Record mapping for this tensor_name -> shared buffer index
        param_tensor_map[task.tensor_name] = key_to_index[key];
    }

    // Step 4: Materialize unique Buffers in increasing physical address order.
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffer_offsets(uniques.size() + 1);
    buffer_offsets[0] = empty_buffer;

    for (const auto& u : uniques) {
        flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data_offset;
        if (u.size > 0 && u.ptr != nullptr) {
            data_offset = _copy_param_data_chunked(builder, u.ptr, u.size);
        } else {
            data_offset = builder.CreateVector(std::vector<uint8_t>{});
        }
        auto buffer = tflite::CreateBuffer(builder, data_offset);
        buffer_offsets[u.index] = buffer;
    }

    return std::make_pair(buffer_offsets, param_tensor_map);
}

// Phase 2.2: Chunked parameter data copy with immediate mmap page release
// This function replaces builder.CreateVector() to reduce peak memory usage
// by releasing source mmap pages immediately after copying each chunk
flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
VirtualizedModelBuilder::_copy_param_data_chunked(
    flatbuffers::FlatBufferBuilder& builder,
    const uint8_t* src_data,
    size_t src_size) {

    // Use CreateUninitializedVector to get a pointer to the destination buffer
    // This avoids an extra copy compared to CreateVector(data, size)
    uint8_t* dest_data = nullptr;
    auto vec_offset = builder.CreateUninitializedVector<uint8_t>(src_size, &dest_data);

    // Copy in chunks and release mmap pages immediately
    // Adaptive chunk size to balance system call overhead and RSS peak.
    // Defaults to 32MB; on Linux for large blobs we reduce to 16MB to
    // shorten the residency of prefetched pages.
    size_t chunk_size = 32ull * 1024ull * 1024ull;  // 32 MB default
#ifdef __linux__
    // Heuristic: if a single parameter blob is large (>=64MB), use 16MB chunks.
    if (src_size >= (64ull * 1024ull * 1024ull)) {
        chunk_size = 16ull * 1024ull * 1024ull;  // 16 MB on Linux for large chunks
    }
#endif

    for (size_t offset = 0; offset < src_size; offset += chunk_size) {
        size_t chunk_len = std::min(chunk_size, src_size - offset);

        // Copy this chunk
        std::memcpy(dest_data + offset, src_data + offset, chunk_len);

        // Phase 2.3: Immediately release the source mmap pages for this chunk
        // This reduces the mmap RSS from 460MB to ~36-40MB (chunk + OS prefetch)
        _release_mmap_pages(src_data + offset, chunk_len);
    }

    return vec_offset;
}

// Phase 2.3: Release mmap pages immediately after copying to reduce RSS peak
// Uses platform-specific APIs: MADV_DONTNEED (Linux), DiscardVirtualMemory (Windows)
void VirtualizedModelBuilder::_release_mmap_pages(const uint8_t* addr, size_t size) {
    // Only applicable when mmap is enabled
    if (!parser_ || !parser_->is_using_mmap()) {
        return;
    }

#ifdef __linux__
    // Linux: MADV_DONTNEED immediately releases physical memory
    // Pages will be reloaded from file on next access (but we won't access them again)
    // Determine system page size safely (fallback to 4096 on error)
    long __page = sysconf(_SC_PAGESIZE);
    const size_t kPageSize = (__page > 0) ? static_cast<size_t>(__page) : static_cast<size_t>(4096);

    // Calculate page-aligned region
    // We must align to page boundaries as madvise operates on pages
    uintptr_t addr_int = reinterpret_cast<uintptr_t>(addr);
    uintptr_t page_aligned_addr = addr_int & ~(kPageSize - 1);  // Round down to page boundary

    // Calculate size including partial pages at both ends
    size_t page_aligned_size =
        ((addr_int - page_aligned_addr + size + kPageSize - 1) / kPageSize) * kPageSize;

    // Safety check: Ensure we're within mmap bounds
    void* mmap_base = parser_->get_mmap_base_address();
    size_t mmap_size = parser_->get_mmap_size();
    uintptr_t mmap_base_int = reinterpret_cast<uintptr_t>(mmap_base);

    if (page_aligned_addr >= mmap_base_int &&
        page_aligned_addr + page_aligned_size <= mmap_base_int + mmap_size) {
        madvise(reinterpret_cast<void*>(page_aligned_addr), page_aligned_size, MADV_DONTNEED);
    }

#elif defined(_WIN32)
    // Windows 10 1703+ supports DiscardVirtualMemory
    // We dynamically check for API availability to support older Windows versions
    typedef DWORD (WINAPI *DiscardVirtualMemoryProc)(PVOID, SIZE_T);
    static DiscardVirtualMemoryProc pDiscardVirtualMemory = nullptr;
    static bool checked = false;

    if (!checked) {
        HMODULE hKernel32 = GetModuleHandleA("kernel32.dll");
        if (hKernel32) {
            pDiscardVirtualMemory = reinterpret_cast<DiscardVirtualMemoryProc>(
                GetProcAddress(hKernel32, "DiscardVirtualMemory"));
        }
        checked = true;
    }

    if (pDiscardVirtualMemory) {
        // Get system page size
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        const size_t kPageSize = si.dwPageSize;

        // Calculate page-aligned region
        uintptr_t addr_int = reinterpret_cast<uintptr_t>(addr);
        uintptr_t page_aligned_addr = addr_int & ~(kPageSize - 1);
        size_t page_aligned_size =
            ((addr_int - page_aligned_addr + size + kPageSize - 1) / kPageSize) * kPageSize;

        // Safety check: Ensure we're within mmap bounds
        void* mmap_base = parser_->get_mmap_base_address();
        size_t mmap_size = parser_->get_mmap_size();
        uintptr_t mmap_base_int = reinterpret_cast<uintptr_t>(mmap_base);

        if (page_aligned_addr >= mmap_base_int &&
            page_aligned_addr + page_aligned_size <= mmap_base_int + mmap_size) {
            pDiscardVirtualMemory(
                reinterpret_cast<PVOID>(page_aligned_addr),
                page_aligned_size);
        }
    }
    // If DiscardVirtualMemory is not available, silently degrade (no madvise support)

#else
    // macOS/BSD: MADV_FREE doesn't guarantee immediate release, depends on memory pressure
    // For simplicity, we don't implement it here (silent degradation)
    // This means macOS will have similar behavior to the original implementation
#endif
}

// Build tensors
std::vector<flatbuffers::Offset<tflite::Tensor>> 
VirtualizedModelBuilder::_build_tensors(flatbuffers::FlatBufferBuilder& builder,
                                        const std::vector<flatbuffers::Offset<tflite::Buffer>>& buffer_offsets,
                                        const std::map<std::string, int>& param_tensor_map) {
    
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensor_offsets(tensor_indices_.size());
    
    // Reverse lookup name mapping and inferred dtype archive
    std::map<int, std::string> index_to_name;
    for (const auto& [name, index] : tensor_indices_) {
        index_to_name[index] = name;
    }
    std::map<std::string, std::string> inferred_tensor_dtypes;
    
    std::vector<std::pair<std::string, int>> sorted_tensor_indices;
    for (const auto& pair : tensor_indices_) {
        sorted_tensor_indices.push_back(pair);
    }
    std::sort(sorted_tensor_indices.begin(), sorted_tensor_indices.end(), 
            [](const auto& a, const auto& b) { return a.second < b.second; });
    
    for (const auto& pair : sorted_tensor_indices) {
        const std::string& tensor_name = pair.first;
        int tensor_index = pair.second;
        
        auto shape_it = tensor_shapes_.find(tensor_name);
        std::vector<int> shape = (shape_it != tensor_shapes_.end()) ? shape_it->second : std::vector<int>{1};
        
        int buffer_index = 0;
        auto param_it = param_tensor_map.find(tensor_name);
        if (param_it != param_tensor_map.end()) {
            buffer_index = param_it->second;
        }
        
        auto name_offset = builder.CreateString(tensor_name);
        auto shape_offset = builder.CreateVector(shape);

        flatbuffers::Offset<tflite::Tensor> tensor;
        flatbuffers::Offset<flatbuffers::Vector<int32_t>> shape_signature_offset;
        bool has_shape_signature = false;

        tflite::TensorType tensor_type = tflite::TensorType_FLOAT32; // default

        // Get input metadata from parser (decrypted from v_infos.json)
        auto input_shapes = parser_->get_input_shapes();
        auto input_dtypes = parser_->get_input_dtypes();
        auto input_shape_signatures = parser_->get_input_shape_signatures();

        for (size_t i = 0; i < input_ops_.size(); i++) {
            if (tensor_name == "input_" + std::to_string(i)) {
                if (i < input_dtypes.size() && input_dtypes[i] == "float32") {
                    tensor_type = tflite::TensorType_FLOAT32;
                } else if (i < input_dtypes.size() && input_dtypes[i] == "int32") {
                    tensor_type = tflite::TensorType_INT32;
                }
                break;
            }
        }
        
        // Parameter tensor type inference
        if (tensor_name.find("param") != std::string::npos) {
            // Parse parameter tensor name: op_X_param_Y
            size_t op_pos = tensor_name.find("op_");
            size_t param_pos = tensor_name.find("_param_");
            if (op_pos != std::string::npos && param_pos != std::string::npos) {
                std::string op_index_str = tensor_name.substr(op_pos + 3, param_pos - op_pos - 3);
                std::string param_idx_str = tensor_name.substr(param_pos + 7); // Number after "_param_"
                int op_index = std::stoi(op_index_str);
                int param_idx = std::stoi(param_idx_str);

                // Use parser's restored parameter descriptors for reliable dtype
                auto op_params_ptrs = parser_->get_op_params_ptrs(op_index);
                if (param_idx >= 0 && static_cast<size_t>(param_idx) < op_params_ptrs.size()) {
                    const auto* p = op_params_ptrs[param_idx];
                    if (p && p->dtype == std::string("int32")) {
                        tensor_type = tflite::TensorType_INT32;
                    } else if (p && p->dtype == std::string("bool")) {
                        tensor_type = tflite::TensorType_BOOL;
                    } else {
                        tensor_type = tflite::TensorType_FLOAT32;
                    }
                }
            }
        }
        
        // based on producer operator type to adjust intermediate tensor dtype
        if (tensor_name.find("op_") == 0 && tensor_name.find("_output") != std::string::npos) {
            // parse producer op index
            size_t op_pos = tensor_name.find("op_");
            size_t output_pos = tensor_name.find("_output");
            if (op_pos != std::string::npos && output_pos != std::string::npos) {
                std::string op_index_str = tensor_name.substr(op_pos + 3, output_pos - op_pos - 3);
                int op_idx = std::stoi(op_index_str);
                
                auto devop_it = std::find_if(devirtualized_ops_.begin(), devirtualized_ops_.end(),
                    [op_idx](const DevirtualizedOp& op) {
                        return op.index == op_idx;
                    });

                const auto& operators_array = v_infos_["operators"];
                auto vinfo_it = std::find_if(operators_array.begin(), operators_array.end(),
                    [op_idx](const nlohmann::json& op) {
                        return op.contains("index") && op["index"] == op_idx;
                    });

                if (devop_it != devirtualized_ops_.end()) {
                    std::string op_type = devop_it->op_type;
                    nlohmann::json options = (vinfo_it != operators_array.end() && vinfo_it->contains("builtin_options")) 
                                        ? vinfo_it->at("builtin_options") : nlohmann::json::object();
                    
                    // pick_first_data_input_dtype function implementation
                    auto pick_first_data_input_dtype = [&]() -> std::string {
                        // get the input indices of this op
                        auto [in_indices, out_indices] = _determine_operator_tensors(*devop_it);

                        // select the dtype of the first non-parameter input (name not contains '_param_')
                        for (int idx : in_indices) {
                            // build tensor name mapping
                            std::string in_name;
                            for (const auto& [name, index] : tensor_indices_) {
                                if (index == idx) {
                                    in_name = name;
                                    break;
                                }
                            }

                            if (in_name.find("_param_") != std::string::npos) {
                                continue;
                            }

                            // find the dtype from the inferred types
                            auto dtype_it = inferred_tensor_dtypes.find(in_name);
                            if (dtype_it != inferred_tensor_dtypes.end()) {
                                return dtype_it->second;
                            }

                            // check if it is an input tensor
                            if (in_name.rfind("input_", 0) == 0) {
                                size_t digits_start = in_name.find('_') + 1;
                                size_t digits_end = digits_start;
                                size_t index_value = 0;
                                bool parsed = false;

                                while (digits_end < in_name.size() &&
                                    std::isdigit(static_cast<unsigned char>(in_name[digits_end]))) {
                                    parsed = true;
                                    index_value =
                                        index_value * 10 + static_cast<size_t>(in_name[digits_end] - '0');
                                    ++digits_end;
                                }

                                if (parsed && index_value < input_dtypes.size()) {
                                    return input_dtypes[index_value];
                                }

                                return "float32";
                            }
                        }

                        return "float32";
                    };
                    
                    if (op_type == "SHAPE") {
                        // out_type default INT32
                        std::string out_type = options.value("out_type", "INT32");
                        if (out_type == "INT32") {
                            tensor_type = tflite::TensorType_INT32;
                        }
                    } else if (op_type == "STRIDED_SLICE") {
                        // output dtype is consistent with the data input; if unknown, use int32
                        std::string cand = pick_first_data_input_dtype();
                        if (cand == "int32") {
                            tensor_type = tflite::TensorType_INT32;
                        } else if (cand == "float32") {
                            tensor_type = tflite::TensorType_FLOAT32;
                        } else if (cand == "bool") {
                            tensor_type = tflite::TensorType_BOOL;
                        } else {
                            tensor_type = tflite::TensorType_INT32;
                        }
                    } else if (op_type == "PACK") {
                        // output dtype is consistent with the first input; if unknown, use int32
                        std::string cand = pick_first_data_input_dtype();
                        if (cand == "int32") {
                            tensor_type = tflite::TensorType_INT32;
                        } else if (cand == "float32") {
                            tensor_type = tflite::TensorType_FLOAT32;
                        } else if (cand == "bool") {
                            tensor_type = tflite::TensorType_BOOL;
                        } else {
                            tensor_type = tflite::TensorType_INT32;
                        }
                    } else if (op_type == "EXPAND_DIMS") {
                        std::string cand = pick_first_data_input_dtype();
                        if (cand == "int32") {
                            tensor_type = tflite::TensorType_INT32;
                        } else if (cand == "float32") {
                            tensor_type = tflite::TensorType_FLOAT32;
                        } else if (cand == "bool") {
                            tensor_type = tflite::TensorType_BOOL;
                        } else {
                            tensor_type = tflite::TensorType_INT32;
                        }
                    } else if (op_type == "RANGE") {
                        // shape/index path, output generally int32
                        tensor_type = tflite::TensorType_INT32;
                    } else if (op_type == "CAST") {
                        std::string out_type = options.value("out_data_type", options.value("out_type", "FLOAT32"));
                        std::string out_type_str = out_type;
                        std::transform(out_type_str.begin(), out_type_str.end(), out_type_str.begin(), ::toupper);
                        if (out_type_str == "INT32") {
                            tensor_type = tflite::TensorType_INT32;
                        } else if (out_type_str == "BOOL") {
                            tensor_type = tflite::TensorType_BOOL;
                        } else {
                            tensor_type = tflite::TensorType_FLOAT32;
                        }
                    } else if (op_type == "GREATER_EQUAL") {
                        tensor_type = tflite::TensorType_BOOL;
                    } else if (op_type == "RESHAPE" || op_type == "TRANSPOSE" || 
                            op_type == "SPLIT" || op_type == "CONCATENATION") {
                        // these operators do not change dtype, inherit their data input
                        std::string cand = pick_first_data_input_dtype();
                        if (cand == "int32" || cand == "float32" || cand == "bool") {
                            if (cand == "int32") tensor_type = tflite::TensorType_INT32;
                            else if (cand == "float32") tensor_type = tflite::TensorType_FLOAT32;
                            else if (cand == "bool") tensor_type = tflite::TensorType_BOOL;
                        }
                    }
                }
            }
        }
        
        if (tensor_name.find("input_", 0) == 0) {
            size_t underscore_pos = tensor_name.find('_');
            size_t pos = underscore_pos + 1;
            int idx = 0;
            bool parsed = false;
            while (pos < tensor_name.size() && std::isdigit(static_cast<unsigned char>(tensor_name[pos]))) {
                parsed = true;
                idx = idx * 10 + (tensor_name[pos] - '0');
                ++pos;
            }
            if (parsed && idx < static_cast<int>(input_shape_signatures.size())) {
                auto& sig = input_shape_signatures[idx];
                shape_signature_offset = builder.CreateVector(sig);
                has_shape_signature = true;
            }
        }

        tensor = tflite::CreateTensor(builder,
                                    shape_offset,           // shape
                                    tensor_type,            // type
                                    buffer_index,           // buffer
                                    name_offset,            // name
                                    /* quantization */ 0,
                                    /* is_variable */ false,
                                    /* sparsity */ 0,
                                    shape_signature_offset,
                                    has_shape_signature);

        tensor_offsets[tensor_index] = tensor;
        
        // record the determined dtype, for subsequent operators to decide their output dtype
        std::string dtype_str = (tensor_type == tflite::TensorType_INT32)
                                    ? "int32"
                                    : (tensor_type == tflite::TensorType_BOOL) ? "bool" : "float32";
        inferred_tensor_dtypes[tensor_name] = dtype_str;
    }
    
    return tensor_offsets;
}

// build operators
std::vector<flatbuffers::Offset<tflite::OperatorCode>> 
VirtualizedModelBuilder::_build_operator_codes(flatbuffers::FlatBufferBuilder& builder) {
    
    std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcode_offsets;
    
    // create operation codes for each unique operator type
    std::vector<std::string> op_types;
    for (const auto& pair : opcode_map_) {
        op_types.resize(std::max(static_cast<size_t>(pair.second + 1), op_types.size()));
        op_types[pair.second] = pair.first;
    }
    
    // key operator version overrides
    std::map<std::string, int> op_version_overrides = {
        {"FULLY_CONNECTED", 5},
        {"GATHER", 3}, 
        {"BATCH_MATMUL", 2}
    };
    
    for (const std::string& op_type : op_types) {
        // find the corresponding deprecated_builtin_code
        int builtin_code = -1;
        for (const auto& mapping : op_type_mapping) {
            if (mapping.second == op_type) {
                builtin_code = mapping.first;
                break;
            }
        }
        
        // set the version number
        int version = (op_version_overrides.find(op_type) != op_version_overrides.end()) 
                    ? op_version_overrides[op_type] : 1;
        
        // fix: only use the new version of builtin_code field, not set deprecated field at the same time
        // only use OperatorCodeAddBuiltinCode
        tflite::OperatorCodeBuilder opcode_builder(builder);
        opcode_builder.add_builtin_code(static_cast<tflite::BuiltinOperator>(builtin_code));
        opcode_builder.add_version(version);
        auto opcode = opcode_builder.Finish();
        opcode_offsets.push_back(opcode);
    }
    
    return opcode_offsets;
}

// build operators
std::vector<flatbuffers::Offset<tflite::Operator>> 
VirtualizedModelBuilder::_build_operators(flatbuffers::FlatBufferBuilder& builder) {
    
    std::vector<flatbuffers::Offset<tflite::Operator>> operator_offsets;
    
    for (const auto& op_info : devirtualized_ops_) {
        
        // determine the input and output tensors of the operator
        auto [inputs, outputs] = _determine_operator_tensors(op_info);
        
        // get the opcode index
        int opcode_index = opcode_map_[op_info.op_type];
        
        // create the operator
        auto inputs_offset = builder.CreateVector(inputs);
        auto outputs_offset = builder.CreateVector(outputs);
        
        // fully implement the builtin_options construction
        tflite::BuiltinOptions builtin_options_type = tflite::BuiltinOptions_NONE;
        std::string op_type = op_info.op_type;
        flatbuffers::Offset<void> builtin_options_offset = 0;
        
        // defensive: ensure builtin_options is a dict
        auto options = op_info.builtin_options;
        if (options.is_null()) {
            options = nlohmann::json::object();
        }
        
        if (op_type == "CONV_2D") {
            int stride_w = options.value("stride_w", 1);
            int stride_h = options.value("stride_h", 1);
            tflite::Padding padding = (options.value("padding", std::string("SAME")) == "VALID") ? 
                                    tflite::Padding_VALID : tflite::Padding_SAME;
            // get the activation function
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            
            // use the same step-by-step build method as Python
            auto conv_options = tflite::CreateConv2DOptions(builder, padding, stride_w, stride_h, activation);
            builtin_options_offset = conv_options.Union();
            builtin_options_type = tflite::BuiltinOptions_Conv2DOptions;
            
        } else if (op_type == "DEPTHWISE_CONV_2D") {
            int stride_w = options.value("stride_w", 1);
            int stride_h = options.value("stride_h", 1);
            int depth_multiplier = options.value("depth_multiplier", 1);
            tflite::Padding padding = (options.value("padding", std::string("SAME")) == "VALID") ? 
                                    tflite::Padding_VALID : tflite::Padding_SAME;
            // get the activation function
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            
            auto depthwise_options = tflite::CreateDepthwiseConv2DOptions(builder, padding, stride_w, stride_h, 
                                                                        depth_multiplier, activation);
            builtin_options_offset = depthwise_options.Union();
            builtin_options_type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
            
        } else if (op_type == "MAX_POOL_2D" || op_type == "AVERAGE_POOL_2D") {
            int stride_w = options.value("stride_w", 2);
            int stride_h = options.value("stride_h", 2);
            int filter_height = options.value("filter_height", 2);
            int filter_width = options.value("filter_width", 2);
            tflite::Padding padding = (options.value("padding", std::string("SAME")) == "VALID") ? 
                                    tflite::Padding_VALID : tflite::Padding_SAME;
            // get the activation function
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            
            auto pool_options = tflite::CreatePool2DOptions(builder, padding, stride_w, stride_h, 
                                                        filter_width, filter_height, activation);
            builtin_options_offset = pool_options.Union();
            builtin_options_type = tflite::BuiltinOptions_Pool2DOptions;
            
        } else if (op_type == "FULLY_CONNECTED") {
            // get the activation function,
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            bool keep_num_dims = options.value("keep_num_dims", false);
            bool asymmetric_quantize_inputs = options.value("asymmetric_quantize_inputs", false);
            
            auto fc_options = tflite::CreateFullyConnectedOptions(builder, activation, tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                                                                keep_num_dims, asymmetric_quantize_inputs);
            builtin_options_offset = fc_options.Union();
            builtin_options_type = tflite::BuiltinOptions_FullyConnectedOptions;
            
        } else if (op_type == "SOFTMAX") {
            float beta = options.value("beta", 1.0f);
            auto softmax_options = tflite::CreateSoftmaxOptions(builder, beta);
            builtin_options_offset = softmax_options.Union();
            builtin_options_type = tflite::BuiltinOptions_SoftmaxOptions;
            
        } else if (op_type == "CONCATENATION") {
            int axis = options.value("axis", 3);
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            
            auto concat_options = tflite::CreateConcatenationOptions(builder, axis, activation);
            builtin_options_offset = concat_options.Union();
            builtin_options_type = tflite::BuiltinOptions_ConcatenationOptions;
            
        } else if (op_type == "ADD") {
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            
            auto add_options = tflite::CreateAddOptions(builder, activation);
            builtin_options_offset = add_options.Union();
            builtin_options_type = tflite::BuiltinOptions_AddOptions;
            
        } else if (op_type == "RESHAPE") {
            // consistent with the original TFLite export: write options.new_shape regardless of whether there is a second output (shape)
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_info.index);
            if (!op_params_ptrs.empty()) {
                auto new_shape = ParamDataToIntVector(op_params_ptrs[0]);
                if (!new_shape.empty()) {
                    auto new_shape_vector = builder.CreateVector(new_shape);
                    auto reshape_options = tflite::CreateReshapeOptions(builder, new_shape_vector);
                    builtin_options_offset = reshape_options.Union();
                    builtin_options_type = tflite::BuiltinOptions_ReshapeOptions;
                }
            }
            
        } else if (op_type == "MUL") {
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            
            auto mul_options = tflite::CreateMulOptions(builder, activation);
            builtin_options_offset = mul_options.Union();
            builtin_options_type = tflite::BuiltinOptions_MulOptions;
            
        } else if (op_type == "MEAN") {
            bool keep_dims = options.value("keep_dims", false);
            auto reducer_options = tflite::CreateReducerOptions(builder, keep_dims);
            builtin_options_offset = reducer_options.Union();
            builtin_options_type = tflite::BuiltinOptions_ReducerOptions;
            
        } else if (op_type == "SQUEEZE") {
            // write squeeze_dims (if any), consistent with the original model
            std::vector<int> dims;
            if (options.contains("squeeze_dims") && options["squeeze_dims"].is_array()) {
                for (const auto& d : options["squeeze_dims"]) {
                    if (d.is_number_integer()) {
                        dims.push_back(d.get<int>());
                    }
                }
            }
            if (!dims.empty()) {
                auto squeeze_dims_vector = builder.CreateVector(dims);
                auto squeeze_options = tflite::CreateSqueezeOptions(builder, squeeze_dims_vector);
                builtin_options_offset = squeeze_options.Union();
                builtin_options_type = tflite::BuiltinOptions_SqueezeOptions;
            }
            
        } else if (op_type == "RESIZE_BILINEAR") {
            bool align_corners = options.value("align_corners", false);
            bool half_pixel_centers = options.value("half_pixel_centers", false);
            auto resize_options = tflite::CreateResizeBilinearOptions(builder, align_corners, half_pixel_centers);
            builtin_options_offset = resize_options.Union();
            builtin_options_type = tflite::BuiltinOptions_ResizeBilinearOptions;
            
        } else if (op_type == "GELU") {
            bool approximate = options.value("approximate", true);
            auto gelu_options = tflite::CreateGeluOptions(builder, approximate);
            builtin_options_offset = gelu_options.Union();
            builtin_options_type = tflite::BuiltinOptions_GeluOptions;
            
        } else if (op_type == "BATCH_MATMUL") {
            bool adj_x = options.value("adj_x", false);
            bool adj_y = options.value("adj_y", false);
            bool asymmetric_quantize_inputs = options.value("asymmetric_quantize_inputs", false);
            auto batch_matmul_options = tflite::CreateBatchMatMulOptions(builder, adj_x, adj_y, asymmetric_quantize_inputs);
            builtin_options_offset = batch_matmul_options.Union();
            builtin_options_type = tflite::BuiltinOptions_BatchMatMulOptions;
            
        } else if (op_type == "TRANSPOSE") {
            auto transpose_options = tflite::CreateTransposeOptions(builder);
            builtin_options_offset = transpose_options.Union();
            builtin_options_type = tflite::BuiltinOptions_TransposeOptions;
            
        } else if (op_type == "GATHER") {
            int axis = options.value("axis", 0);
            int batch_dims = options.value("batch_dims", 0);
            auto gather_options = tflite::CreateGatherOptions(builder, axis, batch_dims);
            builtin_options_offset = gather_options.Union();
            builtin_options_type = tflite::BuiltinOptions_GatherOptions;
            
        } else if (op_type == "SUB") {
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            
            auto sub_options = tflite::CreateSubOptions(builder, activation);
            builtin_options_offset = sub_options.Union();
            builtin_options_type = tflite::BuiltinOptions_SubOptions;
            
        } else if (op_type == "SPLIT") {
            int num_splits = options.value("num_splits", 3);
            auto split_options = tflite::CreateSplitOptions(builder, num_splits);
            builtin_options_offset = split_options.Union();
            builtin_options_type = tflite::BuiltinOptions_SplitOptions;
            
        } else if (op_type == "SHAPE") {
            std::string out_type_str = options.value("out_type", "INT32");
            tflite::TensorType out_type = tflite::TensorType_INT32;
            if (out_type_str == "FLOAT32") out_type = tflite::TensorType_FLOAT32;
            else if (out_type_str == "BOOL") out_type = tflite::TensorType_BOOL;
            
            auto shape_options = tflite::CreateShapeOptions(builder, out_type);
            builtin_options_offset = shape_options.Union();
            builtin_options_type = tflite::BuiltinOptions_ShapeOptions;
            
        } else if (op_type == "STRIDED_SLICE") {
            int begin_mask = options.value("begin_mask", 0);
            int end_mask = options.value("end_mask", 0);
            int ellipsis_mask = options.value("ellipsis_mask", 0);
            int new_axis_mask = options.value("new_axis_mask", 0);
            int shrink_axis_mask = options.value("shrink_axis_mask", 0);
            
            auto strided_slice_options = tflite::CreateStridedSliceOptions(builder, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);
            builtin_options_offset = strided_slice_options.Union();
            builtin_options_type = tflite::BuiltinOptions_StridedSliceOptions;
            
        } else if (op_type == "PACK") {
            int values_count = options.value("values_count", 1);
            int axis = options.value("axis", 0);
            auto pack_options = tflite::CreatePackOptions(builder, values_count, axis);
            builtin_options_offset = pack_options.Union();
            builtin_options_type = tflite::BuiltinOptions_PackOptions;
            
        } else if (op_type == "RANGE") {
            auto range_options = tflite::CreateRangeOptions(builder);
            builtin_options_offset = range_options.Union();
            builtin_options_type = tflite::BuiltinOptions_RangeOptions;
            
        } else if (op_type == "EXPAND_DIMS") {
            auto expand_dims_options = tflite::CreateExpandDimsOptions(builder);
            builtin_options_offset = expand_dims_options.Union();
            builtin_options_type = tflite::BuiltinOptions_ExpandDimsOptions;
            
        } else if (op_type == "CAST") {
            std::string in_type_str = options.value("in_data_type", options.value("in_type", "FLOAT32"));
            std::string out_type_str = options.value("out_data_type", options.value("out_type", "FLOAT32"));

            tflite::TensorType in_type = tflite::TensorType_FLOAT32;
            tflite::TensorType out_type = tflite::TensorType_FLOAT32;

            if (in_type_str == "INT32") in_type = tflite::TensorType_INT32;
            else if (in_type_str == "BOOL") in_type = tflite::TensorType_BOOL;

            if (out_type_str == "INT32") out_type = tflite::TensorType_INT32;
            else if (out_type_str == "BOOL") out_type = tflite::TensorType_BOOL;
            
            auto cast_options = tflite::CreateCastOptions(builder, in_type, out_type);
            builtin_options_offset = cast_options.Union();
            builtin_options_type = tflite::BuiltinOptions_CastOptions;
        
 
        } else if (op_type == "CONCATENATION") {
            int axis = options.value("axis", 3);
            // get the activation function
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            auto concat_options = tflite::CreateConcatenationOptions(builder, axis, activation);
            builtin_options_offset = concat_options.Union();
            builtin_options_type = tflite::BuiltinOptions_ConcatenationOptions;
            
        } else if (op_type == "ADD") {
            // get the activation function
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            auto add_options = tflite::CreateAddOptions(builder, activation);
            builtin_options_offset = add_options.Union();
            builtin_options_type = tflite::BuiltinOptions_AddOptions;
            
        } else if (op_type == "RESHAPE") {
            // process the new_shape parameter of RESHAPE
            std::vector<int> new_shape;
            auto op_params_ptrs = parser_->get_op_params_ptrs(op_info.index);
            
            if (!op_params_ptrs.empty()) {
                auto shape_values = ParamDataToIntVector(op_params_ptrs[0]);
                if (!shape_values.empty()) {
                    new_shape = std::move(shape_values);
                }
            }
            if (new_shape.empty()) {
                if (options.contains("new_shape") && options["new_shape"].is_array()) {
                    for (const auto& val : options["new_shape"]) {
                        new_shape.push_back(val.get<int>());
                    }
                } else {
                    new_shape = {1, -1};
                }
            }
            
            auto new_shape_vec = builder.CreateVector(new_shape);
            auto reshape_options = tflite::CreateReshapeOptions(builder, new_shape_vec);
            builtin_options_offset = reshape_options.Union();
            builtin_options_type = tflite::BuiltinOptions_ReshapeOptions;
            
        } else if (op_type == "MUL") {
            // get the activation function
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            auto mul_options = tflite::CreateMulOptions(builder, activation);
            builtin_options_offset = mul_options.Union();
            builtin_options_type = tflite::BuiltinOptions_MulOptions;
            
        } else if (op_type == "MEAN") {
            bool keep_dims = options.value("keep_dims", false);
            auto reducer_options = tflite::CreateReducerOptions(builder, keep_dims);
            builtin_options_offset = reducer_options.Union();
            builtin_options_type = tflite::BuiltinOptions_ReducerOptions;
            
        } else if (op_type == "SQUEEZE") {
            std::vector<int> squeeze_dims;
            if (options.contains("squeeze_dims") && options["squeeze_dims"].is_array()) {
                for (const auto& dim : options["squeeze_dims"]) {
                    squeeze_dims.push_back(dim.get<int>());
                }
            }
            
            auto squeeze_dims_vec = builder.CreateVector(squeeze_dims);
            auto squeeze_options = tflite::CreateSqueezeOptions(builder, squeeze_dims_vec);
            builtin_options_offset = squeeze_options.Union();
            builtin_options_type = tflite::BuiltinOptions_SqueezeOptions;
            
        } else if (op_type == "RESIZE_BILINEAR") {
            bool align_corners = options.value("align_corners", false);
            bool half_pixel_centers = options.value("half_pixel_centers", false);
            auto resize_options = tflite::CreateResizeBilinearOptions(builder, align_corners, half_pixel_centers);
            builtin_options_offset = resize_options.Union();
            builtin_options_type = tflite::BuiltinOptions_ResizeBilinearOptions;
            
        } else if (op_type == "BATCH_MATMUL") {
            bool adj_x = options.value("adj_x", false);
            bool adj_y = options.value("adj_y", false);
            bool asymmetric_quantize_inputs = options.value("asymmetric_quantize_inputs", false);
            auto bmm_options = tflite::CreateBatchMatMulOptions(builder, adj_x, adj_y, asymmetric_quantize_inputs);
            builtin_options_offset = bmm_options.Union();
            builtin_options_type = tflite::BuiltinOptions_BatchMatMulOptions;
            
        } else if (op_type == "TRANSPOSE") {
            auto transpose_options = tflite::CreateTransposeOptions(builder);
            builtin_options_offset = transpose_options.Union();
            builtin_options_type = tflite::BuiltinOptions_TransposeOptions;
            
        } else if (op_type == "GATHER") {
            int axis = options.value("axis", 0);
            int batch_dims = options.value("batch_dims", 0);
            auto gather_options = tflite::CreateGatherOptions(builder, axis, batch_dims);
            builtin_options_offset = gather_options.Union();
            builtin_options_type = tflite::BuiltinOptions_GatherOptions;
            
        } else if (op_type == "SUB") {
            // get the activation function
            std::string activation_str = options.value("fused_activation_function", "NONE");
            tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;
            if (activation_str == "RELU") activation = tflite::ActivationFunctionType_RELU;
            else if (activation_str == "RELU6") activation = tflite::ActivationFunctionType_RELU6;
            auto sub_options = tflite::CreateSubOptions(builder, activation);
            builtin_options_offset = sub_options.Union();
            builtin_options_type = tflite::BuiltinOptions_SubOptions;
            
        } else if (op_type == "SPLIT") {
            int num_splits = options.value("num_splits", 3);
            auto split_options = tflite::CreateSplitOptions(builder, num_splits);
            builtin_options_offset = split_options.Union();
            builtin_options_type = tflite::BuiltinOptions_SplitOptions;
            
        } else if (op_type == "SHAPE") {
            tflite::TensorType out_type = tflite::TensorType_INT32;
            auto shape_options = tflite::CreateShapeOptions(builder, out_type);
            builtin_options_offset = shape_options.Union();
            builtin_options_type = tflite::BuiltinOptions_ShapeOptions;
            
        } else if (op_type == "STRIDED_SLICE") {
            int begin_mask = options.value("begin_mask", 0);
            int end_mask = options.value("end_mask", 0);
            int ellipsis_mask = options.value("ellipsis_mask", 0);
            int new_axis_mask = options.value("new_axis_mask", 0);
            int shrink_axis_mask = options.value("shrink_axis_mask", 0);
            auto strided_slice_options = tflite::CreateStridedSliceOptions(builder, begin_mask, end_mask, 
                                                                        ellipsis_mask, new_axis_mask, shrink_axis_mask);
            builtin_options_offset = strided_slice_options.Union();
            builtin_options_type = tflite::BuiltinOptions_StridedSliceOptions;
            
        } else if (op_type == "PACK") {
            int values_count = options.value("values_count", 1);
            int axis = options.value("axis", 0);
            auto pack_options = tflite::CreatePackOptions(builder, values_count, axis);
            builtin_options_offset = pack_options.Union();
            builtin_options_type = tflite::BuiltinOptions_PackOptions;
            
        } else if (op_type == "RANGE") {
            auto range_options = tflite::CreateRangeOptions(builder);
            builtin_options_offset = range_options.Union();
            builtin_options_type = tflite::BuiltinOptions_RangeOptions;
            
        } else if (op_type == "EXPAND_DIMS") {
            auto expand_dims_options = tflite::CreateExpandDimsOptions(builder);
            builtin_options_offset = expand_dims_options.Union();
            builtin_options_type = tflite::BuiltinOptions_ExpandDimsOptions;
            
        } else if (op_type == "CAST") {
            tflite::TensorType in_type = tflite::TensorType_FLOAT32;
            tflite::TensorType out_type = tflite::TensorType_FLOAT32;
            auto cast_options = tflite::CreateCastOptions(builder, in_type, out_type);
            builtin_options_offset = cast_options.Union();
            builtin_options_type = tflite::BuiltinOptions_CastOptions;
        }
        
        // create the operator
        auto op = tflite::CreateOperator(builder,
                                       opcode_index,      // opcode_index
                                       inputs_offset,     // inputs
                                       outputs_offset,    // outputs
                                       builtin_options_type,  // builtin_options_type
                                       builtin_options_offset); // builtin_options
        
        operator_offsets.push_back(op);
    }
    
    return operator_offsets;
}

// build the subgraph
flatbuffers::Offset<tflite::SubGraph> 
VirtualizedModelBuilder::_build_subgraph(flatbuffers::FlatBufferBuilder& builder,
                                            const std::vector<flatbuffers::Offset<tflite::Tensor>>& tensor_offsets,
                                            const std::vector<flatbuffers::Offset<tflite::Operator>>& operator_offsets) {
    
    // create the tensor and operator vectors
    auto tensors_offset = builder.CreateVector(tensor_offsets);
    auto operators_offset = builder.CreateVector(operator_offsets);
    
    // create the input and output index vectors
    auto inputs_offset = builder.CreateVector(input_tensor_indices_);
    auto outputs_offset = builder.CreateVector(output_tensor_indices_);
    
    // create the subgraph name
    auto name_offset = builder.CreateString("main");
    
    return tflite::CreateSubGraph(builder,
                                tensors_offset,    // tensors
                                inputs_offset,     // inputs  
                                outputs_offset,    // outputs
                                operators_offset,  // operators
                                name_offset);      // name
}

// build the model
flatbuffers::Offset<tflite::Model> 
VirtualizedModelBuilder::_build_model(flatbuffers::FlatBufferBuilder& builder,
                                        const std::vector<flatbuffers::Offset<tflite::OperatorCode>>& opcode_offsets,
                                        const std::vector<flatbuffers::Offset<tflite::SubGraph>>& subgraph_offsets,
                                        const std::vector<flatbuffers::Offset<tflite::Buffer>>& buffer_offsets) {
    
    // create the various vectors
    auto operator_codes_offset = builder.CreateVector(opcode_offsets);
    auto subgraphs_offset = builder.CreateVector(subgraph_offsets);
    auto buffers_offset = builder.CreateVector(buffer_offsets);
    
    // create the description string
    auto description_offset = builder.CreateString("virtualized model");
    
    return tflite::CreateModel(builder,
                             3,                      // version
                             operator_codes_offset,  // operator_codes
                             subgraphs_offset,       // subgraphs
                             description_offset,     // description
                             buffers_offset);        // buffers
}

// determine the operator tensors
std::pair<std::vector<int>, std::vector<int>> 
VirtualizedModelBuilder::_determine_operator_tensors(const DevirtualizedOp& op_info) {
    std::vector<int> inputs;
    
    // Collect graph_input_slots (still in JSON), param slots come from parser params
    nlohmann::json vinfo = nullptr;
    const auto& operators_array = v_infos_["operators"];
    for (const auto& v : operators_array) {
        if (v["index"].get<int>() == op_info.index) {
            vinfo = v;
            break;
        }
    }
    std::vector<std::vector<int>> graph_input_slots;
    if (!vinfo.is_null() && vinfo.contains("graph_input_slots")) {
        for (const auto& slot_pair : vinfo["graph_input_slots"]) {
            if (slot_pair.is_array() && slot_pair.size() >= 2) {
                graph_input_slots.push_back({slot_pair[0].get<int>(), slot_pair[1].get<int>()});
            }
        }
    }
    
    // data sources: real input (if input operator) + previous layer output
    std::vector<int> data_sources;
    std::vector<std::pair<int, int>> data_sources_meta; // (tensor_index, producer_op_index)
    
    // if input operator
    if (std::find(input_ops_.begin(), input_ops_.end(), op_info.index) != input_ops_.end()) {
        for (size_t i = 0; i < input_ops_.size(); ++i) {
            if (input_ops_[i] == op_info.index) {
                std::string name = "input_" + std::to_string(i);
                auto it = tensor_indices_.find(name);
                if (it != tensor_indices_.end()) {
                    int tidx = it->second;
                    data_sources.push_back(tidx);
                    data_sources_meta.push_back(std::make_pair(tidx, -1)); // -1 represents None
                }
                break;
            }
        }
    }
    
    // forward connections
    auto forward_connections = parser_->get_op_forward_connection(op_info.index);
    auto forward_branches = parser_->get_op_forward_branch(op_info.index);
    
    if (!forward_connections.empty()) {
        for (size_t i = 0; i < forward_connections.size(); ++i) {
            int prev_op = forward_connections[i];
            int branch = (i < forward_branches.size()) ? forward_branches[i] : 0;
            
            std::string prev_name = (branch == 0) ? 
                "op_" + std::to_string(prev_op) + "_output" :
                "op_" + std::to_string(prev_op) + "_output_" + std::to_string(branch);
                
            auto it = tensor_indices_.find(prev_name);
            if (it != tensor_indices_.end()) {
                int tidx = it->second;
                data_sources.push_back(tidx);
                data_sources_meta.push_back(std::make_pair(tidx, prev_op));
            }
        }
    }
    
    // slot->parameter tensor mapping from restored param_slot
    std::map<int, int> slot_to_param;
    auto op_params_ptrs = parser_->get_op_params_ptrs(op_info.index);
    
    if (!op_params_ptrs.empty()) {
        for (size_t i = 0; i < op_params_ptrs.size(); ++i) {
            std::string pname = "op_" + std::to_string(op_info.index) + "_param_" + std::to_string(i);
            int slot = op_params_ptrs[i]->param_slot;
            if (slot != -1) {
                auto it = tensor_indices_.find(pname);
                if (it != tensor_indices_.end()) {
                    slot_to_param[slot] = it->second;
                }
            }
        }
    }
    
    // slot->subgraph input tensor mapping
    std::map<int, int> slot_to_data;
    for (const auto& pair : graph_input_slots) {
        if (pair.size() >= 2) {
            int slot_idx = pair[0];
            int subgraph_input_pos = pair[1];
            if (slot_idx >= 0 && subgraph_input_pos >= 0) {
                std::string name = "input_" + std::to_string(subgraph_input_pos);
                auto it = tensor_indices_.find(name);
                if (it != tensor_indices_.end()) {
                    slot_to_data[slot_idx] = it->second;
                }
            }
        }
    }
    
    // build the input vector (deduplicated)
    std::set<int> used;
    
    // first determine the maximum slot to be traversed
    int max_slot = -1;
    if (!slot_to_param.empty()) {
        max_slot = std::max(max_slot, slot_to_param.rbegin()->first);
    }
    if (!slot_to_data.empty()) {
        max_slot = std::max(max_slot, slot_to_data.rbegin()->first);
    }
    
    // remove data sources covered by slots
    std::vector<int> unique_data_sources;
    for (int tidx : data_sources) {
        bool found_in_slots = false;
        for (const auto& p : slot_to_param) {
            if (p.second == tidx) {
                found_in_slots = true;
                break;
            }
        }
        if (!found_in_slots) {
            for (const auto& d : slot_to_data) {
                if (d.second == tidx) {
                    found_in_slots = true;
                    break;
                }
            }
        }
        if (!found_in_slots) {
            unique_data_sources.push_back(tidx);
        }
    }
    
    // fill the slots in order
    for (int s = 0; s <= max_slot; ++s) {
        auto param_it = slot_to_param.find(s);
        auto data_it = slot_to_data.find(s);
        
        if (param_it != slot_to_param.end()) {
            inputs.push_back(param_it->second);
            used.insert(param_it->second);
        } else if (data_it != slot_to_data.end()) {
            inputs.push_back(data_it->second);
            used.insert(data_it->second);
        } else if (!unique_data_sources.empty()) {
            int tidx = unique_data_sources[0];
            unique_data_sources.erase(unique_data_sources.begin());
            inputs.push_back(tidx);
            used.insert(tidx);
        }
    }
    
    for (int tidx : unique_data_sources) {
        if (used.find(tidx) == used.end()) {
            inputs.push_back(tidx);
            used.insert(tidx);
        }
    }
    
    inputs = _apply_operator_specific_input_logic(op_info, inputs, slot_to_param, slot_to_data, data_sources_meta);
    
    if (!op_params_ptrs.empty() && op_info.op_type != "RESHAPE") {
        for (size_t i = 0; i < op_params_ptrs.size(); ++i) {
            std::string pname = "op_" + std::to_string(op_info.index) + "_param_" + std::to_string(i);
            bool has_slot = (op_params_ptrs[i]->param_slot != -1);
            if (!has_slot) {
                auto it = tensor_indices_.find(pname);
                if (it != tensor_indices_.end()) {
                    inputs.push_back(it->second);
                }
            }
        }
    }
    
    std::vector<int> outputs;
    if (op_info.op_type == "SPLIT") {
        auto options = op_info.builtin_options;
        int num_splits = options.value("num_splits", 3);
        
        auto it = tensor_indices_.find("op_" + std::to_string(op_info.index) + "_output");
        if (it != tensor_indices_.end()) {
            outputs.push_back(it->second);
        }
        
        for (int j = 1; j < num_splits; ++j) {
            auto it_j = tensor_indices_.find("op_" + std::to_string(op_info.index) + "_output_" + std::to_string(j));
            if (it_j != tensor_indices_.end()) {
                outputs.push_back(it_j->second);
            }
        }
    } else {
        auto it = tensor_indices_.find("op_" + std::to_string(op_info.index) + "_output");
        if (it != tensor_indices_.end()) {
            outputs.push_back(it->second);
        }
    }
    
    return std::make_pair(inputs, outputs);
}

std::vector<int> VirtualizedModelBuilder::_apply_operator_specific_input_logic(
    const DevirtualizedOp& op_info, 
    const std::vector<int>& inputs,
    const std::map<int, int>& slot_to_param,
    const std::map<int, int>& slot_to_data,
    const std::vector<std::pair<int, int>>& data_sources_meta) {
    
    std::vector<int> result_inputs = inputs;
    std::string op_type = op_info.op_type;
    
    auto pick_shape_input_from_data = [&]() -> int {
        // build the reverse lookup to query the shape rank
        std::map<int, std::string> index_to_name;
        for (const auto& pair : tensor_indices_) {
            index_to_name[pair.second] = pair.first;
        }
        
        std::vector<std::pair<int, std::vector<int>>> candidates_pack;
        for (const auto& meta : data_sources_meta) {
            int tidx = meta.first;
            int prod = meta.second;
            
            if (prod == -1) continue; // None
            
            DevirtualizedOp prod_dev;
            bool found = false;
            for (const auto& d : devirtualized_ops_) {
                if (d.index == prod) {
                    prod_dev = d;
                    found = true;
                    break;
                }
            }
            if (!found) continue;
            
            std::string tname = index_to_name[tidx];
            auto shape_it = tensor_shapes_.find(tname);
            std::vector<int> tshape = (shape_it != tensor_shapes_.end()) ? shape_it->second : std::vector<int>();
            
            if (prod_dev.op_type == "PACK") {
                // PACK output is usually a 1D shape vector
                candidates_pack.push_back(std::make_pair(tidx, tshape));
            }
        }
        
        if (!candidates_pack.empty()) {
            return candidates_pack[0].first;
        }
        return -1; // None
    };
    
    // specific operator input number and order constraints
    if (op_type == "SHAPE") {
        // only allow 1 input: slot0 from subgraph input or forward data source
        if (!result_inputs.empty()) {
            std::vector<int> enforced;
            
            // use slot0 in slot_to_data first
            auto data_it = slot_to_data.find(0);
            if (data_it != slot_to_data.end()) {
                enforced.push_back(data_it->second);
            } else {
                // use the first non-parameter data source as output
                std::set<int> param_values;
                for (const auto& p : slot_to_param) {
                    param_values.insert(p.second);
                }
                
                std::vector<int> non_param;
                for (int t : result_inputs) {
                    if (param_values.find(t) == param_values.end()) {
                        non_param.push_back(t);
                    }
                }
                
                if (!non_param.empty()) {
                    enforced.push_back(non_param[0]);
                } else if (!result_inputs.empty()) {
                    enforced.push_back(result_inputs[0]);
                }
            }
            result_inputs = enforced;
        }
    } else if (op_type == "RESHAPE") {
        // allow 1 input: [data] or [data, shape]
        
        // 1) shape source priority: param slot1 > data_sources
        int shape_t = -1; // -1 represents None.
        auto param_it = slot_to_param.find(1);
        if (param_it != slot_to_param.end()) {
            shape_t = param_it->second;
        } else {
            int cand = pick_shape_input_from_data();
            if (cand != -1) {
                shape_t = cand;
            }
        }
        
        // 2) data source: priority graph_input_slots slot0; otherwise the first non-parameter data source
        int data_t = -1; // -1 represents None.
        auto data_it = slot_to_data.find(0);
        if (data_it != slot_to_data.end()) {
            data_t = data_it->second;
        } else {
            // select the rank maximum and non-shape subgraph product from data_sources, avoid shape vector as data
            std::map<int, std::string> index_to_name;
            for (const auto& pair : tensor_indices_) {
                index_to_name[pair.second] = pair.first;
            }
            
            int best = -1;
            int best_rank = -1;
            std::set<std::string> shape_like_ops = {"PACK", "SHAPE", "STRIDED_SLICE", "RANGE", "EXPAND_DIMS"};
            
            std::set<int> param_values;
            for (const auto& p : slot_to_param) {
                param_values.insert(p.second);
            }
            
            for (const auto& meta : data_sources_meta) {
                int tidx = meta.first;
                int prod = meta.second;
                
                if (param_values.find(tidx) != param_values.end()) {
                    continue;
                }
                if (shape_t != -1 && tidx == shape_t) {
                    continue;
                }
                
                if (prod != -1) {
                    DevirtualizedOp prod_dev;
                    bool found = false;
                    for (const auto& d : devirtualized_ops_) {
                        if (d.index == prod) {
                            prod_dev = d;
                            found = true;
                            break;
                        }
                    }
                    if (found && shape_like_ops.find(prod_dev.op_type) != shape_like_ops.end()) {
                        continue;
                    }
                }
                
                std::string tname = index_to_name[tidx];
                auto shape_it = tensor_shapes_.find(tname);
                std::vector<int> tshape = (shape_it != tensor_shapes_.end()) ? shape_it->second : std::vector<int>();
                int rank = static_cast<int>(tshape.size());
                
                if (rank > best_rank) {
                    best_rank = rank;
                    best = tidx;
                }
            }
            
            data_t = (best != -1) ? best : ((!result_inputs.empty()) ? result_inputs[0] : -1);
        }
        
        // 3) assemble and clip
        std::vector<int> new_inputs;
        if (data_t != -1) {
            new_inputs.push_back(data_t);
        }
        if (shape_t != -1) {
            new_inputs.push_back(shape_t);
        }
        
        // shape input must be 1D int32; if not, try param slot1 or delete shape input
        if (new_inputs.size() == 2) {
            std::map<int, std::string> index_to_name;
            for (const auto& pair : tensor_indices_) {
                index_to_name[pair.second] = pair.first;
            }
            
            std::string shape_name = index_to_name[new_inputs[1]];
            auto shape_it = tensor_shapes_.find(shape_name);
            std::vector<int> shape_shape = (shape_it != tensor_shapes_.end()) ? shape_it->second : std::vector<int>();
            
            // check if 1D
            if (!(shape_shape.size() == 1)) {
                // try downgrade: if param slot1 exists, replace it, otherwise delete shape input
                auto param_it = slot_to_param.find(1);
                if (param_it != slot_to_param.end()) {
                    new_inputs[1] = param_it->second;
                } else {
                    // most conservative: remove shape input, downgrade to single input Reshape
                    new_inputs.resize(1);
                }
            }
        }
        
        // avoid data and shape pointing to the same tensor
        if (new_inputs.size() == 2 && new_inputs[0] == new_inputs[1]) {
            // try to select another non-shape subgraph product as data
            int alt = -1;
            std::set<std::string> shape_like_ops = {"PACK", "SHAPE", "STRIDED_SLICE", "RANGE", "EXPAND_DIMS"};
            
            for (const auto& meta : data_sources_meta) {
                int tidx = meta.first;
                int prod = meta.second;
                
                if (std::find(new_inputs.begin(), new_inputs.end(), tidx) != new_inputs.end()) {
                    continue;
                }
                
                if (prod != -1) {
                    DevirtualizedOp prod_dev;
                    bool found = false;
                    for (const auto& d : devirtualized_ops_) {
                        if (d.index == prod) {
                            prod_dev = d;
                            found = true;
                            break;
                        }
                    }
                    if (found && shape_like_ops.find(prod_dev.op_type) != shape_like_ops.end()) {
                        continue;
                    }
                }
                
                alt = tidx;
                break;
            }
            
            if (alt != -1) {
                new_inputs[0] = alt;
            } else {
                // cannot find reliable data input, downgrade to single input
                new_inputs.resize(1);
            }
        }
        
        if (new_inputs.empty()) {
            // rollback: at least keep one data input
            new_inputs = (!result_inputs.empty()) ? std::vector<int>{result_inputs[0]} : std::vector<int>();
        }
        
        // limit the maximum number of outputs to 2
        if (new_inputs.size() > 2) {
            new_inputs.resize(2);
        }
        
        result_inputs = new_inputs;
    }
    
    return result_inputs;
}

std::vector<int> VirtualizedModelBuilder::_build_operator_outputs(const DevirtualizedOp& op_info) {
    if (!CheckParserReady()) {
        return {};
    }

    // build the operator outputs
    std::vector<int> outputs;
    
    // process the multi-output operator (like SPLIT)
    if (op_info.op_type == "SPLIT") {
        auto& options = op_info.builtin_options;
        int num_splits = options.value("num_splits", 3);
        
        // add the main output
        auto it = tensor_indices_.find("op_" + std::to_string(op_info.index) + "_output");
        if (it != tensor_indices_.end()) {
            outputs.push_back(it->second);
        }
        
        // add the extra outputs
        for (int j = 1; j < num_splits; ++j) {
            std::string output_name = "op_" + std::to_string(op_info.index) + "_output_" + std::to_string(j);
            auto it_extra = tensor_indices_.find(output_name);
            if (it_extra != tensor_indices_.end()) {
                outputs.push_back(it_extra->second);
            }
        }
    } else {
        // single output operator
        std::string output_name = "op_" + std::to_string(op_info.index) + "_output";
        auto it = tensor_indices_.find(output_name);
        if (it != tensor_indices_.end()) {
            outputs.push_back(it->second);
        }
    }
    
    return outputs;
}

// simplified model information test
std::map<std::string, nlohmann::json> VirtualizedModelBuilder::model_info_test() {
    std::map<std::string, nlohmann::json> info;
    
    info["tensor_count"] = tensor_indices_.size();
    info["op_count"] = devirtualized_ops_.size();
    info["param_count"] = parser_->get_devirtualized_params().size();
    info["input_count"] = input_ops_.size();
    info["output_count"] = output_ops_.size();
    info["model_buffer_size"] = model_buffer_.size();
    
    return info;
}
}  // namespace parser
}  // namespace tflite
