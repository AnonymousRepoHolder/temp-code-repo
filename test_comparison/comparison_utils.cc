// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
#include "comparison_utils.h"
#include "memory_monitor.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <optional>
#include <sstream>

// Random number generator
static std::mt19937 gen(42);

void set_random_seed(uint32_t seed) {
    gen.seed(seed);
}

bool has_int32_input(const std::vector<TensorInfo>& input_details) {
    for (const auto& detail : input_details) {
        if (detail.dtype == "int32") {
            return true;
        }
    }
    return false;
}

bool has_dynamic_signature(const std::vector<TensorInfo>& input_details) {
    for (const auto& detail : input_details) {
        for (int dim : detail.shape_signature) {
            if (dim == -1) {
                return true;
            }
        }
    }
    return false;
}

std::vector<TensorData> collect_outputs(tflite::Interpreter* interp, 
                                        const std::vector<TensorInfo>& out_details) {
    std::vector<TensorData> results;
    
    for (const auto& detail : out_details) {
        TensorData tensor_data;
        
        // Get tensor shape from actual output tensor
        TfLiteTensor* output_tensor = interp->tensor(detail.index);
        tensor_data.shape.clear();
        for (int i = 0; i < output_tensor->dims->size; i++) {
            tensor_data.shape.push_back(output_tensor->dims->data[i]);
        }
        
        // Use ACTUAL tensor type from TFLite, not virtualized dtype metadata
        TfLiteType actual_type = output_tensor->type;
        
        // Set dtype and copy data based on actual tensor type
        switch (actual_type) {
            case kTfLiteFloat32: {
                tensor_data.dtype = "float32";
                const float* data = interp->typed_tensor<float>(detail.index);
                size_t size = 1;
                for (int dim : tensor_data.shape) {
                    size *= dim;
                }
                tensor_data.data_float.assign(data, data + size);
                break;
            }
            case kTfLiteInt32: {
                tensor_data.dtype = "int32";
                const int32_t* data = interp->typed_tensor<int32_t>(detail.index);
                size_t size = 1;
                for (int dim : tensor_data.shape) {
                    size *= dim;
                }
                tensor_data.data_int32.assign(data, data + size);
                break;
            }
            case kTfLiteUInt8: {
                tensor_data.dtype = "uint8";
                const uint8_t* data = interp->typed_tensor<uint8_t>(detail.index);
                size_t size = 1;
                for (int dim : tensor_data.shape) {
                    size *= dim;
                }
                tensor_data.data_uint8.assign(data, data + size);
                break;
            }
            case kTfLiteInt8: {
                tensor_data.dtype = "int8";
                const int8_t* data = interp->typed_tensor<int8_t>(detail.index);
                size_t size = 1;
                for (int dim : tensor_data.shape) {
                    size *= dim;
                }
                tensor_data.data_int8.assign(data, data + size);
                break;
            }
            case kTfLiteInt16: {
                tensor_data.dtype = "int16";
                const int16_t* data = interp->typed_tensor<int16_t>(detail.index);
                size_t size = 1;
                for (int dim : tensor_data.shape) {
                    size *= dim;
                }
                tensor_data.data_int16.assign(data, data + size);
                break;
            }
            case kTfLiteUInt16: {
                tensor_data.dtype = "uint16";
                const uint16_t* data = interp->typed_tensor<uint16_t>(detail.index);
                size_t size = 1;
                for (int dim : tensor_data.shape) {
                    size *= dim;
                }
                tensor_data.data_uint16.assign(data, data + size);
                break;
            }
            case kTfLiteBool: {
                tensor_data.dtype = "bool";
                const bool* data = interp->typed_tensor<bool>(detail.index);
                size_t size = 1;
                for (int dim : tensor_data.shape) {
                    size *= dim;
                }
                // Convert bool to uint8 for storage
                tensor_data.data_uint8.resize(size);
                for (size_t i = 0; i < size; i++) {
                    tensor_data.data_uint8[i] = data[i] ? 1 : 0;
                }
                break;
            }
            default:
                // Unsupported type - create empty tensor with warning
                tensor_data.dtype = "unknown";
                std::cout << "Warning: Unsupported output tensor type: " << actual_type << std::endl;
                break;
        }
        
        results.push_back(tensor_data);
    }
    
    return results;
}

std::vector<float> safe_to_float32(const TensorData& x) {
    if (x.dtype == "float32") {
        return x.data_float;
    } else if (x.dtype == "int32") {
        std::vector<float> result;
        result.reserve(x.data_int32.size());
        for (int32_t val : x.data_int32) {
            result.push_back(static_cast<float>(val));
        }
        return result;
    } else if (x.dtype == "uint8") {
        std::vector<float> result;
        result.reserve(x.data_uint8.size());
        for (uint8_t val : x.data_uint8) {
            result.push_back(static_cast<float>(val));
        }
        return result;
    } else if (x.dtype == "int8") {
        std::vector<float> result;
        result.reserve(x.data_int8.size());
        for (int8_t val : x.data_int8) {
            result.push_back(static_cast<float>(val));
        }
        return result;
    } else if (x.dtype == "int16") {
        std::vector<float> result;
        result.reserve(x.data_int16.size());
        for (int16_t val : x.data_int16) {
            result.push_back(static_cast<float>(val));
        }
        return result;
    } else if (x.dtype == "uint16") {
        std::vector<float> result;
        result.reserve(x.data_uint16.size());
        for (uint16_t val : x.data_uint16) {
            result.push_back(static_cast<float>(val));
        }
        return result;
    } else if (x.dtype == "bool") {
        std::vector<float> result;
        result.reserve(x.data_uint8.size());  // bool stored as uint8
        for (uint8_t val : x.data_uint8) {
            result.push_back(static_cast<float>(val));
        }
        return result;
    }
    return {};
}

float avg(const std::vector<float>& lst) {
    if (lst.empty()) return 0.0f;
    return std::accumulate(lst.begin(), lst.end(), 0.0f) / lst.size();
}

MetricsSummary compute_metrics_summary(
        const std::vector<std::vector<TensorData>>& outputs_vir,
        const std::vector<std::vector<TensorData>>& outputs_ori) {
    MetricsSummary summary;
    int num_tests = outputs_ori.size();
    int num_outputs = std::min(
        outputs_ori.empty() ? 0 : (int)outputs_ori[0].size(),
        outputs_vir.empty() ? 0 : (int)outputs_vir[0].size()
    );
    summary.num_tests = num_tests;
    
    std::vector<std::vector<float>> mse_per_output(num_outputs);
    std::vector<std::vector<float>> mae_per_output(num_outputs);
    std::vector<std::vector<float>> maxae_per_output(num_outputs);
    std::vector<std::vector<float>> relmae_per_output(num_outputs);
    std::vector<std::vector<float>> top1_match_per_output(num_outputs);
    
    for (int i = 0; i < num_tests; i++) {
        for (int j = 0; j < num_outputs; j++) {
            auto vir = outputs_vir[i][j];
            auto ori = outputs_ori[i][j];
            
            // Shape handling and error calculation
            std::vector<float> vir_c, ori_c;
            if (vir.shape != ori.shape) {
                if (vir.size() == ori.size()) {
                    vir_c = safe_to_float32(vir.reshape(-1));
                    ori_c = safe_to_float32(ori.reshape(-1));
                } else {
                    std::cout << "Warning: In inference" << i << ", the " << j 
                            << "‑th output has an inconsistent number of elements; skipping error computation" << std::endl;
                    continue;
                }
            } else {
                vir_c = safe_to_float32(vir);
                ori_c = safe_to_float32(ori);
            }
            
            if (vir_c.empty() || ori_c.empty() || vir_c.size() != ori_c.size()) {
                continue;
            }
            
            // Calculate error metrics
            float mse = 0.0f, mae = 0.0f, maxae = 0.0f;
            float relmae = 0.0f;
            
            for (size_t k = 0; k < vir_c.size(); k++) {
                float diff = vir_c[k] - ori_c[k];
                float abs_diff = std::abs(diff);
                float abs_ori = std::abs(ori_c[k]);
                
                mse += diff * diff;
                mae += abs_diff;
                maxae = std::max(maxae, abs_diff);
                // Per-element relative error
                relmae += abs_diff / (abs_ori + 1e-8f);
            }
            
            mse /= vir_c.size();
            mae /= vir_c.size();
            relmae /= vir_c.size(); // Mean of per-element relative errors
            
            mse_per_output[j].push_back(mse);
            mae_per_output[j].push_back(mae);
            maxae_per_output[j].push_back(maxae);
            relmae_per_output[j].push_back(relmae);
            
            // Top-1 accuracy calculation for classification outputs
            // Only apply to 2D outputs (typical classification: [batch, classes])
            // Exclude 3D/4D outputs (regression tasks like depth estimation: [batch, H, W, C])
            try {
                if (vir.shape.size() == 2 && ori.shape.size() == 2) {
                    int last_dim_vir = vir.shape.back();
                    int last_dim_ori = ori.shape.back();

                    if (last_dim_vir == last_dim_ori && last_dim_vir >= 1) {
                        size_t total_elements = vir_c.size();

                        if (last_dim_vir == 1) {
                            // Binary classification with single output (sigmoid)
                            // Use 0.5 as threshold
                            size_t num_samples = total_elements;
                            int matches = 0;

                            for (size_t sample = 0; sample < num_samples; sample++) {
                                bool vir_pred = vir_c[sample] > 0.5f;
                                bool ori_pred = ori_c[sample] > 0.5f;
                                if (vir_pred == ori_pred) {
                                    matches++;
                                }
                            }

                            float top1_acc = static_cast<float>(matches) / static_cast<float>(num_samples);
                            top1_match_per_output[j].push_back(top1_acc);

                        } else {
                            // Multi-class classification (original logic)
                            // Reshape to (-1, last_dim) equivalent
                            size_t num_samples = total_elements / last_dim_vir;

                            int matches = 0;
                            for (size_t sample = 0; sample < num_samples; sample++) {
                                // Find argmax for current sample in vir_c
                                int vir_max_idx = 0;
                                float vir_max_val = vir_c[sample * last_dim_vir];
                                for (int dim = 1; dim < last_dim_vir; dim++) {
                                    if (vir_c[sample * last_dim_vir + dim] > vir_max_val) {
                                        vir_max_val = vir_c[sample * last_dim_vir + dim];
                                        vir_max_idx = dim;
                                    }
                                }

                                // Find argmax for current sample in ori_c
                                int ori_max_idx = 0;
                                float ori_max_val = ori_c[sample * last_dim_vir];
                                for (int dim = 1; dim < last_dim_vir; dim++) {
                                    if (ori_c[sample * last_dim_vir + dim] > ori_max_val) {
                                        ori_max_val = ori_c[sample * last_dim_vir + dim];
                                        ori_max_idx = dim;
                                    }
                                }

                                if (vir_max_idx == ori_max_idx) {
                                    matches++;
                                }
                            }

                            float top1_acc = static_cast<float>(matches) / static_cast<float>(num_samples);
                            top1_match_per_output[j].push_back(top1_acc);
                        }
                    }
                }
            } catch (...) {
                // Handle exceptions silently
            }
        }
    }
    
    for (int j = 0; j < num_outputs; j++) {
        OutputMetrics output_summary;
        output_summary.output_index = j;
        output_summary.mse = avg(mse_per_output[j]);
        output_summary.mae = avg(mae_per_output[j]);
        output_summary.maxae = avg(maxae_per_output[j]);
        output_summary.relmae = avg(relmae_per_output[j]);
        output_summary.has_top1 = !top1_match_per_output[j].empty();
        output_summary.top1_agreement = avg(top1_match_per_output[j]);
        summary.outputs.push_back(output_summary);
    }

    return summary;
}

void print_metrics_summary(const MetricsSummary& summary) {
    std::cout << "\n=== Error statistics (" << summary.num_tests
              << "‑run average) ===" << std::endl;
    for (const auto& output : summary.outputs) {
        printf("Output%d: MSE=%.15f, MAE=%.8f, MaxAE=%.8f, RelMAE=%.8f\n",
               output.output_index, output.mse, output.mae,
               output.maxae, output.relmae);
        if (output.has_top1) {
            printf("Output%d: Top‑1 agreement rate=%.8f\n",
                   output.output_index, output.top1_agreement);
        }
    }
}

void compute_metrics(const std::vector<std::vector<TensorData>>& outputs_vir,
                    const std::vector<std::vector<TensorData>>& outputs_ori) {
    print_metrics_summary(compute_metrics_summary(outputs_vir, outputs_ori));
}

void print_outputs_meta(const std::vector<TensorInfo>& v_out_details,
                        const std::vector<TensorInfo>& o_out_details) {
    // Check tensor count consistency first
    if (v_out_details.size() != o_out_details.size()) {
        std::cout << "Warning: Inconsistent number of output tensors — virtualized: " << v_out_details.size() 
                << ", original: " << o_out_details.size() << std::endl;
    } else {
        std::cout << "The number of output tensors is consistent: " << o_out_details.size() << std::endl;
    }
    
    // Compare each output tensor
    size_t min_outputs = std::min(v_out_details.size(), o_out_details.size());
    for (size_t k = 0; k < min_outputs; k++) {
        const std::string& v_dt = v_out_details[k].dtype;
        const std::string& o_dt = o_out_details[k].dtype;
        const std::vector<int>& v_sh = v_out_details[k].shape;
        const std::vector<int>& o_sh = o_out_details[k].shape;
        
        // Check dtype consistency
        if (v_dt != o_dt) {
            std::cout << "Warning: Output" << k << " dtype mismatch — virtualized: " << v_dt 
                    << ", original: " << o_dt << std::endl;
        }
        
        // Check shape consistency 
        if (v_sh != o_sh) {
            std::cout << "Warning: Output" << k << " Shape mismatch (metadata) - virtualized: [";
            for (size_t i = 0; i < v_sh.size(); i++) {
                std::cout << v_sh[i];
                if (i < v_sh.size() - 1) std::cout << ", ";
            }
            std::cout << "], original: [";
            for (size_t i = 0; i < o_sh.size(); i++) {
                std::cout << o_sh[i];
                if (i < o_sh.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
}

std::vector<int> new_shape_for_length(const TensorInfo& detail, int length_value) {
    std::vector<int> shape = detail.shape;
    std::vector<int> sig = detail.shape_signature;
    
    // Early return if no shape signature or empty
    if (sig.empty()) {
        return shape;
    }
    
    std::vector<int> new_shape = shape;
    int rank = new_shape.size();
    std::string dtype_local = detail.dtype;
    
    if (dtype_local == "int32") {
        // Find dynamic positions (where sig[i] == -1)
        std::vector<int> dyn_pos;
        for (size_t i = 0; i < sig.size(); i++) {
            if (sig[i] == -1) {
                dyn_pos.push_back(i);
            }
        }
        
        // Set all dynamic positions to 1
        for (int pos : dyn_pos) {
            if (pos < new_shape.size()) {
                new_shape[pos] = 1;
            }
        }
        
        // Set the last dynamic position to length_value
        if (!dyn_pos.empty()) {
            int last_dyn_pos = dyn_pos.back();
            if (last_dyn_pos < new_shape.size()) {
                new_shape[last_dyn_pos] = length_value;
            }
        }
    } else {
        // For non-int32 dtypes (typically float32)
        if (rank == 4) {
            // Special handling for 4D tensors (like image data)
            if (sig.size() > 1 && sig[1] == -1) {
                new_shape[1] = length_value;
            }
            if (sig.size() > 2 && sig[2] == -1) {
                new_shape[2] = length_value;
            }
            if (sig.size() > 0 && sig[0] == -1) {
                new_shape[0] = 1;  // Batch dimension
            }
        } else {
            // General case for non-4D tensors
            std::vector<int> dyn_pos;
            for (size_t i = 0; i < sig.size(); i++) {
                if (sig[i] == -1) {
                    dyn_pos.push_back(i);
                }
            }
            
            // Set all dynamic positions to 1
            for (int pos : dyn_pos) {
                if (pos < new_shape.size()) {
                    new_shape[pos] = 1;
                }
            }
            
            // Set the last dynamic position to length_value
            if (!dyn_pos.empty()) {
                int last_dyn_pos = dyn_pos.back();
                if (last_dyn_pos < new_shape.size()) {
                    new_shape[last_dyn_pos] = length_value;
                }
            }
        }
    }
    
    return new_shape;
}

std::vector<TensorInfo> get_input_details(tflite::Interpreter* interp) {
    std::vector<TensorInfo> details;
    
    for (int i = 0; i < interp->inputs().size(); i++) {
        TensorInfo info;
        info.index = interp->inputs()[i];
        
        TfLiteTensor* tensor = interp->input_tensor(i);
        info.name = tensor->name ? tensor->name : "";
        
        // Get shape
        info.shape.clear();
        for (int j = 0; j < tensor->dims->size; j++) {
            info.shape.push_back(tensor->dims->data[j]);
        }
        
        // Get shape signature (for dynamic shapes)
        if (tensor->dims_signature) {
            info.shape_signature.clear();
            for (int j = 0; j < tensor->dims_signature->size; j++) {
                info.shape_signature.push_back(tensor->dims_signature->data[j]);
            }
        } else {
            info.shape_signature = info.shape;
        }
        
        // Get data type
        switch (tensor->type) {
            case kTfLiteFloat32:
                info.dtype = "float32";
                break;
            case kTfLiteInt32:
                info.dtype = "int32";
                break;
            case kTfLiteUInt8:
                info.dtype = "uint8";
                break;
            case kTfLiteInt8:
                info.dtype = "int8";
                break;
            case kTfLiteInt16:
                info.dtype = "int16";
                break;
            case kTfLiteUInt16:
                info.dtype = "uint16";
                break;
            case kTfLiteBool:
                info.dtype = "bool";
                break;
            default:
                info.dtype = "unknown";
                break;
        }
        
        // Get quantization info
        if (tensor->quantization.type != kTfLiteNoQuantization) {
            info.quantization_scale = tensor->params.scale;
            info.quantization_zero_point = tensor->params.zero_point;
        }
        
        details.push_back(info);
    }
    
    return details;
}

std::vector<TensorInfo> get_output_details(tflite::Interpreter* interp) {
    std::vector<TensorInfo> details;
    
    for (int i = 0; i < interp->outputs().size(); i++) {
        TensorInfo info;
        info.index = interp->outputs()[i];
        
        TfLiteTensor* tensor = interp->output_tensor(i);
        info.name = tensor->name ? tensor->name : "";
        
        // Get shape
        info.shape.clear();
        for (int j = 0; j < tensor->dims->size; j++) {
            info.shape.push_back(tensor->dims->data[j]);
        }
        
        // Get data type
        switch (tensor->type) {
            case kTfLiteFloat32:
                info.dtype = "float32";
                break;
            case kTfLiteInt32:
                info.dtype = "int32";
                break;
            case kTfLiteUInt8:
                info.dtype = "uint8";
                break;
            case kTfLiteInt8:
                info.dtype = "int8";
                break;
            case kTfLiteInt16:
                info.dtype = "int16";
                break;
            case kTfLiteUInt16:
                info.dtype = "uint16";
                break;
            case kTfLiteBool:
                info.dtype = "bool";
                break;
            default:
                info.dtype = "unknown";
                break;
        }
        
        details.push_back(info);
    }
    
    return details;
}

std::unique_ptr<tflite::Interpreter> create_interpreter_from_model(
    const tflite::FlatBufferModel* model) {
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    
    tflite::InterpreterBuilder builder(*model, resolver);
    if (builder(&interpreter) != kTfLiteOk) {
        return nullptr;
    }
    
    return interpreter;
}

void set_interpreter_inputs(tflite::Interpreter* interp, 
                            std::vector<TestData>& inputs) {
    for (auto& input : inputs) {
        auto& tensor_data = input.tensor_data;
        
        // Get the actual tensor type from TFLite interpreter
        TfLiteTensor* actual_tensor = interp->input_tensor(input.input_index);
        TfLiteType actual_type = actual_tensor->type;
        
        // Calculate total size for fallback generation
        size_t total_size = 1;
        for (int dim : tensor_data.shape) total_size *= dim;
        
        // Convert and set data based on ACTUAL tensor type, using TestData for consistency
        switch (actual_type) {
            case kTfLiteFloat32: {
                float* input_data = interp->typed_input_tensor<float>(input.input_index);
                if (tensor_data.dtype == "float32" && tensor_data.data_float.size() == total_size) {
                    std::copy(tensor_data.data_float.begin(), tensor_data.data_float.end(), input_data);
                } else if (tensor_data.dtype == "int32" && tensor_data.data_int32.size() == total_size) {
                    // Convert int32 to float32
                    for (size_t i = 0; i < total_size; i++) {
                        input_data[i] = static_cast<float>(tensor_data.data_int32[i]);
                    }
                } else {
                    // Fallback: generate consistent float32 test data and cache it
                    std::vector<float> generated_data(total_size);
                    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
                    for (size_t i = 0; i < total_size; i++) {
                        generated_data[i] = dis(gen);
                        input_data[i] = generated_data[i];
                    }
                    // Write back to TestData for consistency in future calls
                    tensor_data.data_float = generated_data;
                    tensor_data.dtype = "float32";
                }
                break;
            }
            case kTfLiteInt32: {
                int32_t* input_data = interp->typed_input_tensor<int32_t>(input.input_index);
                if (tensor_data.dtype == "int32" && tensor_data.data_int32.size() == total_size) {
                    std::copy(tensor_data.data_int32.begin(), tensor_data.data_int32.end(), input_data);
                } else if (tensor_data.dtype == "float32" && tensor_data.data_float.size() == total_size) {
                    // Convert float32 to int32
                    for (size_t i = 0; i < total_size; i++) {
                        input_data[i] = static_cast<int32_t>(tensor_data.data_float[i]);
                    }
                } else {
                    // Fallback: generate consistent int32 test data and cache it
                    std::vector<int32_t> generated_data(total_size);
                    std::uniform_int_distribution<int32_t> dis(0, 100);
                    for (size_t i = 0; i < total_size; i++) {
                        generated_data[i] = dis(gen);
                        input_data[i] = generated_data[i];
                    }
                    // Write back to TestData for consistency in future calls
                    tensor_data.data_int32 = generated_data;
                    tensor_data.dtype = "int32";
                }
                break;
            }
            case kTfLiteUInt8: {
                uint8_t* input_data = interp->typed_input_tensor<uint8_t>(input.input_index);
                if (tensor_data.dtype == "uint8" && tensor_data.data_uint8.size() == total_size) {
                    std::copy(tensor_data.data_uint8.begin(), tensor_data.data_uint8.end(), input_data);
                } else if (tensor_data.dtype == "float32" && tensor_data.data_float.size() == total_size) {
                    // Convert from float32 [0,1] to uint8 [0,255] - DETERMINISTIC
                    for (size_t i = 0; i < total_size; i++) {
                        float val = std::max(0.0f, std::min(1.0f, tensor_data.data_float[i]));
                        input_data[i] = static_cast<uint8_t>(val * 255.0f);
                    }
                } else if (tensor_data.dtype == "int32" && tensor_data.data_int32.size() == total_size) {
                    // Convert from int32 to uint8 - DETERMINISTIC
                    for (size_t i = 0; i < total_size; i++) {
                        int32_t val = std::max(0, std::min(255, tensor_data.data_int32[i]));
                        input_data[i] = static_cast<uint8_t>(val);
                    }
                } else {
                    // Fallback: generate consistent uint8 data and cache it
                    std::vector<uint8_t> generated_data(total_size);
                    std::uniform_int_distribution<uint16_t> dis(0, 255);
                    for (size_t i = 0; i < total_size; i++) {
                        generated_data[i] = static_cast<uint8_t>(dis(gen));
                        input_data[i] = generated_data[i];
                    }
                    // Write back to TestData for consistency in future calls
                    tensor_data.data_uint8 = generated_data;
                    tensor_data.dtype = "uint8";
                }
                break;
            }
            case kTfLiteInt8: {
                int8_t* input_data = interp->typed_input_tensor<int8_t>(input.input_index);
                if (tensor_data.dtype == "int8" && tensor_data.data_int8.size() == total_size) {
                    std::copy(tensor_data.data_int8.begin(), tensor_data.data_int8.end(), input_data);
                } else if (tensor_data.dtype == "float32" && tensor_data.data_float.size() == total_size) {
                    // Convert from float32 [0,1] to int8 [-128,127] - DETERMINISTIC
                    for (size_t i = 0; i < total_size; i++) {
                        float val = std::max(0.0f, std::min(1.0f, tensor_data.data_float[i]));
                        input_data[i] = static_cast<int8_t>((val - 0.5f) * 256.0f);
                    }
                } else if (tensor_data.dtype == "int32" && tensor_data.data_int32.size() == total_size) {
                    // Convert from int32 to int8 - DETERMINISTIC
                    for (size_t i = 0; i < total_size; i++) {
                        int32_t val = std::max(-128, std::min(127, tensor_data.data_int32[i]));
                        input_data[i] = static_cast<int8_t>(val);
                    }
                } else {
                    // Fallback: generate consistent int8 data and cache it
                    std::vector<int8_t> generated_data(total_size);
                    std::uniform_int_distribution<int16_t> dis(-128, 127);
                    for (size_t i = 0; i < total_size; i++) {
                        generated_data[i] = static_cast<int8_t>(dis(gen));
                        input_data[i] = generated_data[i];
                    }
                    // Write back to TestData for consistency in future calls
                    tensor_data.data_int8 = generated_data;
                    tensor_data.dtype = "int8";
                }
                break;
            }
            case kTfLiteInt16: {
                int16_t* input_data = interp->typed_input_tensor<int16_t>(input.input_index);
                if (tensor_data.dtype == "int16" && tensor_data.data_int16.size() == total_size) {
                    std::copy(tensor_data.data_int16.begin(), tensor_data.data_int16.end(), input_data);
                } else if (tensor_data.dtype == "float32" && tensor_data.data_float.size() == total_size) {
                    // Convert from float32 [0,1] to int16 [-1000,1000] - DETERMINISTIC
                    for (size_t i = 0; i < total_size; i++) {
                        float val = std::max(0.0f, std::min(1.0f, tensor_data.data_float[i]));
                        input_data[i] = static_cast<int16_t>((val - 0.5f) * 2000.0f);
                    }
                } else if (tensor_data.dtype == "int32" && tensor_data.data_int32.size() == total_size) {
                    // Convert from int32 to int16 - DETERMINISTIC
                    for (size_t i = 0; i < total_size; i++) {
                        int32_t val = std::max(-32768, std::min(32767, tensor_data.data_int32[i]));
                        input_data[i] = static_cast<int16_t>(val);
                    }
                } else {
                    // Fallback: generate consistent int16 data and cache it
                    std::vector<int16_t> generated_data(total_size);
                    std::uniform_int_distribution<int16_t> dis(-1000, 1000);
                    for (size_t i = 0; i < total_size; i++) {
                        generated_data[i] = dis(gen);
                        input_data[i] = generated_data[i];
                    }
                    // Write back to TestData for consistency in future calls
                    tensor_data.data_int16 = generated_data;
                    tensor_data.dtype = "int16";
                }
                break;
            }
            case kTfLiteUInt16: {
                uint16_t* input_data = interp->typed_input_tensor<uint16_t>(input.input_index);
                if (tensor_data.dtype == "uint16" && tensor_data.data_uint16.size() == total_size) {
                    std::copy(tensor_data.data_uint16.begin(), tensor_data.data_uint16.end(), input_data);
                } else if (tensor_data.dtype == "float32" && tensor_data.data_float.size() == total_size) {
                    // Convert from float32 [0,1] to uint16 [0,65535] - DETERMINISTIC
                    for (size_t i = 0; i < total_size; i++) {
                        float val = std::max(0.0f, std::min(1.0f, tensor_data.data_float[i]));
                        input_data[i] = static_cast<uint16_t>(val * 65535.0f);
                    }
                } else if (tensor_data.dtype == "int32" && tensor_data.data_int32.size() == total_size) {
                    // Convert from int32 to uint16 - DETERMINISTIC
                    for (size_t i = 0; i < total_size; i++) {
                        int32_t val = std::max(0, std::min(65535, tensor_data.data_int32[i]));
                        input_data[i] = static_cast<uint16_t>(val);
                    }
                } else {
                    // Fallback: generate consistent uint16 data and cache it
                    std::vector<uint16_t> generated_data(total_size);
                    std::uniform_int_distribution<uint32_t> dis(0, 65535);
                    for (size_t i = 0; i < total_size; i++) {
                        generated_data[i] = static_cast<uint16_t>(dis(gen));
                        input_data[i] = generated_data[i];
                    }
                    // Write back to TestData for consistency in future calls
                    tensor_data.data_uint16 = generated_data;
                    tensor_data.dtype = "uint16";
                }
                break;
            }
            case kTfLiteBool: {
                bool* input_data = interp->typed_input_tensor<bool>(input.input_index);
                if (tensor_data.dtype == "bool" && tensor_data.data_uint8.size() == total_size) {
                    // bool data stored as uint8
                    for (size_t i = 0; i < total_size; i++) {
                        input_data[i] = tensor_data.data_uint8[i] != 0;
                    }
                } else if (tensor_data.dtype == "float32" && tensor_data.data_float.size() == total_size) {
                    // Convert from float32 to bool - DETERMINISTIC
                    for (size_t i = 0; i < total_size; i++) {
                        input_data[i] = tensor_data.data_float[i] > 0.5f;
                    }
                } else if (tensor_data.dtype == "int32" && tensor_data.data_int32.size() == total_size) {
                    // Convert from int32 to bool - DETERMINISTIC
                    for (size_t i = 0; i < total_size; i++) {
                        input_data[i] = tensor_data.data_int32[i] != 0;
                    }
                } else {
                    // Fallback: generate consistent bool data and cache it
                    std::vector<uint8_t> generated_data(total_size);  // bool stored as uint8
                    std::uniform_int_distribution<int> dis(0, 1);
                    for (size_t i = 0; i < total_size; i++) {
                        bool bool_val = dis(gen) == 1;
                        generated_data[i] = bool_val ? 1 : 0;
                        input_data[i] = bool_val;
                    }
                    // Write back to TestData for consistency in future calls
                    tensor_data.data_uint8 = generated_data;
                    tensor_data.dtype = "bool";
                }
                break;
            }
            default:
                // Unsupported type - log warning but don't crash
                std::cout << "Warning: Unsupported input tensor type: " << actual_type << std::endl;
                break;
        }
    }
}

TestData generate_random_int32(const std::vector<int>& shape, 
                                int min_val, int max_val, int input_idx) {
    TestData test_data;
    test_data.input_index = input_idx;
    test_data.tensor_data.shape = shape;
    test_data.tensor_data.dtype = "int32";
    
    size_t total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }
    
    std::uniform_int_distribution<int32_t> dis(min_val, max_val);
    test_data.tensor_data.data_int32.reserve(total_size);
    
    for (size_t i = 0; i < total_size; i++) {
        test_data.tensor_data.data_int32.push_back(dis(gen));
    }
    
    return test_data;
}

TestData generate_random_float32(const std::vector<int>& shape, int input_idx) {
    TestData test_data;
    test_data.input_index = input_idx;
    test_data.tensor_data.shape = shape;
    test_data.tensor_data.dtype = "float32";
    
    size_t total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }
    
    // Use uniform distribution [0, 1) to match Python's np.random.rand()
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    test_data.tensor_data.data_float.reserve(total_size);
    
    for (size_t i = 0; i < total_size; i++) {
        test_data.tensor_data.data_float.push_back(dis(gen));
    }
    
    return test_data;
}

double measure_inference_time(tflite::Interpreter* interp, 
                            std::vector<std::vector<TestData>>& test_inputs) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (auto& test_batch : test_inputs) {
        set_interpreter_inputs(interp, test_batch);
        interp->Invoke();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}

void print_output_shapes(const std::vector<std::vector<TensorData>>& outputs, 
                        const std::string& model_type) {
    if (!outputs.empty() && !outputs[0].empty()) {
        std::cout << "[";
        for (size_t i = 0; i < outputs[0].size(); i++) {
            const auto& shape = outputs[0][i].shape;
            std::cout << "(";
            for (size_t j = 0; j < shape.size(); j++) {
                std::cout << shape[j];
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << ")";
            if (i < outputs[0].size() - 1) std::cout << ", ";
        }
        std::cout << "]";
    }
}

void print_performance_comparison_python_style(double vir_time_cost, double ori_time_cost,
                                            std::optional<double> mem_v_before_create,
                                            std::optional<double> mem_o_before_create,
                                            std::optional<double> mem_v_after_create,
                                            std::optional<double> mem_o_after_create,
                                            std::optional<double> mem_v_before_runs,
                                            std::optional<double> mem_o_before_runs,
                                            std::optional<double> mem_v_after_runs,
                                            std::optional<double> mem_o_after_runs) {
    // Handle zero original time case
    if (ori_time_cost <= 0) {
        std::cout << "The original model’s elapsed time is 0; unable to compute the relative time ratio" << std::endl;
        return;
    }
    
    // Calculate time ratio as percentage
    double time_ratio = vir_time_cost / ori_time_cost * 100.0;
    std::string time_ratio_text = "Time ratio (virtualized/native) = " + std::to_string(time_ratio) + "%";
    
    // Format to 2 decimal places
    std::ostringstream time_stream;
    time_stream << std::fixed << std::setprecision(2) << time_ratio;
    time_ratio_text = "Time ratio (virtualized/native) = " + time_stream.str() + "%";
    
    // Memory ratio calculation
    std::string mem_ratio_text = "";
    try {
        if (mem_v_before_create.has_value() && mem_o_before_create.has_value() && 
            mem_v_after_create.has_value() && mem_o_after_create.has_value() &&
            mem_v_before_runs.has_value() && mem_o_before_runs.has_value() && 
            mem_v_after_runs.has_value() && mem_o_after_runs.has_value()) {
            // Correct calculation: each branch uses its own completion snapshot
            double mem_v_delta = *mem_v_after_create - *mem_v_before_create + *mem_v_after_runs - *mem_v_before_runs;
            double mem_o_delta = *mem_o_after_create - *mem_o_before_create + *mem_o_after_runs - *mem_o_before_runs;
            
            if (mem_o_delta > 0) {
                double mem_ratio = mem_v_delta / mem_o_delta * 100.0;
                std::ostringstream mem_stream;
                mem_stream << std::fixed << std::setprecision(2) << mem_ratio;
                mem_ratio_text = ", Memory usage (relative to baseline) ratio = " + mem_stream.str() + "%";
            }
        }
    } catch (...) {
        // Handle exceptions silently
    }
    
    // Output combined text
    std::cout << time_ratio_text << mem_ratio_text << std::endl;
}

void print_mem_status(const std::string& label, const MemoryMonitor& mem_monitor) {
    mem_monitor.print_mem(label);
}

void print_performance_comparison(double vir_time_cost, double ori_time_cost, 
                                const MemoryMonitor& mem_monitor) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== Performance comparison ===" << std::endl;
    std::cout << "Virtualized model inference time: " << vir_time_cost << " 秒" << std::endl;
    std::cout << "Original model inference time: " << ori_time_cost << " 秒" << std::endl;
    std::cout << "Time ratio (virtualized/native): " << (ori_time_cost > 0 ? vir_time_cost / ori_time_cost : 0.0) << std::endl;
    
    mem_monitor.print_mem("Final memory usage");
}

void resize_interpreter_tensors(tflite::Interpreter* interp,
                                const std::vector<TensorInfo>& input_details,
                                int length_value) {
    for (const auto& detail : input_details) {
        std::vector<int> new_shape = new_shape_for_length(detail, length_value);
        interp->ResizeInputTensor(detail.index, new_shape);
    }
    interp->AllocateTensors();
}
