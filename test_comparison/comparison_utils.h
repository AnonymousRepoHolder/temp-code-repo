#ifndef COMPARISON_UTILS_H
#define COMPARISON_UTILS_H

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <cstdint>
#include <optional>
#include "memory_monitor.h"

// TensorFlow Lite includes
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// Tensor Information Structure
struct TensorInfo {
    int index;
    std::string name;
    std::vector<int> shape;
    std::vector<int> shape_signature;
    std::string dtype;
    float quantization_scale = 0.0f;
    int32_t quantization_zero_point = 0;
};

// Tensor Data Structure
// Holds tensor data with shape and type information
struct TensorData {
    std::vector<float> data_float;
    std::vector<int32_t> data_int32;
    std::vector<uint8_t> data_uint8;
    std::vector<int8_t> data_int8;
    std::vector<int16_t> data_int16;
    std::vector<uint16_t> data_uint16;
    std::vector<int> shape;
    std::string dtype;
    
    size_t size() const {
        if (dtype == "float32") {
            return data_float.size();
        } else if (dtype == "int32") {
            return data_int32.size();
        } else if (dtype == "uint8") {
            return data_uint8.size();
        } else if (dtype == "int8") {
            return data_int8.size();
        } else if (dtype == "int16") {
            return data_int16.size();
        } else if (dtype == "uint16") {
            return data_uint16.size();
        } else if (dtype == "bool") {
            return data_uint8.size();  // Boolean data is stored in data_uint8
        }
        return 0;
    }
    
    TensorData reshape(const std::vector<int>& new_shape) const {
        TensorData result = *this;
        result.shape = new_shape;
        return result;
    }
    
    TensorData reshape(int flatten) const {
        if (flatten == -1) {
            // Flatten to 1D
            size_t total_size = 1;
            for (int dim : shape) {
                total_size *= dim;
            }
            TensorData result = *this;
            result.shape = {static_cast<int>(total_size)};
            return result;
        }
        return *this;
    }
};

// Test Input Data Structure
// Holds test input data for model inference
struct TestData {
    TensorData tensor_data;
    int input_index;
};

// Utility Functions
// Check if any input has int32 dtype (for dynamic length selection)
bool has_int32_input(const std::vector<TensorInfo>& input_details);

// Check if input details have dynamic signature
bool has_dynamic_signature(const std::vector<TensorInfo>& input_details);

// Collect outputs from interpreter
std::vector<TensorData> collect_outputs(tflite::Interpreter* interp, 
                                        const std::vector<TensorInfo>& out_details);

// Convert tensor data to float32 safely
std::vector<float> safe_to_float32(const TensorData& x);

// Calculate average of float vector
float avg(const std::vector<float>& lst);

// Compute error metrics between virtualized and original outputs
void compute_metrics(const std::vector<std::vector<TensorData>>& outputs_vir,
                    const std::vector<std::vector<TensorData>>& outputs_ori);

// Print outputs metadata comparison
void print_outputs_meta(const std::vector<TensorInfo>& v_out_details,
                        const std::vector<TensorInfo>& o_out_details);

// Create new shape for dynamic length testing
std::vector<int> new_shape_for_length(const TensorInfo& detail, int length_value);

// Get input tensor details from interpreter
// Helper function to extract input tensor information
std::vector<TensorInfo> get_input_details(tflite::Interpreter* interp);

// Get output tensor details from interpreter
// Helper function to extract output tensor information
std::vector<TensorInfo> get_output_details(tflite::Interpreter* interp);

// Create interpreter from model
// Helper function to create and configure interpreter
std::unique_ptr<tflite::Interpreter> create_interpreter_from_model(
    const tflite::FlatBufferModel* model);

// Set interpreter inputs
// Helper function to set input data for inference
void set_interpreter_inputs(tflite::Interpreter* interp, 
                            std::vector<TestData>& inputs);

// Generate random int32 test data
// Helper function to generate test inputs
TestData generate_random_int32(const std::vector<int>& shape, 
                                int min_val = 0, int max_val = 100, 
                                int input_idx = 0);

// Generate random float32 test data  
// Helper function to generate test inputs
TestData generate_random_float32(const std::vector<int>& shape, int input_idx = 0);

// Measure inference time for multiple iterations
// Helper function to measure performance
double measure_inference_time(tflite::Interpreter* interp, 
                            std::vector<std::vector<TestData>>& test_inputs);

// Print performance comparison results in Python style
// Helper function to display timing and memory results with percentage format
void print_performance_comparison_python_style(double vir_time_cost, double ori_time_cost,
                                            std::optional<double> mem_v_before_create,
                                            std::optional<double> mem_o_before_create,
                                            std::optional<double> mem_v_after_create,
                                            std::optional<double> mem_o_after_create,
                                            std::optional<double> mem_v_before_runs,
                                            std::optional<double> mem_o_before_runs,
                                            std::optional<double> mem_v_after_runs,
                                            std::optional<double> mem_o_after_runs);

// Print memory status at specific points
// Helper function to display memory information
void print_mem_status(const std::string& label, const MemoryMonitor& mem_monitor);

// Print output shapes from tensor data list
// Helper function to display output shape information
void print_output_shapes(const std::vector<std::vector<TensorData>>& outputs, 
                        const std::string& model_type = "模型");

// Resize interpreter tensors for dynamic shape testing
// Helper function to handle dynamic tensor resizing
void resize_interpreter_tensors(tflite::Interpreter* interp,
                                const std::vector<TensorInfo>& input_details,
                                int length_value);

#endif // COMPARISON_UTILS_H
