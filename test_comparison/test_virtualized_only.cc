// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
//
// Test virtualized model only - for isolated memory and performance measurement

#include "comparison_utils.h"
#include "memory_monitor.h"
#include "peak_memory_tracker.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>

void test_virtualized_model_only(const std::string& model_name) {
    std::cout << "=== Virtualized Model Test ===" << std::endl;
    std::cout << "Model: " << model_name << std::endl;

    // Extract input metadata from original model (temporary)
    std::string original_model_path = "tflite_model/" + model_name + ".tflite";
    std::vector<std::vector<int>> input_shapes;
    std::vector<std::string> input_dtypes;
    std::vector<std::vector<int>> input_shape_signatures;

    {
        auto temp_model = tflite::FlatBufferModel::BuildFromFile(original_model_path.c_str());
        if (!temp_model) {
            std::cerr << "Error: Failed to load model for metadata extraction" << std::endl;
            return;
        }

        const tflite::Model* model_ptr = temp_model->GetModel();
        const auto* subgraph = model_ptr->subgraphs()->Get(0);
        const auto* tensors = subgraph->tensors();
        const auto* inputs = subgraph->inputs();

        for (size_t i = 0; i < inputs->size(); i++) {
            int tensor_idx = inputs->Get(i);
            const auto* tensor = tensors->Get(tensor_idx);

            // Extract shape
            std::vector<int> shape;
            if (tensor->shape()) {
                for (size_t j = 0; j < tensor->shape()->size(); j++) {
                    shape.push_back(tensor->shape()->Get(j));
                }
            }
            input_shapes.push_back(shape);

            // Extract dtype
            auto tensor_type = tensor->type();
            std::string dtype_str = "float32";
            if (tensor_type == tflite::TensorType_INT32) dtype_str = "int32";
            else if (tensor_type == tflite::TensorType_UINT8) dtype_str = "uint8";
            else if (tensor_type == tflite::TensorType_INT64) dtype_str = "int64";
            input_dtypes.push_back(dtype_str);

            // Extract shape_signature
            std::vector<int> signature;
            if (tensor->shape_signature() && tensor->shape_signature()->size() > 0) {
                for (size_t j = 0; j < tensor->shape_signature()->size(); j++) {
                    signature.push_back(tensor->shape_signature()->Get(j));
                }
            } else {
                signature = shape;
            }
            input_shape_signatures.push_back(signature);
        }
    }

    // Generate test inputs
    std::vector<std::vector<TestData>> test_inputs;
    for (int i = 0; i < 10; i++) {
        std::vector<TestData> test_batch;
        for (size_t j = 0; j < input_shapes.size(); j++) {
            if (input_dtypes[j] == "int32") {
                test_batch.push_back(generate_random_int32(input_shapes[j], 0, 100, j));
            } else {
                test_batch.push_back(generate_random_float32(input_shapes[j], j));
            }
        }
        test_inputs.push_back(test_batch);
    }

    // Start peak memory tracking
    PeakMemoryTracker tracker(5);  // 5ms sampling
    tracker.start();
    
    // Start timing for model build
    auto build_start = std::chrono::high_resolution_clock::now();

    // Load virtualized model
    std::string v_infos_path = model_name + "_v_infos.json";
    std::string params_path = model_name + "_params.bin";

    auto v_model = tflite::FlatBufferModel::BuildFromVirtualizedFiles(
        v_infos_path.c_str(), params_path.c_str());
    if (!v_model) {
        std::cerr << "Error: Failed to load virtualized model" << std::endl;
        return;
    }

    auto v_interpreter = create_interpreter_from_model(v_model.get());
    if (!v_interpreter) {
        std::cerr << "Error: Failed to create interpreter" << std::endl;
        return;
    }

    v_interpreter->AllocateTensors();
    auto v_output_details = get_output_details(v_interpreter.get());

    auto build_end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(build_end - build_start).count();

    // Start timing for inference
    auto inference_start = std::chrono::high_resolution_clock::now();

    // Run inference
    std::vector<std::vector<TensorData>> outputs;
    for (size_t i = 0; i < test_inputs.size(); i++) {
        set_interpreter_inputs(v_interpreter.get(), test_inputs[i]);
        v_interpreter->Invoke();
        outputs.push_back(collect_outputs(v_interpreter.get(), v_output_details));
    }

    auto inference_end = std::chrono::high_resolution_clock::now();
    double inference_time = std::chrono::duration<double>(inference_end - inference_start).count();

    // Stop tracking
    tracker.stop();

    // Output results
    std::cout << "\nBaseline: " << tracker.get_baseline_mb() << " MB" << std::endl;
    std::cout << "Peak: " << tracker.get_peak_mb() << " MB" << std::endl;
    std::cout << "Overhead: " << tracker.get_overhead_mb() << " MB" << std::endl;
    std::cout << "Model Build Time: " << build_time << " s" << std::endl;
    std::cout << "Inference Time (10 iterations): " << inference_time << " s" << std::endl;
    std::cout << "Total Time: " << build_time + inference_time << " s" << std::endl;

    std::cout << "\n=== Test Completed ===" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Virtualized Model Performance Test ===" << std::endl;
    std::cout << std::endl;

    std::string model_name = "squeezenet";
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--model_name=", 13) == 0) {
            model_name = argv[i] + 13;
        }
    }

    try {
        test_virtualized_model_only(model_name);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
