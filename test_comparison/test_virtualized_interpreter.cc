// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
#include "comparison_utils.h"
#include "memory_monitor.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>


// Main test function
void test_virtualized_interpreter(const std::string& model_name) {
    // === Start Virtualized TFLite Interpreter Testing ===
    // Note: gc_collect() removed - C++ uses RAII for memory management
    std::cout << "=== Start virtualized TFLite interpreter testing ===" << std::endl;

    // Memory monitoring initialization (corresponds to psutil section)
    MemoryMonitor mem_monitor;

    // Auto-assemble file paths based on model_name
    std::string v_infos_path = model_name + "_v_infos.json";
    std::string params_path = model_name + "_params.bin";
    std::string original_model_path = "tflite_model/" + model_name + ".tflite";

    std::cout << "Virtualized model files: " << v_infos_path << ", " << params_path << std::endl;
    std::cout << "Original model file: " << original_model_path << std::endl;

    // === Stage 0: Temporary metadata extraction (not counted in memory baseline) ===
    // This stage temporarily loads the original .tflite model to extract input metadata
    // The temporary model and interpreter are destroyed before Stage 1 begins
    std::cout << "\n=== Stage 0: Extract input metadata from original model ===" << std::endl;

    std::vector<std::vector<int>> input_shapes;
    std::vector<std::string> input_dtypes;
    std::vector<std::vector<int>> input_shape_signatures;

    {
        // Scoped block for temporary model loading - objects auto-destroyed at scope exit
        auto temp_model = tflite::FlatBufferModel::BuildFromFile(original_model_path.c_str());
        if (!temp_model) {
            std::cerr << "Error: Failed to load temporary model for metadata extraction: " << original_model_path << std::endl;
            return;
        }

        // Get FlatBuffer model pointer for metadata extraction
        const tflite::Model* model_ptr = temp_model->GetModel();
        const auto* subgraphs = model_ptr->subgraphs();
        if (!subgraphs || subgraphs->size() == 0) {
            std::cerr << "Error: Model has no subgraphs" << std::endl;
            return;
        }

        const auto* subgraph = subgraphs->Get(0);
        const auto* tensors = subgraph->tensors();
        const auto* inputs = subgraph->inputs();

        // Extract input_shapes, input_dtypes, input_shape_signatures
        for (size_t i = 0; i < inputs->size(); i++) {
            int tensor_idx = inputs->Get(i);
            const auto* tensor = tensors->Get(tensor_idx);

            // Extract shape
            const auto* shape_fb = tensor->shape();
            std::vector<int> shape;
            if (shape_fb) {
                for (size_t j = 0; j < shape_fb->size(); j++) {
                    shape.push_back(shape_fb->Get(j));
                }
            }
            input_shapes.push_back(shape);

            // Extract dtype
            auto tensor_type = tensor->type();
            std::string dtype_str;
            if (tensor_type == tflite::TensorType_FLOAT32) {
                dtype_str = "float32";
            } else if (tensor_type == tflite::TensorType_INT32) {
                dtype_str = "int32";
            } else if (tensor_type == tflite::TensorType_UINT8) {
                dtype_str = "uint8";
            } else if (tensor_type == tflite::TensorType_INT64) {
                dtype_str = "int64";
            } else {
                dtype_str = "float32";  // Default fallback
            }
            input_dtypes.push_back(dtype_str);

            // Extract real shape_signature (for dynamic models)
            const auto* sig = tensor->shape_signature();
            std::vector<int> signature;
            if (sig && sig->size() > 0) {
                // Real shape_signature exists (may contain -1 for dynamic dimensions)
                for (size_t j = 0; j < sig->size(); j++) {
                    signature.push_back(sig->Get(j));
                }
            } else {
                // No shape_signature, fallback to shape
                signature = shape;
            }
            input_shape_signatures.push_back(signature);
        }

        // temp_model and all related objects are automatically destroyed here
    }

    std::cout << "Extracted input metadata:" << std::endl;
    std::cout << "  Number of inputs: " << input_shapes.size() << std::endl;
    std::cout << "  Input shapes: ";
    for (size_t i = 0; i < input_shapes.size(); i++) {
        std::cout << "[";
        for (size_t j = 0; j < input_shapes[i].size(); j++) {
            std::cout << input_shapes[i][j];
            if (j < input_shapes[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]";
        if (i < input_shapes.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "  Input dtypes: ";
    for (size_t i = 0; i < input_dtypes.size(); i++) {
        std::cout << input_dtypes[i];
        if (i < input_dtypes.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "  Input shape_signatures: ";
    for (size_t i = 0; i < input_shape_signatures.size(); i++) {
        std::cout << "[";
        for (size_t j = 0; j < input_shape_signatures[i].size(); j++) {
            std::cout << input_shape_signatures[i][j];
            if (j < input_shape_signatures[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]";
        if (i < input_shape_signatures.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    // === Stage 1-4: Original testing logic (memory baseline starts here) ===

    // Test data generation
    std::cout << "\n=== Generate test data ===" << std::endl;
    std::vector<std::vector<TestData>> test_inputs;
    for (int i = 0; i < 10; i++) {
        std::vector<TestData> test_batch;
        for (size_t j = 0; j < input_shapes.size(); j++) {
            if (input_dtypes[j] == "int32") {
                auto test_data = generate_random_int32(input_shapes[j], 0, 100, j);
                test_batch.push_back(test_data);
            } else {  // float32
                auto test_data = generate_random_float32(input_shapes[j], j);
                test_batch.push_back(test_data);
            }
        }
        test_inputs.push_back(test_batch);
    }
    std::cout << "Generated " << test_inputs.size() << " sets of test data" << std::endl;
    
    // Virtualized Model Loading
    std::cout << "\n=== Load the virtualized model ===" << std::endl;

    // Create virtualized model using custom API with parameterized file paths
    auto v_model = tflite::FlatBufferModel::BuildFromVirtualizedFiles(
        v_infos_path.c_str(), params_path.c_str());
    if (!v_model) {
        std::cerr << "Error: Failed to load the virtualized model file: " << v_infos_path << ", " << params_path << std::endl;
        return;
    }

    auto v_interpreter = create_interpreter_from_model(v_model.get());
    if (!v_interpreter) {
        std::cerr << "Error: Failed to create the virtualized model interpreter" << std::endl;
        return;
    }

    v_interpreter->AllocateTensors();
    auto v_input_details = get_input_details(v_interpreter.get());
    auto v_output_details = get_output_details(v_interpreter.get());

    std::cout << "Virtualized model loaded successfully" << std::endl;
    std::cout << "Number of inputs: " << v_input_details.size() << std::endl;
    std::cout << "Number of outputs: " << v_output_details.size() << std::endl;

    // Original Model Loading
    std::cout << "\n=== Load the original model ===" << std::endl;

    // Create original model using standard API (reload from original_model_path)
    // Note: This is a separate load, not reusing the Stage 0 temporary model
    auto o_model = tflite::FlatBufferModel::BuildFromFile(original_model_path.c_str());
    if (!o_model) {
        std::cerr << "Error: Failed to load the original model file: " << original_model_path << std::endl;
        return;
    }
    
    auto o_interpreter = create_interpreter_from_model(o_model.get());
    if (!o_interpreter) {
        std::cerr << "Error: Failed to create the original model interpreter" << std::endl;
        return;
    }
    
    o_interpreter->AllocateTensors();
    auto input_details = get_input_details(o_interpreter.get());
    auto output_details = get_output_details(o_interpreter.get());

    std::cout << "Original model loaded successfully" << std::endl;
    std::cout << "Number of inputs: " << input_details.size() << std::endl;
    std::cout << "Number of outputs: " << output_details.size() << std::endl;

    // Static/Dynamic Path Branching
    if (has_dynamic_signature(v_input_details)) {
        std::cout << "\n=== Detected dynamic shape input, starting dynamic test ===" << std::endl;
        
        // Dynamic path
        std::vector<int> test_lengths;
        
        // Length selection logic
        // has_int32 = any((d.get('dtype') == np.int32) for d in v_input_details)
        // length_set = [1, 8, 16, 32, 64] if has_int32 else [32, 64, 128]
        bool has_int32 = has_int32_input(v_input_details);
        if (has_int32) {
            test_lengths = {1, 8, 16, 32, 64};
        } else {
            test_lengths = {32, 64, 128};
        }
        
        std::cout << "Selected test length: [";
        for (size_t i = 0; i < test_lengths.size(); i++) {
            std::cout << test_lengths[i];
            if (i < test_lengths.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::vector<std::vector<TensorData>> outputs_vir_all;
        std::vector<std::vector<TensorData>> outputs_ori_all;
        
        for (int L : test_lengths) {
            std::cout << "\n--- Dynamic length/size = " << L << " ---" << std::endl;
            
            // Resize interpreters for current length
            resize_interpreter_tensors(v_interpreter.get(), v_input_details, L);
            resize_interpreter_tensors(o_interpreter.get(), input_details, L);
            
            // Generate test inputs for current length
            std::vector<std::vector<TestData>> test_inputs_L;
            for (int i = 0; i < 10; i++) {
                std::vector<TestData> test_batch;
                for (size_t j = 0; j < v_input_details.size(); j++) {
                    auto new_shape = new_shape_for_length(v_input_details[j], L);
                    if (input_dtypes[j] == "int32") {
                        auto test_data = generate_random_int32(new_shape, 0, 100, j);
                        test_batch.push_back(test_data);
                    } else {
                        auto test_data = generate_random_float32(new_shape, j);
                        test_batch.push_back(test_data);
                    }
                }
                test_inputs_L.push_back(test_batch);
            }

            // Run virtualized model inference
            std::cout << "Start executing virtualized model inference..." << std::endl;

            std::vector<std::vector<TensorData>> outputs_vir_L;
            for (int i = 0; i < 10; i++) {
                set_interpreter_inputs(v_interpreter.get(), test_inputs_L[i]);
                v_interpreter->Invoke();
                outputs_vir_L.push_back(collect_outputs(v_interpreter.get(), v_output_details));
            }

            // Print virtualized model inference completion info for dynamic test
            std::cout << "Virtualized model inference completed (10 runs), output shape of the last run: ";
            if (!outputs_vir_L.empty()) {
                std::cout << "[";
                for (size_t i = 0; i < outputs_vir_L.back().size(); i++) {
                    const auto& shape = outputs_vir_L.back()[i].shape;
                    std::cout << "(";
                    for (size_t j = 0; j < shape.size(); j++) {
                        std::cout << shape[j];
                        if (j < shape.size() - 1) std::cout << ", ";
                    }
                    std::cout << ")";
                    if (i < outputs_vir_L.back().size() - 1) std::cout << ", ";
                }
                std::cout << "]";
            }
            std::cout << std::endl;
            
            // Run original model inference
            std::cout << "Start executing original model inference..." << std::endl;
            std::vector<std::vector<TensorData>> outputs_ori_L;
            for (int i = 0; i < 10; i++) {
                set_interpreter_inputs(o_interpreter.get(), test_inputs_L[i]);
                o_interpreter->Invoke();
                outputs_ori_L.push_back(collect_outputs(o_interpreter.get(), output_details));
            }

            // Print original model inference completion info for dynamic test
            std::cout << "Original model inference completed (10 runs), output shape of the last run: ";
            if (!outputs_ori_L.empty()) {
                std::cout << "[";
                for (size_t i = 0; i < outputs_ori_L.back().size(); i++) {
                    const auto& shape = outputs_ori_L.back()[i].shape;
                    std::cout << "(";
                    for (size_t j = 0; j < shape.size(); j++) {
                        std::cout << shape[j];
                        if (j < shape.size() - 1) std::cout << ", ";
                    }
                    std::cout << ")";
                    if (i < outputs_ori_L.back().size() - 1) std::cout << ", ";
                }
                std::cout << "]";
            }
            std::cout << std::endl;

            // Store results for overall analysis
            outputs_vir_all.insert(outputs_vir_all.end(), outputs_vir_L.begin(), outputs_vir_L.end());
            outputs_ori_all.insert(outputs_ori_all.end(), outputs_ori_L.begin(), outputs_ori_L.end());

            // Metrics calculation for this length
            compute_metrics(outputs_vir_L, outputs_ori_L);
        }
        
        // Overall results for all lengths
        std::cout << "\n=== Overall results for all dynamic lengths ===" << std::endl;
        compute_metrics(outputs_vir_all, outputs_ori_all);
        // print_outputs_meta(v_output_details, output_details);
        
    } else {
        std::cout << "\n=== Detected static-shape input; starting static testing ===" << std::endl;
        
        // Static path
        std::cout << "Start executing virtualized model inference..." << std::endl;

        // Run 10 iterations for virtualized model
        std::vector<std::vector<TensorData>> outputs_vir;
        for (int i = 0; i < 10; i++) {
            set_interpreter_inputs(v_interpreter.get(), test_inputs[i]);
            v_interpreter->Invoke();
            outputs_vir.push_back(collect_outputs(v_interpreter.get(), v_output_details));
        }

        // Print virtualized model inference completion info
        std::cout << "Virtualized model inference completed (10 runs), output shape: ";
        print_output_shapes(outputs_vir);
        std::cout << std::endl;

        std::cout << "Start executing original model inference..." << std::endl;

        // Run 10 iterations for original model
        std::vector<std::vector<TensorData>> outputs_ori;
        for (int i = 0; i < 10; i++) {
            set_interpreter_inputs(o_interpreter.get(), test_inputs[i]);
            o_interpreter->Invoke();
            outputs_ori.push_back(collect_outputs(o_interpreter.get(), output_details));
        }

        // Print original model inference completion info
        std::cout << "Original model inference completed (10 runs), output shape: ";
        print_output_shapes(outputs_ori);
        std::cout << std::endl;

        // Metrics calculation
        compute_metrics(outputs_vir, outputs_ori);
    }
    
    std::cout << "Test completed!" << std::endl;
    std::cout << "\n=== Virtualized TFLite interpreter testing completed ===" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== C++ TensorFlow Lite virtualized model comparison test ===" << std::endl;
    std::cout << std::endl;

    // Command-line argument parsing for model_name
    std::string model_name = "squeezenet";  // Default value
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--model_name=", 13) == 0) {
            model_name = argv[i] + 13;
        }
    }

    std::cout << "Model name: " << model_name << std::endl;
    std::cout << std::endl;

    try {
        test_virtualized_interpreter(model_name);
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during the test: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown exception occurred during the test" << std::endl;
        return 1;
    }

    return 0;
}
