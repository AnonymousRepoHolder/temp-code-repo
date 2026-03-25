// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
#include "experiment_utils.h"
#include "peak_memory_tracker.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace {

const std::vector<std::string>& paper11_models() {
    static const std::vector<std::string> models = {
        "squeezenet",
        "posenet",
        "fruit",
        "lenet",
        "mobilenet",
        "skin",
        "mnasnet",
        "efficientnet",
        "ssd",
        "depth_estimation",
        "distilgpt2-official",
    };
    return models;
}

const std::vector<std::string>& comparison10_models() {
    static const std::vector<std::string> models = {
        "squeezenet",
        "posenet",
        "fruit",
        "lenet",
        "mobilenet",
        "skin",
        "mnasnet",
        "efficientnet",
        "ssd",
        "depth_estimation",
    };
    return models;
}

const std::unordered_set<std::string>& classification_models() {
    static const std::unordered_set<std::string> models = {
        "squeezenet",
        "fruit",
        "lenet",
        "mobilenet",
        "skin",
        "mnasnet",
        "efficientnet",
    };
    return models;
}

std::string original_model_path(const std::string& model_name) {
    return "tflite_model/" + model_name + ".tflite";
}

std::string virtualized_infos_path(const std::string& model_name) {
    return model_name + "_v_infos.json";
}

std::string virtualized_params_path(const std::string& model_name) {
    return model_name + "_params.bin";
}

std::string tensor_type_to_dtype(tflite::TensorType tensor_type) {
    if (tensor_type == tflite::TensorType_FLOAT32) {
        return "float32";
    }
    if (tensor_type == tflite::TensorType_INT32) {
        return "int32";
    }
    if (tensor_type == tflite::TensorType_UINT8) {
        return "uint8";
    }
    if (tensor_type == tflite::TensorType_INT64) {
        return "int64";
    }
    return "float32";
}

void validate_model_file_exists(const std::string& model_name) {
    if (!std::filesystem::exists(original_model_path(model_name))) {
        throw std::runtime_error("Model file not found: " + original_model_path(model_name));
    }
}

std::vector<TestData> generate_test_input_batch(
        const std::vector<std::vector<int>>& input_shapes,
        const std::vector<std::string>& input_dtypes) {
    std::vector<TestData> test_batch;
    test_batch.reserve(input_shapes.size());
    for (size_t j = 0; j < input_shapes.size(); j++) {
        if (input_dtypes[j] == "int32") {
            test_batch.push_back(generate_random_int32(input_shapes[j], 0, 100, static_cast<int>(j)));
        } else {
            test_batch.push_back(generate_random_float32(input_shapes[j], static_cast<int>(j)));
        }
    }
    return test_batch;
}

std::vector<std::vector<TestData>> generate_test_inputs(
        const std::vector<std::vector<int>>& input_shapes,
        const std::vector<std::string>& input_dtypes,
        int num_inputs) {
    std::vector<std::vector<TestData>> test_inputs;
    test_inputs.reserve(std::max(num_inputs, 0));
    for (int i = 0; i < num_inputs; i++) {
        test_inputs.push_back(generate_test_input_batch(input_shapes, input_dtypes));
    }
    return test_inputs;
}

std::vector<TestData> generate_dynamic_test_input_batch(
        const std::vector<TensorInfo>& input_details,
        const std::vector<std::string>& input_dtypes,
        int length_value) {
    std::vector<std::vector<int>> reshaped_inputs;
    reshaped_inputs.reserve(input_details.size());
    for (const auto& detail : input_details) {
        reshaped_inputs.push_back(new_shape_for_length(detail, length_value));
    }
    return generate_test_input_batch(reshaped_inputs, input_dtypes);
}

void print_metadata(const ModelMetadata& metadata) {
    std::cout << "Extracted input metadata:" << std::endl;
    std::cout << "  Number of inputs: " << metadata.input_shapes.size() << std::endl;
    std::cout << "  Input shapes: ";
    for (size_t i = 0; i < metadata.input_shapes.size(); i++) {
        std::cout << "[";
        for (size_t j = 0; j < metadata.input_shapes[i].size(); j++) {
            std::cout << metadata.input_shapes[i][j];
            if (j + 1 < metadata.input_shapes[i].size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i + 1 < metadata.input_shapes.size()) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    std::cout << "  Input dtypes: ";
    for (size_t i = 0; i < metadata.input_dtypes.size(); i++) {
        std::cout << metadata.input_dtypes[i];
        if (i + 1 < metadata.input_dtypes.size()) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    std::cout << "  Input shape_signatures: ";
    for (size_t i = 0; i < metadata.input_shape_signatures.size(); i++) {
        std::cout << "[";
        for (size_t j = 0; j < metadata.input_shape_signatures[i].size(); j++) {
            std::cout << metadata.input_shape_signatures[i][j];
            if (j + 1 < metadata.input_shape_signatures[i].size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i + 1 < metadata.input_shape_signatures.size()) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

std::vector<int> default_dynamic_lengths(const std::vector<TensorInfo>& input_details) {
    if (has_int32_input(input_details)) {
        return {1, 8, 16, 32, 64};
    }
    return {32, 64, 128};
}

std::vector<int> distribute_evenly(int total, size_t buckets) {
    std::vector<int> distribution(buckets, 0);
    if (buckets == 0 || total <= 0) {
        return distribution;
    }
    int base = total / static_cast<int>(buckets);
    int remainder = total % static_cast<int>(buckets);
    for (size_t i = 0; i < buckets; i++) {
        distribution[i] = base + (static_cast<int>(i) < remainder ? 1 : 0);
    }
    return distribution;
}

void merge_metrics_accumulator(MetricsAccumulator& target,
                               const MetricsAccumulator& source) {
    target.num_tests += source.num_tests;
    if (target.outputs.size() < source.outputs.size()) {
        size_t original_size = target.outputs.size();
        target.outputs.resize(source.outputs.size());
        for (size_t i = original_size; i < source.outputs.size(); i++) {
            target.outputs[i].output_index = source.outputs[i].output_index;
        }
    }

    for (size_t i = 0; i < source.outputs.size(); i++) {
        target.outputs[i].mse_sum += source.outputs[i].mse_sum;
        target.outputs[i].mae_sum += source.outputs[i].mae_sum;
        target.outputs[i].maxae_sum += source.outputs[i].maxae_sum;
        target.outputs[i].relmae_sum += source.outputs[i].relmae_sum;
        target.outputs[i].metric_count += source.outputs[i].metric_count;
        target.outputs[i].top1_sum += source.outputs[i].top1_sum;
        target.outputs[i].top1_count += source.outputs[i].top1_count;
    }
}

void print_last_output_shapes(const std::vector<TensorData>& outputs,
                              const std::string& label) {
    std::cout << label;
    if (!outputs.empty()) {
        std::cout << "[";
        for (size_t i = 0; i < outputs.size(); i++) {
            const auto& shape = outputs[i].shape;
            std::cout << "(";
            for (size_t j = 0; j < shape.size(); j++) {
                std::cout << shape[j];
                if (j + 1 < shape.size()) {
                    std::cout << ", ";
                }
            }
            std::cout << ")";
            if (i + 1 < outputs.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
    }
    std::cout << std::endl;
}

CorrectnessAggregate build_correctness_aggregate(const MetricsSummary& summary) {
    CorrectnessAggregate aggregate;
    for (const auto& output : summary.outputs) {
        aggregate.mse = std::max(aggregate.mse, output.mse);
        aggregate.mae = std::max(aggregate.mae, output.mae);
        aggregate.maxae = std::max(aggregate.maxae, output.maxae);
        aggregate.relmae = std::max(aggregate.relmae, output.relmae);
        if (output.has_top1) {
            aggregate.has_top1_difference = true;
            aggregate.top1_difference = std::max(
                aggregate.top1_difference,
                1.0 - output.top1_agreement);
        }
    }
    return aggregate;
}

std::string trim(const std::string& input) {
    const std::string whitespace = " \t\n\r";
    const size_t start = input.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        return "";
    }
    const size_t end = input.find_last_not_of(whitespace);
    return input.substr(start, end - start + 1);
}

std::vector<std::string> split_csv(const std::string& csv) {
    std::vector<std::string> items;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            items.push_back(token);
        }
    }
    return items;
}

std::vector<std::string> all_tflite_models() {
    std::vector<std::string> models;
    if (!std::filesystem::exists("tflite_model")) {
        throw std::runtime_error("Directory not found: tflite_model");
    }
    for (const auto& entry : std::filesystem::directory_iterator("tflite_model")) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() != ".tflite") {
            continue;
        }
        models.push_back(entry.path().stem().string());
    }
    std::sort(models.begin(), models.end());
    return models;
}

std::string format_decimal(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string format_metric_value(double value) {
    if (value == 0.0) {
        return "0.0";
    }
    std::ostringstream oss;
    double abs_value = std::abs(value);
    if (abs_value < 1e-6 || abs_value >= 1e6) {
        oss << std::scientific << std::setprecision(3) << value;
    } else if (abs_value < 1e-3) {
        oss << std::fixed << std::setprecision(10) << value;
    } else {
        oss << std::fixed << std::setprecision(8) << value;
    }
    return oss.str();
}

std::string ratio_to_percent(double ratio) {
    return format_decimal(ratio * 100.0, 1) + "%";
}

std::string escape_json(const std::string& input) {
    std::ostringstream oss;
    for (char ch : input) {
        switch (ch) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << ch;
                break;
        }
    }
    return oss.str();
}

std::string quote_json(const std::string& input) {
    return "\"" + escape_json(input) + "\"";
}

std::string json_bool(bool value) {
    return value ? "true" : "false";
}

std::string ints_to_json(const std::vector<int>& values) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); i++) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

std::string metrics_summary_to_json(const MetricsSummary& summary) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"num_tests\":" << summary.num_tests << ",";
    oss << "\"outputs\":[";
    for (size_t i = 0; i < summary.outputs.size(); i++) {
        const auto& output = summary.outputs[i];
        if (i > 0) {
            oss << ",";
        }
        oss << "{"
            << "\"output_index\":" << output.output_index << ","
            << "\"mse\":" << format_decimal(output.mse, 15) << ","
            << "\"mae\":" << format_decimal(output.mae, 12) << ","
            << "\"maxae\":" << format_decimal(output.maxae, 12) << ","
            << "\"relmae\":" << format_decimal(output.relmae, 12) << ","
            << "\"has_top1\":" << json_bool(output.has_top1) << ","
            << "\"top1_agreement\":" << format_decimal(output.top1_agreement, 12)
            << "}";
    }
    oss << "]";
    oss << "}";
    return oss.str();
}

std::string correctness_result_to_json(const CorrectnessResult& result) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"model_name\":" << quote_json(result.model_name) << ",";
    oss << "\"seed\":" << result.seed << ",";
    oss << "\"num_inputs\":" << result.num_inputs << ",";
    oss << "\"is_dynamic\":" << json_bool(result.is_dynamic) << ",";
    oss << "\"dynamic_lengths\":" << ints_to_json(result.dynamic_lengths) << ",";
    oss << "\"length_allocations\":" << ints_to_json(result.length_allocations) << ",";
    oss << "\"aggregate\":{"
        << "\"mse\":" << format_decimal(result.aggregate.mse, 15) << ","
        << "\"mae\":" << format_decimal(result.aggregate.mae, 12) << ","
        << "\"maxae\":" << format_decimal(result.aggregate.maxae, 12) << ","
        << "\"relmae\":" << format_decimal(result.aggregate.relmae, 12) << ","
        << "\"has_top1_difference\":" << json_bool(result.aggregate.has_top1_difference) << ","
        << "\"top1_difference\":" << format_decimal(result.aggregate.top1_difference, 12)
        << "},";
    oss << "\"metrics\":" << metrics_summary_to_json(result.metrics);
    oss << "}";
    return oss.str();
}

std::string performance_result_to_json(const PerformanceResult& result) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"model_name\":" << quote_json(result.model_name) << ",";
    oss << "\"seed\":" << result.seed << ",";
    oss << "\"num_iters\":" << result.num_iters << ",";
    oss << "\"memory_iters\":" << result.memory_iters << ",";
    oss << "\"baseline_mb\":" << format_decimal(result.baseline_mb, 6) << ",";
    oss << "\"peak_mb\":" << format_decimal(result.peak_mb, 6) << ",";
    oss << "\"overhead_mb\":" << format_decimal(result.overhead_mb, 6) << ",";
    oss << "\"build_time_sec\":" << format_decimal(result.build_time_sec, 9) << ",";
    oss << "\"inference_time_sec\":" << format_decimal(result.inference_time_sec, 9) << ",";
    oss << "\"total_time_sec\":" << format_decimal(result.total_time_sec, 9);
    oss << "}";
    return oss.str();
}

std::string model_suite_result_to_json(const ModelSuiteResult& result) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"model_name\":" << quote_json(result.model_name) << ",";
    oss << "\"has_correctness\":" << json_bool(result.has_correctness) << ",";
    oss << "\"has_performance\":" << json_bool(result.has_performance);
    if (result.has_correctness) {
        oss << ",\"correctness\":" << correctness_result_to_json(result.correctness);
    }
    if (result.has_performance) {
        oss << ",\"virtualized\":" << performance_result_to_json(result.virtualized);
        oss << ",\"original\":" << performance_result_to_json(result.original);
        oss << ",\"ratios\":{"
            << "\"build_ratio\":" << format_decimal(result.build_ratio, 9) << ","
            << "\"inference_ratio\":" << format_decimal(result.inference_ratio, 9) << ","
            << "\"memory_ratio\":" << format_decimal(result.memory_ratio, 9)
            << "}";
    }
    oss << "}";
    return oss.str();
}

std::string suite_selection_to_string(SuiteSelection selection) {
    switch (selection) {
        case SuiteSelection::RQ1:
            return "rq1";
        case SuiteSelection::RQ2:
            return "rq2";
        case SuiteSelection::All:
        default:
            return "all";
    }
}

}  // namespace

ModelMetadata load_model_metadata(const std::string& model_name) {
    validate_model_file_exists(model_name);
    ModelMetadata metadata;

    auto model = tflite::FlatBufferModel::BuildFromFile(original_model_path(model_name).c_str());
    if (!model) {
        throw std::runtime_error("Failed to load model for metadata extraction: " +
                                 original_model_path(model_name));
    }

    const tflite::Model* model_ptr = model->GetModel();
    const auto* subgraphs = model_ptr->subgraphs();
    if (!subgraphs || subgraphs->size() == 0) {
        throw std::runtime_error("Model has no subgraphs: " + model_name);
    }
    const auto* subgraph = subgraphs->Get(0);
    const auto* tensors = subgraph->tensors();
    const auto* inputs = subgraph->inputs();
    if (!tensors || !inputs) {
        throw std::runtime_error("Model has incomplete tensor metadata: " + model_name);
    }

    for (size_t i = 0; i < inputs->size(); i++) {
        int tensor_idx = inputs->Get(i);
        const auto* tensor = tensors->Get(tensor_idx);
        if (!tensor) {
            throw std::runtime_error("Input tensor metadata missing for model: " + model_name);
        }

        std::vector<int> shape;
        if (tensor->shape()) {
            for (size_t j = 0; j < tensor->shape()->size(); j++) {
                shape.push_back(tensor->shape()->Get(j));
            }
        }
        metadata.input_shapes.push_back(shape);
        metadata.input_dtypes.push_back(tensor_type_to_dtype(tensor->type()));

        std::vector<int> signature;
        if (tensor->shape_signature() && tensor->shape_signature()->size() > 0) {
            for (size_t j = 0; j < tensor->shape_signature()->size(); j++) {
                signature.push_back(tensor->shape_signature()->Get(j));
            }
        } else {
            signature = shape;
        }
        metadata.input_shape_signatures.push_back(signature);
    }

    return metadata;
}

CorrectnessResult run_correctness_test(const std::string& model_name,
                                       int num_inputs,
                                       uint32_t seed) {
    if (num_inputs <= 0) {
        throw std::runtime_error("Correctness input count must be positive");
    }

    std::cout << "=== Start virtualized TFLite interpreter testing ===" << std::endl;
    std::cout << "Model: " << model_name << std::endl;
    std::cout << "Seed: " << seed << std::endl;
    std::cout << "Random inputs: " << num_inputs << std::endl;
    std::cout << "Virtualized model files: "
              << virtualized_infos_path(model_name) << ", "
              << virtualized_params_path(model_name) << std::endl;
    std::cout << "Original model file: " << original_model_path(model_name) << std::endl;

    ModelMetadata metadata = load_model_metadata(model_name);
    print_metadata(metadata);

    auto v_model = tflite::FlatBufferModel::BuildFromVirtualizedFiles(
        virtualized_infos_path(model_name).c_str(),
        virtualized_params_path(model_name).c_str());
    if (!v_model) {
        throw std::runtime_error("Failed to load virtualized model for " + model_name);
    }
    auto v_interpreter = create_interpreter_from_model(v_model.get());
    if (!v_interpreter) {
        throw std::runtime_error("Failed to create virtualized interpreter for " + model_name);
    }
    v_interpreter->AllocateTensors();
    auto v_input_details = get_input_details(v_interpreter.get());
    auto v_output_details = get_output_details(v_interpreter.get());

    auto o_model = tflite::FlatBufferModel::BuildFromFile(original_model_path(model_name).c_str());
    if (!o_model) {
        throw std::runtime_error("Failed to load original model for " + model_name);
    }
    auto o_interpreter = create_interpreter_from_model(o_model.get());
    if (!o_interpreter) {
        throw std::runtime_error("Failed to create original interpreter for " + model_name);
    }
    o_interpreter->AllocateTensors();
    auto o_input_details = get_input_details(o_interpreter.get());
    auto o_output_details = get_output_details(o_interpreter.get());

    CorrectnessResult result;
    result.model_name = model_name;
    result.seed = seed;
    result.num_inputs = num_inputs;
    result.is_dynamic = has_dynamic_signature(v_input_details);

    set_random_seed(seed);

    if (result.is_dynamic) {
        std::cout << "\n=== Detected dynamic shape input, starting dynamic test ===" << std::endl;
        result.dynamic_lengths = default_dynamic_lengths(v_input_details);
        result.length_allocations = distribute_evenly(num_inputs, result.dynamic_lengths.size());

        std::cout << "Selected dynamic lengths: [";
        for (size_t i = 0; i < result.dynamic_lengths.size(); i++) {
            std::cout << result.dynamic_lengths[i];
            if (i + 1 < result.dynamic_lengths.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;

        MetricsAccumulator overall_accumulator;

        for (size_t idx = 0; idx < result.dynamic_lengths.size(); idx++) {
            int length = result.dynamic_lengths[idx];
            int current_inputs = result.length_allocations[idx];
            std::cout << "\n--- Dynamic length/size = " << length
                      << " (" << current_inputs << " inputs) ---" << std::endl;
            if (current_inputs == 0) {
                std::cout << "Skip this bucket because no inputs were allocated." << std::endl;
                continue;
            }

            resize_interpreter_tensors(v_interpreter.get(), v_input_details, length);
            resize_interpreter_tensors(o_interpreter.get(), o_input_details, length);

            MetricsAccumulator bucket_accumulator;
            std::vector<TensorData> last_outputs_vir;
            std::vector<TensorData> last_outputs_ori;

            for (int input_idx = 0; input_idx < current_inputs; input_idx++) {
                auto test_batch = generate_dynamic_test_input_batch(
                    v_input_details, metadata.input_dtypes, length);

                set_interpreter_inputs(v_interpreter.get(), test_batch);
                v_interpreter->Invoke();
                last_outputs_vir = collect_outputs(v_interpreter.get(), v_output_details);

                set_interpreter_inputs(o_interpreter.get(), test_batch);
                o_interpreter->Invoke();
                last_outputs_ori = collect_outputs(o_interpreter.get(), o_output_details);

                update_metrics_accumulator(bucket_accumulator, last_outputs_vir, last_outputs_ori);
            }

            print_last_output_shapes(
                last_outputs_vir,
                "Virtualized model inference completed, output shape of the last run: ");
            print_last_output_shapes(
                last_outputs_ori,
                "Original model inference completed, output shape of the last run: ");
            print_metrics_summary(finalize_metrics_accumulator(bucket_accumulator));
            merge_metrics_accumulator(overall_accumulator, bucket_accumulator);
        }

        std::cout << "\n=== Overall results for all dynamic lengths ===" << std::endl;
        result.metrics = finalize_metrics_accumulator(overall_accumulator);
    } else {
        std::cout << "\n=== Detected static-shape input; starting static testing ===" << std::endl;
        MetricsAccumulator accumulator;
        std::vector<TensorData> last_outputs_vir;
        std::vector<TensorData> last_outputs_ori;

        for (int input_idx = 0; input_idx < num_inputs; input_idx++) {
            auto test_batch = generate_test_input_batch(metadata.input_shapes, metadata.input_dtypes);

            set_interpreter_inputs(v_interpreter.get(), test_batch);
            v_interpreter->Invoke();
            last_outputs_vir = collect_outputs(v_interpreter.get(), v_output_details);

            set_interpreter_inputs(o_interpreter.get(), test_batch);
            o_interpreter->Invoke();
            last_outputs_ori = collect_outputs(o_interpreter.get(), o_output_details);

            update_metrics_accumulator(accumulator, last_outputs_vir, last_outputs_ori);
        }

        print_last_output_shapes(last_outputs_vir,
                                 "Virtualized model inference completed, output shape: ");
        print_last_output_shapes(last_outputs_ori,
                                 "Original model inference completed, output shape: ");
        result.metrics = finalize_metrics_accumulator(accumulator);
    }

    print_metrics_summary(result.metrics);
    result.aggregate = build_correctness_aggregate(result.metrics);
    return result;
}

PerformanceResult run_virtualized_performance_test(const std::string& model_name,
                                                   int num_iters,
                                                   uint32_t seed) {
    if (num_iters <= 0) {
        throw std::runtime_error("Performance iteration count must be positive");
    }

    std::cout << "=== Virtualized Model Performance Test ===" << std::endl;
    std::cout << "Model: " << model_name << std::endl;
    std::cout << "Seed: " << seed << std::endl;
    std::cout << "Iterations: " << num_iters << std::endl;

    ModelMetadata metadata = load_model_metadata(model_name);
    set_random_seed(seed);
    auto test_inputs = generate_test_inputs(metadata.input_shapes, metadata.input_dtypes, num_iters);
    const int memory_iters = 1;

    PeakMemoryTracker tracker(5);
    tracker.start();

    auto build_start = std::chrono::high_resolution_clock::now();
    auto v_model = tflite::FlatBufferModel::BuildFromVirtualizedFiles(
        virtualized_infos_path(model_name).c_str(),
        virtualized_params_path(model_name).c_str());
    if (!v_model) {
        throw std::runtime_error("Failed to load virtualized model for " + model_name);
    }

    auto v_interpreter = create_interpreter_from_model(v_model.get());
    if (!v_interpreter) {
        throw std::runtime_error("Failed to create virtualized interpreter for " + model_name);
    }

    v_interpreter->AllocateTensors();
    auto build_end = std::chrono::high_resolution_clock::now();

    if (!test_inputs.empty()) {
        set_interpreter_inputs(v_interpreter.get(), test_inputs.front());
        v_interpreter->Invoke();
    }
    tracker.stop();

    auto inference_start = std::chrono::high_resolution_clock::now();
    for (auto& test_batch : test_inputs) {
        set_interpreter_inputs(v_interpreter.get(), test_batch);
        v_interpreter->Invoke();
    }
    auto inference_end = std::chrono::high_resolution_clock::now();

    PerformanceResult result;
    result.model_name = model_name;
    result.seed = seed;
    result.num_iters = num_iters;
    result.memory_iters = memory_iters;
    result.baseline_mb = tracker.get_baseline_mb();
    result.peak_mb = tracker.get_peak_mb();
    result.overhead_mb = tracker.get_overhead_mb();
    result.build_time_sec = std::chrono::duration<double>(build_end - build_start).count();
    result.inference_time_sec = std::chrono::duration<double>(inference_end - inference_start).count();
    result.total_time_sec = result.build_time_sec + result.inference_time_sec;
    return result;
}

PerformanceResult run_original_performance_test(const std::string& model_name,
                                                int num_iters,
                                                uint32_t seed) {
    if (num_iters <= 0) {
        throw std::runtime_error("Performance iteration count must be positive");
    }

    std::cout << "=== Original Model Performance Test ===" << std::endl;
    std::cout << "Model: " << model_name << std::endl;
    std::cout << "Seed: " << seed << std::endl;
    std::cout << "Iterations: " << num_iters << std::endl;

    ModelMetadata metadata = load_model_metadata(model_name);
    set_random_seed(seed);
    auto test_inputs = generate_test_inputs(metadata.input_shapes, metadata.input_dtypes, num_iters);
    const int memory_iters = 1;

    PeakMemoryTracker tracker(5);
    tracker.start();

    auto build_start = std::chrono::high_resolution_clock::now();
    auto o_model = tflite::FlatBufferModel::BuildFromFile(original_model_path(model_name).c_str());
    if (!o_model) {
        throw std::runtime_error("Failed to load original model for " + model_name);
    }

    auto o_interpreter = create_interpreter_from_model(o_model.get());
    if (!o_interpreter) {
        throw std::runtime_error("Failed to create original interpreter for " + model_name);
    }

    o_interpreter->AllocateTensors();
    auto build_end = std::chrono::high_resolution_clock::now();

    if (!test_inputs.empty()) {
        set_interpreter_inputs(o_interpreter.get(), test_inputs.front());
        o_interpreter->Invoke();
    }
    tracker.stop();

    auto inference_start = std::chrono::high_resolution_clock::now();
    for (auto& test_batch : test_inputs) {
        set_interpreter_inputs(o_interpreter.get(), test_batch);
        o_interpreter->Invoke();
    }
    auto inference_end = std::chrono::high_resolution_clock::now();

    PerformanceResult result;
    result.model_name = model_name;
    result.seed = seed;
    result.num_iters = num_iters;
    result.memory_iters = memory_iters;
    result.baseline_mb = tracker.get_baseline_mb();
    result.peak_mb = tracker.get_peak_mb();
    result.overhead_mb = tracker.get_overhead_mb();
    result.build_time_sec = std::chrono::duration<double>(build_end - build_start).count();
    result.inference_time_sec = std::chrono::duration<double>(inference_end - inference_start).count();
    result.total_time_sec = result.build_time_sec + result.inference_time_sec;
    return result;
}

void print_correctness_result(const CorrectnessResult& result) {
    std::cout << "\n=== Correctness Summary ===" << std::endl;
    std::cout << "Model: " << result.model_name << std::endl;
    std::cout << "Random inputs: " << result.num_inputs << std::endl;
    if (result.is_dynamic) {
        std::cout << "Dynamic allocations: ";
        for (size_t i = 0; i < result.dynamic_lengths.size(); i++) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << result.dynamic_lengths[i] << "->" << result.length_allocations[i];
        }
        std::cout << std::endl;
    }
    std::cout << "Model-level metrics (max across outputs): "
              << "MSE=" << format_metric_value(result.aggregate.mse)
              << ", MAE=" << format_metric_value(result.aggregate.mae)
              << ", MaxAE=" << format_metric_value(result.aggregate.maxae)
              << ", RelMAE=" << format_metric_value(result.aggregate.relmae);
    if (result.aggregate.has_top1_difference) {
        std::cout << ", Top-1 Difference="
                  << format_metric_value(result.aggregate.top1_difference);
    }
    std::cout << std::endl;
}

void print_performance_result(const PerformanceResult& result,
                              const std::string& label) {
    std::cout << "\n=== " << label << " Summary ===" << std::endl;
    std::cout << "Model: " << result.model_name << std::endl;
    std::cout << "Baseline: " << result.baseline_mb << " MB" << std::endl;
    std::cout << "Peak: " << result.peak_mb << " MB" << std::endl;
    std::cout << "Overhead: " << result.overhead_mb << " MB" << std::endl;
    std::cout << "Peak RSS window: model load + " << result.memory_iters
              << " inference iteration(s)" << std::endl;
    std::cout << "Model Build Time: " << result.build_time_sec << " s" << std::endl;
    std::cout << "Inference Time (" << result.num_iters
              << " iterations): " << result.inference_time_sec << " s" << std::endl;
    std::cout << "Total Time: " << result.total_time_sec << " s" << std::endl;
}

std::vector<std::string> resolve_model_names(const std::string& model_name,
                                             const std::string& models_csv,
                                             const std::string& model_set,
                                             bool all_models) {
    int selectors = 0;
    selectors += model_name.empty() ? 0 : 1;
    selectors += models_csv.empty() ? 0 : 1;
    selectors += model_set.empty() ? 0 : 1;
    selectors += all_models ? 1 : 0;
    if (selectors > 1) {
        throw std::runtime_error(
            "Use only one of --model_name, --models, --model_set, --all_models");
    }

    std::vector<std::string> resolved_models;
    if (!model_name.empty()) {
        resolved_models.push_back(model_name);
    } else if (!models_csv.empty()) {
        resolved_models = split_csv(models_csv);
    } else {
        std::string resolved_set = all_models ? "all_tflite" : model_set;
        if (resolved_set.empty()) {
            resolved_set = "paper11";
        }
        if (resolved_set == "paper11") {
            resolved_models = paper11_models();
        } else if (resolved_set == "comparison10") {
            resolved_models = comparison10_models();
        } else if (resolved_set == "all_tflite") {
            resolved_models = all_tflite_models();
        } else {
            throw std::runtime_error("Unsupported model set: " + resolved_set);
        }
    }

    if (resolved_models.empty()) {
        throw std::runtime_error("No models selected");
    }

    std::set<std::string> seen_models;
    std::vector<std::string> unique_models;
    for (const auto& item : resolved_models) {
        if (seen_models.insert(item).second) {
            validate_model_file_exists(item);
            unique_models.push_back(item);
        }
    }
    return unique_models;
}

bool is_classification_model(const std::string& model_name) {
    return classification_models().count(model_name) > 0;
}

std::string build_rq1_markdown(const std::vector<ModelSuiteResult>& results,
                               const std::string& platform_label,
                               int num_inputs,
                               uint32_t seed) {
    std::ostringstream oss;
    oss << "# RQ1 Functional Equivalence";
    if (!platform_label.empty()) {
        oss << " (" << platform_label << ")";
    }
    oss << "\n\n";
    oss << "> Random inputs per model: " << num_inputs << "; seed=" << seed << "\n\n";
    oss << "| Model | MSE | MAE | MaxAE | RelMAE | Top-1 Difference |\n";
    oss << "| ----- | --- | --- | ----- | ------ | ---------------- |\n";
    for (const auto& item : results) {
        if (!item.has_correctness) {
            continue;
        }
        oss << "| " << item.model_name
            << " | " << format_metric_value(item.correctness.aggregate.mse)
            << " | " << format_metric_value(item.correctness.aggregate.mae)
            << " | " << format_metric_value(item.correctness.aggregate.maxae)
            << " | " << format_metric_value(item.correctness.aggregate.relmae)
            << " | ";
        if (is_classification_model(item.model_name) &&
            item.correctness.aggregate.has_top1_difference) {
            oss << format_metric_value(item.correctness.aggregate.top1_difference);
        } else {
            oss << "N/A";
        }
        oss << " |\n";
    }
    return oss.str();
}

std::string build_rq2_markdown(const std::vector<ModelSuiteResult>& results,
                               const std::string& platform_label,
                               int num_iters,
                               uint32_t seed) {
    std::ostringstream oss;
    oss << "# RQ2 Efficiency";
    if (!platform_label.empty()) {
        oss << " (" << platform_label << ")";
    }
    oss << "\n\n";
    oss << "> Inference iterations per model: " << num_iters
        << "; peak RSS window: model load + 1 inference; seed=" << seed << "\n\n";
    oss << "| Model | Virt. Inference (s) | Native Inference (s) | Inference Ratio | Virt. Peak RSS (MB) | Native Peak RSS (MB) | Memory Ratio |\n";
    oss << "| ----- | ------------------- | -------------------- | --------------- | ------------------- | -------------------- | ------------ |\n";

    double inference_ratio_sum = 0.0;
    double memory_ratio_sum = 0.0;
    int ratio_count = 0;
    for (const auto& item : results) {
        if (!item.has_performance) {
            continue;
        }
        oss << "| " << item.model_name
            << " | " << format_decimal(item.virtualized.inference_time_sec, 6)
            << " | " << format_decimal(item.original.inference_time_sec, 6)
            << " | " << ratio_to_percent(item.inference_ratio)
            << " | " << format_decimal(item.virtualized.peak_mb, 4)
            << " | " << format_decimal(item.original.peak_mb, 4)
            << " | " << ratio_to_percent(item.memory_ratio)
            << " |\n";
        inference_ratio_sum += item.inference_ratio;
        memory_ratio_sum += item.memory_ratio;
        ratio_count++;
    }

    if (ratio_count > 0) {
        oss << "| Average | - | - | " << ratio_to_percent(inference_ratio_sum / ratio_count)
            << " | - | - | " << ratio_to_percent(memory_ratio_sum / ratio_count)
            << " |\n";
    }
    return oss.str();
}

std::string build_summary_json(const std::vector<ModelSuiteResult>& results,
                               const std::string& platform_label,
                               int num_inputs,
                               int num_iters,
                               uint32_t seed,
                               SuiteSelection selection) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"platform_label\":" << quote_json(platform_label) << ",";
    oss << "\"selection\":" << quote_json(suite_selection_to_string(selection)) << ",";
    oss << "\"correctness_inputs\":" << num_inputs << ",";
    oss << "\"performance_iters\":" << num_iters << ",";
    oss << "\"seed\":" << seed << ",";
    oss << "\"models\":[";
    for (size_t i = 0; i < results.size(); i++) {
        if (i > 0) {
            oss << ",";
        }
        oss << model_suite_result_to_json(results[i]);
    }
    oss << "]";
    oss << "}";
    return oss.str();
}

void write_text_file(const std::string& path, const std::string& content) {
    std::filesystem::path target(path);
    if (target.has_parent_path()) {
        std::filesystem::create_directories(target.parent_path());
    }
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    ofs << content;
}

void write_correctness_result_json(const CorrectnessResult& result,
                                   const std::string& path) {
    write_text_file(path, correctness_result_to_json(result));
}

void write_performance_result_json(const PerformanceResult& result,
                                   const std::string& path) {
    write_text_file(path, performance_result_to_json(result));
}

void write_model_result_json(const ModelSuiteResult& result,
                             const std::string& path) {
    write_text_file(path, model_suite_result_to_json(result));
}
