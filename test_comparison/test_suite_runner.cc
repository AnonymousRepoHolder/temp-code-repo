// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
#include "experiment_utils.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int parse_positive_int(const std::string& value, const std::string& flag_name) {
    int parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::runtime_error(flag_name + " must be positive");
    }
    return parsed;
}

uint32_t parse_seed(const std::string& value) {
    return static_cast<uint32_t>(std::stoul(value));
}

SuiteSelection parse_suite_selection(const std::string& value) {
    if (value == "all") {
        return SuiteSelection::All;
    }
    if (value == "rq1") {
        return SuiteSelection::RQ1;
    }
    if (value == "rq2") {
        return SuiteSelection::RQ2;
    }
    throw std::runtime_error("Unsupported suite selection: " + value);
}

std::string default_platform_label() {
#if defined(ANDROID)
    return "arm64";
#elif defined(_WIN32)
    return "windows";
#else
    return "x86";
#endif
}

void prepare_output_dir(const std::string& output_dir, bool fail_if_exists) {
    std::filesystem::path output_path(output_dir);
    if (std::filesystem::exists(output_path)) {
        if (fail_if_exists) {
            throw std::runtime_error(
                "Output directory already exists. Remove it or omit --fail_if_exists: " +
                output_dir);
        }
        std::filesystem::remove_all(output_path);
    }
    std::filesystem::create_directories(output_path / "raw");
}

}  // namespace

int main(int argc, char* argv[]) {
    std::cout << "=== NeuralVirtualizer Test Suite Runner ===" << std::endl;
    std::cout << std::endl;

    SuiteSelection selection = SuiteSelection::All;
    std::string model_name;
    std::string models_csv;
    std::string model_set;
    bool all_models = false;
    int correctness_inputs = 1000;
    int perf_iters = 1000;
    uint32_t seed = 42;
    std::string output_dir = "results";
    std::string platform_label = default_platform_label();
    bool fail_if_exists = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.rfind("--suite=", 0) == 0) {
            selection = parse_suite_selection(arg.substr(8));
        } else if (arg.rfind("--model_name=", 0) == 0) {
            model_name = arg.substr(13);
        } else if (arg.rfind("--models=", 0) == 0) {
            models_csv = arg.substr(9);
        } else if (arg.rfind("--model_set=", 0) == 0) {
            model_set = arg.substr(12);
        } else if (arg == "--all_models") {
            all_models = true;
        } else if (arg.rfind("--correctness_inputs=", 0) == 0) {
            correctness_inputs = parse_positive_int(
                arg.substr(21), "--correctness_inputs");
        } else if (arg.rfind("--perf_iters=", 0) == 0) {
            perf_iters = parse_positive_int(arg.substr(13), "--perf_iters");
        } else if (arg.rfind("--seed=", 0) == 0) {
            seed = parse_seed(arg.substr(7));
        } else if (arg.rfind("--output_dir=", 0) == 0) {
            output_dir = arg.substr(13);
        } else if (arg.rfind("--platform_label=", 0) == 0) {
            platform_label = arg.substr(17);
        } else if (arg == "--overwrite") {
            // Kept for backward compatibility. Overwrite is now the default behavior.
        } else if (arg == "--fail_if_exists") {
            fail_if_exists = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    try {
        auto models = resolve_model_names(model_name, models_csv, model_set, all_models);
        prepare_output_dir(output_dir, fail_if_exists);

        std::cout << "Selected models:";
        for (const auto& model : models) {
            std::cout << " " << model;
        }
        std::cout << std::endl;
        std::cout << "Suite: "
                  << (selection == SuiteSelection::RQ1 ? "RQ1" :
                      selection == SuiteSelection::RQ2 ? "RQ2" : "ALL")
                  << std::endl;
        std::cout << "Platform label: " << platform_label << std::endl;
        std::cout << "Output directory: " << output_dir << std::endl;
        std::cout << std::endl;

        std::vector<ModelSuiteResult> results;
        for (const auto& model : models) {
            std::cout << "\n==============================" << std::endl;
            std::cout << "Running model: " << model << std::endl;
            std::cout << "==============================" << std::endl;

            ModelSuiteResult model_result;
            model_result.model_name = model;

            if (selection != SuiteSelection::RQ2) {
                model_result.correctness = run_correctness_test(
                    model, correctness_inputs, seed);
                model_result.has_correctness = true;
                print_correctness_result(model_result.correctness);
            }

            if (selection != SuiteSelection::RQ1) {
                model_result.virtualized = run_virtualized_performance_test(
                    model, perf_iters, seed);
                model_result.original = run_original_performance_test(
                    model, perf_iters, seed);
                model_result.has_performance = true;
                model_result.build_ratio = model_result.original.build_time_sec > 0.0
                    ? model_result.virtualized.build_time_sec /
                      model_result.original.build_time_sec
                    : 0.0;
                model_result.inference_ratio = model_result.original.inference_time_sec > 0.0
                    ? model_result.virtualized.inference_time_sec /
                      model_result.original.inference_time_sec
                    : 0.0;
                model_result.memory_ratio = model_result.original.peak_mb > 0.0
                    ? model_result.virtualized.peak_mb /
                      model_result.original.peak_mb
                    : 0.0;

                print_performance_result(model_result.virtualized, "Virtualized Performance");
                print_performance_result(model_result.original, "Original Performance");
                std::cout << "\n=== Derived Ratios ===" << std::endl;
                std::cout << "Build ratio: " << model_result.build_ratio << std::endl;
                std::cout << "Inference ratio: " << model_result.inference_ratio << std::endl;
                std::cout << "Memory ratio: " << model_result.memory_ratio << std::endl;
            }

            results.push_back(model_result);
            write_model_result_json(
                model_result,
                (std::filesystem::path(output_dir) / "raw" / (model + ".json")).string());
        }

        write_text_file(
            (std::filesystem::path(output_dir) / "summary.json").string(),
            build_summary_json(
                results, platform_label, correctness_inputs, perf_iters, seed, selection));

        if (selection != SuiteSelection::RQ2) {
            write_text_file(
                (std::filesystem::path(output_dir) / "RQ1.md").string(),
                build_rq1_markdown(results, platform_label, correctness_inputs, seed));
        }
        if (selection != SuiteSelection::RQ1) {
            write_text_file(
                (std::filesystem::path(output_dir) / "RQ2.md").string(),
                build_rq2_markdown(results, platform_label, perf_iters, seed));
        }

        std::cout << "\nSuite execution completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Suite runner failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
