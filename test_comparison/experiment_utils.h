#ifndef EXPERIMENT_UTILS_H
#define EXPERIMENT_UTILS_H

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
#include "comparison_utils.h"
#include <cstdint>
#include <string>
#include <vector>

struct ModelMetadata {
    std::vector<std::vector<int>> input_shapes;
    std::vector<std::string> input_dtypes;
    std::vector<std::vector<int>> input_shape_signatures;
};

struct CorrectnessAggregate {
    double mse = 0.0;
    double mae = 0.0;
    double maxae = 0.0;
    double relmae = 0.0;
    bool has_top1_difference = false;
    double top1_difference = 0.0;
};

struct CorrectnessResult {
    std::string model_name;
    uint32_t seed = 42;
    int num_inputs = 0;
    bool is_dynamic = false;
    std::vector<int> dynamic_lengths;
    std::vector<int> length_allocations;
    MetricsSummary metrics;
    CorrectnessAggregate aggregate;
};

struct PerformanceResult {
    std::string model_name;
    uint32_t seed = 42;
    int num_iters = 0;
    int memory_iters = 1;
    double baseline_mb = 0.0;
    double peak_mb = 0.0;
    double overhead_mb = 0.0;
    double build_time_sec = 0.0;
    double inference_time_sec = 0.0;
    double total_time_sec = 0.0;
};

struct ModelSuiteResult {
    std::string model_name;
    bool has_correctness = false;
    bool has_performance = false;
    CorrectnessResult correctness;
    PerformanceResult virtualized;
    PerformanceResult original;
    double build_ratio = 0.0;
    double inference_ratio = 0.0;
    double memory_ratio = 0.0;
};

enum class SuiteSelection {
    All,
    RQ1,
    RQ2,
};

ModelMetadata load_model_metadata(const std::string& model_name);
CorrectnessResult run_correctness_test(const std::string& model_name,
                                       int num_inputs,
                                       uint32_t seed);
PerformanceResult run_virtualized_performance_test(const std::string& model_name,
                                                   int num_iters,
                                                   uint32_t seed);
PerformanceResult run_original_performance_test(const std::string& model_name,
                                                int num_iters,
                                                uint32_t seed);

void print_correctness_result(const CorrectnessResult& result);
void print_performance_result(const PerformanceResult& result,
                              const std::string& label);

std::vector<std::string> resolve_model_names(const std::string& model_name,
                                             const std::string& models_csv,
                                             const std::string& model_set,
                                             bool all_models);
bool is_classification_model(const std::string& model_name);

std::string build_rq1_markdown(const std::vector<ModelSuiteResult>& results,
                               const std::string& platform_label,
                               int num_inputs,
                               uint32_t seed);
std::string build_rq2_markdown(const std::vector<ModelSuiteResult>& results,
                               const std::string& platform_label,
                               int num_iters,
                               uint32_t seed);
std::string build_summary_json(const std::vector<ModelSuiteResult>& results,
                               const std::string& platform_label,
                               int num_inputs,
                               int num_iters,
                               uint32_t seed,
                               SuiteSelection selection);

void write_text_file(const std::string& path, const std::string& content);
void write_correctness_result_json(const CorrectnessResult& result,
                                   const std::string& path);
void write_performance_result_json(const PerformanceResult& result,
                                   const std::string& path);
void write_model_result_json(const ModelSuiteResult& result,
                             const std::string& path);

#endif  // EXPERIMENT_UTILS_H
