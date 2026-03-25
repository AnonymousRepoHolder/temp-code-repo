// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
#include "experiment_utils.h"
#include <iostream>
#include <stdexcept>
#include <string>

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

}  // namespace

int main(int argc, char* argv[]) {
    std::cout << "=== Virtualized Model Performance Test ===" << std::endl;
    std::cout << std::endl;

    std::string model_name = "squeezenet";
    int num_iters = 10;
    uint32_t seed = 42;
    std::string output_json;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.rfind("--model_name=", 0) == 0) {
            model_name = arg.substr(13);
        } else if (arg.rfind("--num_iters=", 0) == 0) {
            num_iters = parse_positive_int(arg.substr(12), "--num_iters");
        } else if (arg.rfind("--seed=", 0) == 0) {
            seed = parse_seed(arg.substr(7));
        } else if (arg.rfind("--output_json=", 0) == 0) {
            output_json = arg.substr(14);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    try {
        auto result = run_virtualized_performance_test(model_name, num_iters, seed);
        print_performance_result(result, "Virtualized Performance");
        if (!output_json.empty()) {
            write_performance_result_json(result, output_json);
            std::cout << "Performance JSON written to: " << output_json << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
