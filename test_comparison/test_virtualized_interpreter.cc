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
    std::cout << "=== C++ TensorFlow Lite virtualized model comparison test ===" << std::endl;
    std::cout << std::endl;

    std::string model_name = "squeezenet";
    int num_inputs = 10;
    uint32_t seed = 42;
    std::string output_json;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.rfind("--model_name=", 0) == 0) {
            model_name = arg.substr(13);
        } else if (arg.rfind("--num_inputs=", 0) == 0) {
            num_inputs = parse_positive_int(arg.substr(13), "--num_inputs");
        } else if (arg.rfind("--seed=", 0) == 0) {
            seed = parse_seed(arg.substr(7));
        } else if (arg.rfind("--output_json=", 0) == 0) {
            output_json = arg.substr(14);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    std::cout << "Model name: " << model_name << std::endl;
    std::cout << "Number of correctness inputs: " << num_inputs << std::endl;
    std::cout << "Seed: " << seed << std::endl;
    std::cout << std::endl;

    try {
        auto result = run_correctness_test(model_name, num_inputs, seed);
        print_correctness_result(result);
        if (!output_json.empty()) {
            write_correctness_result_json(result, output_json);
            std::cout << "Correctness JSON written to: " << output_json << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during the test: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown exception occurred during the test" << std::endl;
        return 1;
    }

    return 0;
}
