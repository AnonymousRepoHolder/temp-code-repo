# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The NeuralVirtualizer Authors
load("//tensorflow/lite:build_def.bzl", "tflite_copts")

cc_library(
    name = "virtualization_core",
    srcs = [
        "virtualized_model_builder.cc",
        "virtualized_model_parser.cc",
    ],
    hdrs = [
        "virtualized_model_builder.h",
        "virtualized_model_parser.h",
    ],
    copts = tflite_copts(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "//tensorflow/lite/core:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
        "@nlohmann_json_lib//:nlohmann_json",
        "@boringssl//:crypto"
    ],
)

cc_library(
    name = "virtualized_model_loader",
    srcs = ["virtualized_model_loader.cc"],
    hdrs = ["virtualized_model_loader.h"],
    copts = tflite_copts(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "//tensorflow/lite/core:__subpackages__",
    ],
    deps = [
        ":virtualization_core",
        "//tensorflow/compiler/mlir/lite:allocation",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite:macros",
        "@com_google_absl//absl/status:status",
        "@com_google_absl//absl/status:statusor",
    ],
)
