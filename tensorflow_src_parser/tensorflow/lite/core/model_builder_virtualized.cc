// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The NeuralVirtualizer Authors
#include "tensorflow/lite/core/model_builder.h"

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/mlir/lite/core/macros.h"
#include "tensorflow/lite/parser/virtualized_model_loader.h"

namespace tflite {
namespace impl {

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromVirtualizedFiles(
    const char* info_path, const char* params_path, ErrorReporter* reporter) {
  reporter = reporter ? reporter : GetDefaultErrorReporter();
  if (info_path == nullptr || params_path == nullptr) {
    TF_LITE_REPORT_ERROR(reporter,
                         "BuildFromVirtualizedFiles requires non-null paths.");
    return nullptr;
  }

  // Prefer zero-copy path: build an Allocation that owns the finalized buffer
  // without copying. This removes the double-peak memory usage during copy.
  auto alloc_or =
      parser::BuildVirtualizedAllocation(info_path, params_path, reporter);
  if (alloc_or.ok()) {
    std::unique_ptr<Allocation> allocation = std::move(alloc_or.value());
    if (!allocation || !allocation->valid()) {
      TF_LITE_REPORT_ERROR(reporter,
                           "Virtualized zero-copy allocation is invalid.");
      return nullptr;
    }
    return VerifyAndBuildFromAllocation(
        std::move(allocation), /*extra_verifier=*/nullptr, reporter);
  }

  // Fallback: legacy vector path for compatibility.
  auto buffer_or =
      parser::BuildVirtualizedFlatbuffer(info_path, params_path, reporter);
  if (!buffer_or.ok()) {
    return nullptr;
  }
  std::shared_ptr<std::vector<uint8_t>> buffer = buffer_or.value();
  if (!buffer || buffer->empty()) {
    TF_LITE_REPORT_ERROR(reporter,
                         "Virtualized model builder returned an empty buffer.");
    return nullptr;
  }
  auto allocation = std::make_unique<parser::VirtualizedVectorAllocation>(
      buffer, reporter);
  if (!allocation || !allocation->valid()) {
    TF_LITE_REPORT_ERROR(reporter,
                         "Virtualized buffer allocation is invalid.");
    return nullptr;
  }
  return VerifyAndBuildFromAllocation(
      std::move(allocation), /*extra_verifier=*/nullptr, reporter);
}

}  // namespace impl
}  // namespace tflite
