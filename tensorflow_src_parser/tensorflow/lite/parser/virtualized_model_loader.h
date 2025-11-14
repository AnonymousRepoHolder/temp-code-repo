// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The NeuralVirtualizer Authors
#ifndef TENSORFLOW_LITE_PARSER_VIRTUALIZED_MODEL_LOADER_H_
#define TENSORFLOW_LITE_PARSER_VIRTUALIZED_MODEL_LOADER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {
namespace parser {

class VirtualizedVectorAllocation : public Allocation {
public:
  VirtualizedVectorAllocation(std::shared_ptr<std::vector<uint8_t>> buffer,
                              ErrorReporter* reporter);
  ~VirtualizedVectorAllocation() override = default;

  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

  std::shared_ptr<std::vector<uint8_t>> buffer() const { return buffer_; }

private:
  std::shared_ptr<std::vector<uint8_t>> buffer_;
};

absl::StatusOr<std::shared_ptr<std::vector<uint8_t>>> BuildVirtualizedFlatbuffer(
    const std::string& v_infos_path, const std::string& params_path,
    ErrorReporter* reporter);

// Zero-copy path: build the model and return an Allocation that owns
// the finalized FlatBuffer without copying. The underlying memory stays
// valid as long as the returned Allocation is alive.
class VirtualizedDetachedAllocation : public Allocation {
 public:
  VirtualizedDetachedAllocation(flatbuffers::DetachedBuffer buffer,
                                ErrorReporter* reporter)
      : Allocation(reporter, Allocation::Type::kMemory),
        buffer_(std::move(buffer)) {}
  ~VirtualizedDetachedAllocation() override = default;

  const void* base() const override { return buffer_.data(); }
  size_t bytes() const override { return buffer_.size(); }
  bool valid() const override { return buffer_.data() != nullptr && buffer_.size() > 0; }

 private:
  flatbuffers::DetachedBuffer buffer_;
};

// Build zero-copy Allocation for TFLite consumption.
absl::StatusOr<std::unique_ptr<Allocation>> BuildVirtualizedAllocation(
    const std::string& v_infos_path, const std::string& params_path,
    ErrorReporter* reporter);

}  // namespace parser
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PARSER_VIRTUALIZED_MODEL_LOADER_H_
