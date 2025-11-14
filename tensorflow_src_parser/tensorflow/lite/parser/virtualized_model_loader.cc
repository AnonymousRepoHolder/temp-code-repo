#include "tensorflow/lite/parser/virtualized_model_loader.h"

#include <utility>
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The NeuralVirtualizer Authors

#include "absl/status/status.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/parser/virtualized_model_builder.h"

namespace tflite {
namespace parser {

VirtualizedVectorAllocation::VirtualizedVectorAllocation(
        std::shared_ptr<std::vector<uint8_t>> buffer, ErrorReporter* reporter)
        : Allocation(reporter, Allocation::Type::kMemory),
            buffer_(std::move(buffer)) {}

const void* VirtualizedVectorAllocation::base() const {
    if (!buffer_ || buffer_->empty()) {
        return nullptr;
    }
    return buffer_->data();
}

size_t VirtualizedVectorAllocation::bytes() const {
    return buffer_ ? buffer_->size() : 0;
}

bool VirtualizedVectorAllocation::valid() const {
    return static_cast<bool>(buffer_);
}

absl::StatusOr<std::shared_ptr<std::vector<uint8_t>>>
BuildVirtualizedFlatbuffer(const std::string& v_infos_path,
                                                    const std::string& params_path,
                                                    ErrorReporter* reporter) {
    VirtualizedModelBuilder model_builder(v_infos_path, params_path);
    if (!model_builder.ok()) {
        const std::string message = model_builder.error_message().empty()
                                                                        ? std::string("Virtualized model rebuild failed.")
                                                                        : model_builder.error_message();
        if (reporter) {
            TF_LITE_REPORT_ERROR(reporter, "%s", message.c_str());
        }
        return absl::InternalError(message);
    }

    // Use release_model_buffer to avoid copying and move the buffer directly.
    auto buffer =
            std::make_shared<std::vector<uint8_t>>(model_builder.release_model_buffer());
    if (!buffer || buffer->empty()) {
        if (reporter) {
            TF_LITE_REPORT_ERROR(
                    reporter, "Virtualized model rebuild produced an empty buffer.");
        }
        return absl::InternalError(
                "Virtualized model rebuild produced an empty buffer.");
    }

    return buffer;
}

absl::StatusOr<std::unique_ptr<Allocation>>
BuildVirtualizedAllocation(const std::string& v_infos_path,
                           const std::string& params_path,
                           ErrorReporter* reporter) {
    // Build using the zero-copy path (DetachedBuffer ownership).
    VirtualizedModelBuilder model_builder(
        v_infos_path, params_path,
        VirtualizedModelBuilder::BuildOutput::kDetached);
    if (!model_builder.ok()) {
        const std::string message = model_builder.error_message().empty()
                                        ? std::string("Virtualized model rebuild failed.")
                                        : model_builder.error_message();
        if (reporter) {
            TF_LITE_REPORT_ERROR(reporter, "%s", message.c_str());
        }
        return absl::InternalError(message);
    }

    auto detached = model_builder.release_model_detached();
    if (detached.data() == nullptr || detached.size() == 0) {
        if (reporter) {
            TF_LITE_REPORT_ERROR(reporter,
                                 "Virtualized model rebuild produced an empty buffer.");
        }
        return absl::InternalError(
            "Virtualized model rebuild produced an empty buffer.");
    }

    // Wrap the DetachedBuffer into an Allocation that owns its lifetime.
    std::unique_ptr<Allocation> allocation =
        std::make_unique<VirtualizedDetachedAllocation>(std::move(detached), reporter);
    return allocation;
}

}    // namespace parser
}    // namespace tflite
