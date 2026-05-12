#include "dli/common/gguf_inspector.hpp"

#include "gguf.h"

#include <stdexcept>
#include <string>

namespace dli::common {

namespace {

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

} // namespace

GgufInspection inspect_gguf_tensors(const std::string& path) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;

    gguf_context* ctx = gguf_init_from_file(path.c_str(), params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to open GGUF file for inspection: " + path);
    }

    GgufInspection inspection;
    inspection.path = path;

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    inspection.tensor_count = static_cast<int>(n_tensors);
    inspection.tensors.reserve(static_cast<std::size_t>(n_tensors));

    for (int64_t i = 0; i < n_tensors; ++i) {
        GgufTensorInfo info;

        const char* raw_name = gguf_get_tensor_name(ctx, i);
        info.name = raw_name != nullptr ? raw_name : "";

        const ggml_type tensor_type = gguf_get_tensor_type(ctx, i);
        const char* raw_type_name = ggml_type_name(tensor_type);
        info.type_name = raw_type_name != nullptr ? raw_type_name : "";

        info.byte_size = static_cast<uint64_t>(gguf_get_tensor_size(ctx, i));
        info.offset = static_cast<uint64_t>(gguf_get_tensor_offset(ctx, i));

        // Shape extraction is intentionally deferred until we stabilize against
        // the exact gguf API available in the pinned llama.cpp revision.
        inspection.tensors.push_back(std::move(info));
    }

    gguf_free(ctx);

    return inspection;
}

bool gguf_contains_tensor(
    const GgufInspection& inspection,
    const std::string& tensor_name
) {
    for (const auto& tensor : inspection.tensors) {
        if (tensor.name == tensor_name) {
            return true;
        }
    }

    return false;
}

bool gguf_contains_tensor_with_prefix(
    const GgufInspection& inspection,
    const std::string& prefix
) {
    for (const auto& tensor : inspection.tensors) {
        if (starts_with(tensor.name, prefix)) {
            return true;
        }
    }

    return false;
}

} // namespace dli::common