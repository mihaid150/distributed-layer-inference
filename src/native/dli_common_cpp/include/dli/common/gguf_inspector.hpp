#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dli::common {

struct GgufTensorInfo {
    std::string name;
    std::string type_name;
    std::vector<int64_t> shape;
    uint64_t byte_size = 0;
    uint64_t offset = 0;
};

struct GgufInspection {
    std::string path;
    int tensor_count = 0;
    std::vector<GgufTensorInfo> tensors;
};

GgufInspection inspect_gguf_tensors(const std::string& path);

bool gguf_contains_tensor(
    const GgufInspection& inspection,
    const std::string& tensor_name
);

bool gguf_contains_tensor_with_prefix(
    const GgufInspection& inspection,
    const std::string& prefix
);

} // namespace dli::common