#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dli_stage {

struct TensorMetadata {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::string byte_order = "little";
};

struct TensorBuffer {
    TensorMetadata metadata;
    std::vector<std::uint8_t> bytes;
};

} // namespace dli_stage