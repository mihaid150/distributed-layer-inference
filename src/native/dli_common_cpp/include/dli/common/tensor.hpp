#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dli::common {

struct TensorMetadata {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::string byte_order = "little";
};

struct TensorBuffer {
    TensorMetadata metadata;
    std::vector<std::uint8_t> bytes;
};

int infer_sequence_length_from_shape(const std::vector<std::int64_t>& shape);

} // namespace dli::common