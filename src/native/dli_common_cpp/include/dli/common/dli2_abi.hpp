#pragma once

#include "dli/common/tensor.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dli::common {

enum class Dli2PayloadKind {
    TokenIds,
    HiddenStates,
    Empty
};

struct Dli2TensorAbi {
    static constexpr const char* dtype_token_ids = "int64";
    static constexpr const char* dtype_hidden_f32 = "float32";
    static constexpr const char* dtype_hidden_f16 = "float16";

    static constexpr const char* byte_order_little = "little";
};

inline std::int64_t product_shape(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }

    std::int64_t product = 1;

    for (const std::int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("DLI2 tensor shape contains non-positive dimension");
        }

        product *= dim;
    }

    return product;
}

inline std::size_t dtype_size_bytes(const std::string& dtype) {
    if (dtype == Dli2TensorAbi::dtype_token_ids) {
        return sizeof(std::int64_t);
    }

    if (dtype == Dli2TensorAbi::dtype_hidden_f32) {
        return sizeof(float);
    }

    if (dtype == Dli2TensorAbi::dtype_hidden_f16) {
        return 2;
    }

    throw std::runtime_error("unsupported DLI2 dtype: " + dtype);
}

inline void validate_tensor_byte_size(const TensorBuffer& tensor) {
    const std::int64_t elements = product_shape(tensor.metadata.shape);

    if (elements < 0) {
        throw std::runtime_error("invalid DLI2 tensor element count");
    }

    const std::size_t expected =
        static_cast<std::size_t>(elements) *
        dtype_size_bytes(tensor.metadata.dtype);

    if (tensor.bytes.size() != expected) {
        throw std::runtime_error(
            "DLI2 tensor byte size mismatch: expected " +
            std::to_string(expected) +
            " bytes, got " +
            std::to_string(tensor.bytes.size())
        );
    }
}

inline void validate_little_endian(const TensorBuffer& tensor) {
    if (
        tensor.metadata.byte_order.empty() ||
        tensor.metadata.byte_order == Dli2TensorAbi::byte_order_little
    ) {
        return;
    }

    throw std::runtime_error(
        "unsupported DLI2 tensor byte_order: " +
        tensor.metadata.byte_order
    );
}

inline void validate_token_ids_tensor(const TensorBuffer& tensor) {
    validate_little_endian(tensor);

    if (tensor.metadata.dtype != Dli2TensorAbi::dtype_token_ids) {
        throw std::runtime_error(
            "DLI2 token ids tensor must have dtype=int64, got " +
            tensor.metadata.dtype
        );
    }

    if (tensor.metadata.shape.size() != 2) {
        throw std::runtime_error("DLI2 token ids tensor must have shape [batch, seq]");
    }

    validate_tensor_byte_size(tensor);
}

inline void validate_hidden_states_tensor(
    const TensorBuffer& tensor,
    int expected_hidden_size,
    bool allow_float16
) {
    validate_little_endian(tensor);

    const bool is_f32 = tensor.metadata.dtype == Dli2TensorAbi::dtype_hidden_f32;
    const bool is_f16 = tensor.metadata.dtype == Dli2TensorAbi::dtype_hidden_f16;

    if (!is_f32 && !(allow_float16 && is_f16)) {
        throw std::runtime_error(
            "DLI2 hidden states tensor must have dtype=float32"
            " during correctness phase"
        );
    }

    if (tensor.metadata.shape.size() != 3) {
        throw std::runtime_error(
            "DLI2 hidden states tensor must have shape [batch, seq, hidden_size]"
        );
    }

    if (expected_hidden_size > 0 && tensor.metadata.shape[2] != expected_hidden_size) {
        throw std::runtime_error(
            "DLI2 hidden_size mismatch: expected " +
            std::to_string(expected_hidden_size) +
            ", got " +
            std::to_string(tensor.metadata.shape[2])
        );
    }

    validate_tensor_byte_size(tensor);
}

inline Dli2PayloadKind infer_dli2_payload_kind(const TensorBuffer& tensor) {
    if (tensor.bytes.empty() && tensor.metadata.shape.empty()) {
        return Dli2PayloadKind::Empty;
    }

    if (tensor.metadata.dtype == Dli2TensorAbi::dtype_token_ids) {
        return Dli2PayloadKind::TokenIds;
    }

    if (
        tensor.metadata.dtype == Dli2TensorAbi::dtype_hidden_f32 ||
        tensor.metadata.dtype == Dli2TensorAbi::dtype_hidden_f16
    ) {
        return Dli2PayloadKind::HiddenStates;
    }

    throw std::runtime_error("cannot infer DLI2 payload kind for dtype: " + tensor.metadata.dtype);
}

inline std::int64_t dli2_batch_size(const TensorBuffer& tensor) {
    if (tensor.metadata.shape.empty()) {
        return 0;
    }

    return tensor.metadata.shape[0];
}

inline std::int64_t dli2_sequence_length(const TensorBuffer& tensor) {
    if (tensor.metadata.shape.size() < 2) {
        return 0;
    }

    return tensor.metadata.shape[1];
}

} // namespace dli::common