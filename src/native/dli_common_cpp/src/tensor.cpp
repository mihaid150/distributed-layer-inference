#include "dli/common/tensor.hpp"

namespace dli::common {

int infer_sequence_length_from_shape(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }

    // token ids: [batch, seq]
    // hidden states: [batch, seq, hidden]
    if (shape.size() >= 2) {
        const auto seq = shape[1];
        return seq > 0 ? static_cast<int>(seq) : 0;
    }

    // fallback: [seq]
    const auto seq = shape[0];
    return seq > 0 ? static_cast<int>(seq) : 0;
}

} // namespace dli::common