#pragma once

#include "dli_stage/tensor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dli_stage {

struct ParsedRequestMetadata {
    std::string request_id;
    int token_index = 0;
    std::string generation_mode = "unknown";

    TensorMetadata tensor;

    bool kv_cache_enabled = false;

    bool has_temperature = false;
    double temperature = 0.0;

    bool has_top_k = false;
    int top_k = 0;

    bool has_top_p = false;
    double top_p = 0.0;

    int stage_input_token_count = 0;
};

ParsedRequestMetadata parse_request_metadata(const std::string& metadata_json);

int infer_sequence_length_from_shape(const std::vector<std::int64_t>& shape);

} // namespace dli_stage