#pragma once

#include "dli/common/tensor.hpp"

#include <string>

namespace dli::common {

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

} // namespace dli::common