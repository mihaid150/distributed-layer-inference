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

    std::string activation_precision = "fp32";
    std::string transport_mode = "binary_octet_stream";
    std::string rebalance_profile = "baseline";
    bool persistent_sessions_enabled = false;
    bool topology_aware_routing = false;
    bool native_stage_chaining_enabled = false;

    bool has_temperature = false;
    double temperature = 0.0;

    bool has_top_k = false;
    int top_k = 0;

    bool has_top_p = false;
    double top_p = 0.0;

    bool has_next_token_id = false;
    int next_token_id = -1;

    int stage_input_token_count = 0;
};
ParsedRequestMetadata parse_request_metadata(const std::string& metadata_json);

} // namespace dli::common