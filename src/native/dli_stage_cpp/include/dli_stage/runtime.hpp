#pragma once

#include "dli/common/metrics.hpp"
#include "dli/common/tensor.hpp"

#include <string>

namespace dli_stage {

struct RuntimeRequest {
    std::string request_id;
    int token_index = 0;
    int stage_id = 0;
    std::string generation_mode = "legacy";

    std::string input_metadata_json;
    dli::common::TensorBuffer input_tensor;

    bool kv_cache_enabled = false;

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

struct RuntimeResponse {
    bool is_final_stage = false;
    int next_token_id = -1;

    

    std::string output_metadata_json;
    dli::common::TensorBuffer output_tensor;

    dli::common::StageMetrics metrics;

    // Optional runtime-specific metadata, already serialized as JSON object text.
    // Example: {"model_loaded":true,"n_layer":22}
    std::string backend_metadata_json;
};

class StageRuntime {
public:
    virtual ~StageRuntime() = default;

    virtual RuntimeResponse forward(const RuntimeRequest& request) = 0;

    virtual std::string backend_name() const = 0;
};

} // namespace dli_stage