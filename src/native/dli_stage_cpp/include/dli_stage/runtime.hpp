#pragma once

#include "dli_stage/metrics.hpp"
#include "dli_stage/tensor.hpp"

#include <string>

namespace dli_stage {

struct RuntimeRequest {
    std::string request_id;
    int token_index = 0;
    int stage_id = 0;
    std::string generation_mode = "legacy";

    std::string input_metadata_json;
    TensorBuffer input_tensor;

    bool kv_cache_enabled = false;

    bool has_temperature = false;
    double temperature = 0.0;

    bool has_top_k = false;
    int top_k = 0;

    bool has_top_p = false;
    double top_p = 0.0;
};

struct RuntimeResponse {
    bool is_final_stage = false;
    int next_token_id = -1;

    std::string output_metadata_json;
    TensorBuffer output_tensor;

    StageMetrics metrics;
};

class StageRuntime {
public:
    virtual ~StageRuntime() = default;

    virtual RuntimeResponse forward(const RuntimeRequest& request) = 0;

    virtual std::string backend_name() const = 0;
};

} // namespace dli_stage