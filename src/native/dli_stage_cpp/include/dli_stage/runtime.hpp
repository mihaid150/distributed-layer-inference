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