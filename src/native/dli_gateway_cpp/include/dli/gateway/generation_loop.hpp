#pragma once

#include "dli/gateway/stage_client.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace dli::gateway {

struct GenerationLoopConfig {
    std::string first_stage_url;
    int max_new_tokens = 2;
};

struct GenerationStepTrace {
    int token_index = 0;
    std::string generation_mode;
    int stage_http_status = 0;
    std::string stage_http_reason;
    std::string stage_metadata_json;
    std::size_t stage_tensor_bytes = 0;
};

struct GenerationLoopResult {
    std::string request_id;
    std::vector<GenerationStepTrace> steps;

    double total_latency_ms = 0.0;
    double tokens_per_second = 0.0;
    int generated_token_count = 0;
};

class GenerationLoop {
public:
    explicit GenerationLoop(GenerationLoopConfig config);

    GenerationLoopResult run_stub_generation(int max_new_tokens) const;

private:
    GenerationLoopConfig config_;
};

std::string generation_loop_result_json(
    const GenerationLoopResult& result,
    std::size_t request_body_bytes
);

} // namespace dli::gateway