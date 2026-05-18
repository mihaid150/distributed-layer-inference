#pragma once

#include "dli/gateway/config.hpp"
#include "dli/gateway/stage_client.hpp"
#include "dli/gateway/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dli::gateway {

struct GenerationLoopConfig {
    std::string first_stage_url;
    std::string model_path;
    int max_new_tokens = 2;

    std::vector<PartitionNodeConfig> partitions;

    bool persistent_sessions_enabled = false;
    std::string activation_precision = "fp32";
};

struct GenerationStepTrace {
    int token_index = 0;
    std::string generation_mode;

    std::string partition_id;
    int stage_id = 0;
    std::string stage_url;

    int stage_http_status = 0;
    std::string stage_http_reason;
    double stage_http_elapsed_ms = 0.0;
    std::string stage_metadata_json;
    std::size_t stage_tensor_bytes = 0;
    std::string error_body;
};

struct GatewayAggregatedMetrics {
    double compute_ms = 0.0;
    double rpc_wall_ms = 0.0;
    double true_comm_ms = 0.0;

    std::uint64_t tensor_bytes_in = 0;
    std::uint64_t tensor_bytes_out = 0;

    double model_load_ms = 0.0;
    std::uint64_t kv_cache_bytes = 0;
    std::uint64_t memory_rss_mb = 0;
};

struct GenerationLoopResult {
    std::string request_id;
    std::string prompt;

    std::string tokenizer_backend;
    std::vector<std::int64_t> prompt_token_ids;
    std::vector<int> generated_token_ids;

    std::string generated_text;

    std::vector<GenerationStepTrace> steps;

    GatewayAggregatedMetrics aggregate_metrics;

    double total_latency_ms = 0.0;
    double tokens_per_second = 0.0;
    int generated_token_count = 0;

    bool ok = true;
    std::string error;
    std::string termination_reason = "unknown";
};

class GenerationLoop {
public:
    GenerationLoop(
        GenerationLoopConfig config,
        const Tokenizer& tokenizer
    );

    GenerationLoopResult run_stub_generation(
        const std::string& prompt,
        int max_new_tokens
    ) const;

private:
    GenerationLoopConfig config_;
    const Tokenizer& tokenizer_;
};

std::string generation_loop_result_json(
    const GenerationLoopResult& result,
    std::size_t request_body_bytes
);

} // namespace dli::gateway
