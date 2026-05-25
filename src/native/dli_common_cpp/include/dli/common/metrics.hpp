#pragma once

#include <cstdint>
#include <string>

namespace dli::common {

struct StageMetrics {
    std::string backend = "stub";
    std::string status = "ok";

    double compute_time_ms = 0.0;
    double true_comm_ms = 0.0;
    double rpc_wall_time_ms = 0.0;

    std::uint64_t input_tensor_bytes = 0;
    std::uint64_t output_tensor_bytes = 0;

    int stage_input_token_count = 0;
    int stage_output_token_count = 0;

    bool kv_cache_step_valid = true;

    int kv_cache_seq_before = 0;
    int kv_cache_seq_after = 0;
    std::uint64_t kv_cache_bytes = 0;
    bool kv_cache_valid = true;

    double model_load_ms = 0.0;
    std::uint64_t memory_rss_mb = 0;
    std::uint64_t memory_cgroup_current_mb = 0;
    std::uint64_t memory_cgroup_limit_mb = 0;
    double memory_cgroup_percent = 0.0;

    std::uint64_t model_file_size_mb = 0;
    std::uint64_t session_count = 0;
    std::uint64_t session_kv_cache_bytes = 0;
};

struct ChainMetrics {
    double compute_ms = 0.0;
    double rpc_wall_ms = 0.0;
    double true_comm_ms = 0.0;
    std::uint64_t transport_payload_bytes = 0;
    std::uint64_t stage_count = 0;
};

ChainMetrics local_chain_metrics_from_stage(const StageMetrics& metrics);

bool metadata_has_chain_metrics(const std::string& metadata_json);


ChainMetrics extract_chain_metrics_from_metadata(const std::string& metadata_json);

std::string put_chain_metrics_in_metadata(
    std::string metadata_json,
    const ChainMetrics& metrics
);

ChainMetrics merge_chain_metrics_values(
    const StageMetrics& local_metrics,
    const ChainMetrics& downstream_metrics,
    double downstream_rpc_wall_ms,
    std::uint64_t request_bytes,
    std::uint64_t response_bytes
);

std::string merge_chain_metrics(
    std::string downstream_metadata,
    const StageMetrics& local_metrics,
    double downstream_rpc_wall_ms,
    std::uint64_t request_bytes,
    std::uint64_t response_bytes
);

} // namespace dli::common