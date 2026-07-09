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

    // Per-request resource usage sampled from /proc during this forward call.
    // CPU percentages are measured across the request's wall-clock window;
    // context-switch and IO figures are deltas accumulated over the same window.
    double process_cpu_percent = 0.0;
    double system_cpu_percent = 0.0;
    std::uint64_t process_threads = 0;
    std::uint64_t ctx_switches_voluntary = 0;
    std::uint64_t ctx_switches_involuntary = 0;
    std::uint64_t io_read_bytes_delta = 0;
    std::uint64_t io_write_bytes_delta = 0;
    std::uint64_t io_read_count_delta = 0;
    std::uint64_t io_write_count_delta = 0;

    // Compute breakdown for this forward call (ggml matmul vs scalar attention)
    // plus the ggml CPU worker-thread count, so CPU optimizations are measurable.
    double matmul_ms = 0.0;
    double attention_ms = 0.0;
    int ggml_threads = 0;
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