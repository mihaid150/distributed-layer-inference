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

} // namespace dli::common