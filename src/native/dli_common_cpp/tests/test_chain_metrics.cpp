#include "dli/common/metrics.hpp"

#include <cassert>
#include <cstddef>
#include <string>

namespace {

std::size_t count_occurrences(
    const std::string& text,
    const std::string& needle
) {
    std::size_t count = 0;
    std::size_t pos = 0;

    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }

    return count;
}

void test_terminal_fallback() {
    const std::string metadata =
        "{\"ok\":true,"
        "\"metrics\":{\"compute_time_ms\":31.5},"
        "\"compute_time_ms\":31.5,"
        "\"input_tensor_bytes\":8192,"
        "\"output_tensor_bytes\":0,"
        "\"stage_id\":4,"
        "\"is_final_stage\":true}";

    const dli::common::ChainMetrics metrics =
        dli::common::extract_chain_metrics_from_metadata(metadata);

    assert(metrics.compute_ms == 31.5);
    assert(metrics.stage_count == 1);
}

void test_merge_and_deduplicate() {
    const std::string downstream =
        "{\"ok\":true,"
        "\"compute_time_ms\":31.0,"
        "\"chain_compute_time_ms\":30.0,"
        "\"chain_compute_time_ms\":31.0,"
        "\"chain_rpc_wall_time_ms\":31.0,"
        "\"chain_true_comm_ms\":0.0,"
        "\"chain_transport_payload_bytes\":100,"
        "\"chain_hops\":1,"
        "\"stage_id\":4}";

    dli::common::StageMetrics local;
    local.compute_time_ms = 20.0;

    const std::string merged = dli::common::merge_chain_metrics(
        downstream,
        local,
        70.0,
        1000,
        2000
    );

    assert(count_occurrences(merged, "\"chain_metrics\"") == 1);
    assert(count_occurrences(merged, "\"chain_compute_time_ms\"") == 1);
    assert(count_occurrences(merged, "\"chain_rpc_wall_time_ms\"") == 1);
    assert(count_occurrences(merged, "\"chain_true_comm_ms\"") == 1);
    assert(count_occurrences(merged, "\"chain_transport_payload_bytes\"") == 1);
    assert(count_occurrences(merged, "\"chain_stage_count\"") == 1);
    assert(count_occurrences(merged, "\"chain_hops\"") == 1);

    const dli::common::ChainMetrics metrics =
        dli::common::extract_chain_metrics_from_metadata(merged);

    assert(metrics.compute_ms == 51.0);
    assert(metrics.rpc_wall_ms == 90.0);
    assert(metrics.true_comm_ms == 39.0);
    assert(metrics.transport_payload_bytes == 3100);
    assert(metrics.stage_count == 2);
}

} // namespace

int main() {
    test_terminal_fallback();
    test_merge_and_deduplicate();
    return 0;
}