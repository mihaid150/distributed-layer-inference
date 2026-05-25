// src/native/dli_common_cpp/src/metrics.cpp

#include "dli/common/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace dli::common {
namespace {

std::optional<double> extract_number_field(
    const std::string& json,
    const std::string& key
) {
    const std::regex pattern(
        "\"" + key + R"("\s*:\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))"
    );

    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return std::nullopt;
    }

    try {
        return std::stod(match[1].str());
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::uint64_t> extract_u64_field(
    const std::string& json,
    const std::string& key
) {
    const auto value = extract_number_field(json, key);
    if (!value || *value < 0.0) {
        return std::nullopt;
    }

    return static_cast<std::uint64_t>(*value);
}

std::optional<double> extract_number_aliases(
    const std::string& json,
    const std::vector<std::string>& keys
) {
    for (const std::string& key : keys) {
        if (const auto value = extract_number_field(json, key)) {
            return value;
        }
    }

    return std::nullopt;
}

std::optional<std::uint64_t> extract_u64_aliases(
    const std::string& json,
    const std::vector<std::string>& keys
) {
    for (const std::string& key : keys) {
        if (const auto value = extract_u64_field(json, key)) {
            return value;
        }
    }

    return std::nullopt;
}

std::string remove_object_field(std::string json, const std::string& key) {
    const std::regex middle_pattern(
        R"(,\s*")" + key + R"("\s*:\s*\{[^{}]*\})"
    );
    json = std::regex_replace(json, middle_pattern, "");

    const std::regex first_pattern(
        R"(")" + key + R"("\s*:\s*\{[^{}]*\}\s*,)"
    );
    json = std::regex_replace(json, first_pattern, "");

    const std::regex only_pattern(
        R"(")" + key + R"("\s*:\s*\{[^{}]*\})"
    );
    json = std::regex_replace(json, only_pattern, "");

    return json;
}

std::string remove_number_field(std::string json, const std::string& key) {
    const std::regex middle_pattern(
        R"(,\s*")" + key + R"("\s*:\s*-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    );
    json = std::regex_replace(json, middle_pattern, "");

    const std::regex first_pattern(
        R"(")" + key + R"("\s*:\s*-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*,)"
    );
    json = std::regex_replace(json, first_pattern, "");

    const std::regex only_pattern(
        R"(")" + key + R"("\s*:\s*-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    );
    json = std::regex_replace(json, only_pattern, "");

    return json;
}

std::string remove_chain_metric_fields(std::string json) {
    json = remove_object_field(std::move(json), "chain_metrics");

    const std::vector<std::string> number_keys = {
        "chain_compute_time_ms",
        "chain_rpc_wall_time_ms",
        "chain_true_comm_ms",
        "chain_transport_payload_bytes",
        "chain_stage_count",
        "chain_hops",

        // compact aliases
        "ccm",
        "crw",
        "ctc",
        "cpb",
        "csc"
    };

    for (const std::string& key : number_keys) {
        json = remove_number_field(std::move(json), key);
    }

    return json;
}

std::string chain_metrics_json(const ChainMetrics& metrics) {
    std::ostringstream out;
    out
        << "\"chain_metrics\":{"
        << "\"compute_ms\":" << metrics.compute_ms << ","
        << "\"rpc_wall_ms\":" << metrics.rpc_wall_ms << ","
        << "\"true_comm_ms\":" << metrics.true_comm_ms << ","
        << "\"transport_payload_bytes\":" << metrics.transport_payload_bytes << ","
        << "\"stage_count\":" << metrics.stage_count
        << "},"
        << "\"chain_compute_time_ms\":" << metrics.compute_ms << ","
        << "\"chain_rpc_wall_time_ms\":" << metrics.rpc_wall_ms << ","
        << "\"chain_true_comm_ms\":" << metrics.true_comm_ms << ","
        << "\"chain_transport_payload_bytes\":" << metrics.transport_payload_bytes << ","
        << "\"chain_stage_count\":" << metrics.stage_count << ","
        << "\"chain_hops\":" << metrics.stage_count;

    return out.str();
}

} // namespace

ChainMetrics local_chain_metrics_from_stage(const StageMetrics& metrics) {
    ChainMetrics out;
    out.compute_ms = metrics.compute_time_ms;
    out.rpc_wall_ms = metrics.rpc_wall_time_ms > 0.0
        ? metrics.rpc_wall_time_ms
        : metrics.compute_time_ms;
    out.true_comm_ms = std::max(0.0, out.rpc_wall_ms - out.compute_ms);
    out.transport_payload_bytes = metrics.input_tensor_bytes + metrics.output_tensor_bytes;
    out.stage_count = 1;
    return out;
}

bool metadata_has_chain_metrics(const std::string& metadata_json) {
    return
        metadata_json.find("\"chain_metrics\"") != std::string::npos ||
        metadata_json.find("\"chain_compute_time_ms\"") != std::string::npos ||
        metadata_json.find("\"ccm\"") != std::string::npos;
}

ChainMetrics extract_chain_metrics_from_metadata(const std::string& metadata_json) {
    ChainMetrics out;

    if (const auto value = extract_number_aliases(
            metadata_json,
            {"compute_ms", "chain_compute_time_ms", "ccm"})) {
        out.compute_ms = *value;
    }

    if (const auto value = extract_number_aliases(
            metadata_json,
            {"rpc_wall_ms", "chain_rpc_wall_time_ms", "crw"})) {
        out.rpc_wall_ms = *value;
    }

    if (const auto value = extract_number_aliases(
            metadata_json,
            {"true_comm_ms", "chain_true_comm_ms", "ctc"})) {
        out.true_comm_ms = *value;
    }

    if (const auto value = extract_u64_aliases(
            metadata_json,
            {"transport_payload_bytes", "chain_transport_payload_bytes", "cpb"})) {
        out.transport_payload_bytes = *value;
    }

    if (const auto value = extract_u64_aliases(
            metadata_json,
            {"stage_count", "chain_stage_count", "chain_hops", "csc"})) {
        out.stage_count = *value;
    }

    // Fallback for terminal-stage metadata that has only normal StageMetrics.
    if (!metadata_has_chain_metrics(metadata_json)) {
        if (const auto value = extract_number_field(metadata_json, "compute_time_ms")) {
            out.compute_ms = *value;
        }
        if (const auto value = extract_number_field(metadata_json, "rpc_wall_time_ms")) {
            out.rpc_wall_ms = *value;
        }
        if (const auto value = extract_number_field(metadata_json, "true_comm_ms")) {
            out.true_comm_ms = *value;
        }
        if (const auto value = extract_u64_field(metadata_json, "input_tensor_bytes")) {
            out.transport_payload_bytes += *value;
        }
        if (const auto value = extract_u64_field(metadata_json, "output_tensor_bytes")) {
            out.transport_payload_bytes += *value;
        }
        out.stage_count = 1;
    }

    if (out.rpc_wall_ms <= 0.0 && out.compute_ms > 0.0) {
        out.rpc_wall_ms = out.compute_ms;
    }

    if (out.true_comm_ms <= 0.0) {
        out.true_comm_ms = std::max(0.0, out.rpc_wall_ms - out.compute_ms);
    }

    if (out.stage_count == 0) {
        out.stage_count = 1;
    }

    return out;
}

std::string put_chain_metrics_in_metadata(
    std::string metadata_json,
    const ChainMetrics& metrics
) {
    metadata_json = remove_chain_metric_fields(std::move(metadata_json));

    const std::size_t pos = metadata_json.rfind('}');
    if (pos == std::string::npos) {
        return metadata_json;
    }

    const bool needs_comma = pos > 0 && metadata_json[pos - 1] != '{';
    metadata_json.insert(
        pos,
        std::string(needs_comma ? "," : "") + chain_metrics_json(metrics)
    );

    return metadata_json;
}

ChainMetrics merge_chain_metrics_values(
    const StageMetrics& local_metrics,
    const ChainMetrics& downstream_metrics,
    double downstream_rpc_wall_ms,
    std::uint64_t request_bytes,
    std::uint64_t response_bytes
) {
    const ChainMetrics local = local_chain_metrics_from_stage(local_metrics);

    ChainMetrics out;
    out.compute_ms = local.compute_ms + downstream_metrics.compute_ms;

    out.rpc_wall_ms =
        local.compute_ms + std::max(0.0, downstream_rpc_wall_ms);

    out.true_comm_ms =
        std::max(0.0, out.rpc_wall_ms - out.compute_ms);

    out.transport_payload_bytes =
        downstream_metrics.transport_payload_bytes +
        request_bytes +
        response_bytes;

    out.stage_count =
        std::max<std::uint64_t>(1, downstream_metrics.stage_count) + 1;

    return out;
}

std::string merge_chain_metrics(
    std::string downstream_metadata,
    const StageMetrics& local_metrics,
    double downstream_rpc_wall_ms,
    std::uint64_t request_bytes,
    std::uint64_t response_bytes
) {
    const ChainMetrics downstream =
        extract_chain_metrics_from_metadata(downstream_metadata);

    const ChainMetrics merged =
        merge_chain_metrics_values(
            local_metrics,
            downstream,
            downstream_rpc_wall_ms,
            request_bytes,
            response_bytes
        );

    return put_chain_metrics_in_metadata(
        std::move(downstream_metadata),
        merged
    );
}

} // namespace dli::common