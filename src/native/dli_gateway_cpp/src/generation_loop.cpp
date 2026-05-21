#include "dli/gateway/generation_loop.hpp"

#include "dli/common/json_escape.hpp"
#include "dli/common/metadata.hpp"
#include "dli/common/protocol.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <regex>
#include <atomic>
#include <cctype>
#include <cstdlib>

namespace dli::gateway {

namespace {


std::string trim_copy(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }

    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return value.substr(begin, end - begin);
}

bool contains_generation_control_marker(const std::string& text) {
    return
        text.find("</s>") != std::string::npos ||
        text.find("<s>") != std::string::npos ||
        text.find("<|user|>") != std::string::npos ||
        text.find("<|assistant|>") != std::string::npos ||
        text.find("<|system|>") != std::string::npos;
}

std::string format_prompt_for_generation(
    const std::string& prompt,
    bool apply_chat_template
) {
    if (!apply_chat_template) {
        return prompt;
    }

    if (
        prompt.find("<|user|>") != std::string::npos ||
        prompt.find("<|assistant|>") != std::string::npos
    ) {
        return prompt;
    }

    std::ostringstream out;
    out
        << "<|user|>\n"
        << prompt
        << "</s>\n<|assistant|>\n";
    return out.str();
}

std::string cleanup_generated_text(std::string text) {
    const std::vector<std::string> cut_markers = {
        "<|user|>",
        "<|system|>",
        "<|assistant|>"
    };

    for (const std::string& marker : cut_markers) {
        const std::size_t pos = text.find(marker);
        if (pos != std::string::npos) {
            text = text.substr(0, pos);
        }
    }

    const std::vector<std::string> erase_markers = {
        "</s>",
        "<s>"
    };

    for (const std::string& marker : erase_markers) {
        std::size_t pos = 0;
        while ((pos = text.find(marker, pos)) != std::string::npos) {
            text.erase(pos, marker.size());
        }
    }

    return trim_copy(text);
}

bool token_text_is_stop_marker(const std::string& token_text) {
    return contains_generation_control_marker(token_text);
}


bool metadata_level_is_compact(const std::string& metadata_level) {
    return metadata_level == "compact" || metadata_level == "minimal";
}

bool extract_number_field(
    const std::string& json,
    const std::string& key,
    double& out
) {
    const std::regex pattern(
        "\"" + key + R"("\s*:\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))"
    );

    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return false;
    }

    try {
        out = std::stod(match[1].str());
        return true;
    } catch (...) {
        return false;
    }
}

bool extract_u64_field(
    const std::string& json,
    const std::string& key,
    std::uint64_t& out
) {
    double value = 0.0;

    if (!extract_number_field(json, key, value)) {
        return false;
    }

    if (value < 0.0) {
        return false;
    }

    out = static_cast<std::uint64_t>(value);
    return true;
}

void aggregate_stage_metrics(
    GenerationLoopResult& result,
    const std::string& metadata_json
) {
    double d = 0.0;
    std::uint64_t u = 0;

    bool has_chain_compute = false;
    bool has_chain_rpc = false;
    bool has_chain_true_comm = false;

    if (extract_number_field(metadata_json, "chain_compute_time_ms", d)) {
        result.aggregate_metrics.compute_ms += d;
        has_chain_compute = true;
    }

    if (extract_number_field(metadata_json, "chain_rpc_wall_time_ms", d)) {
        result.aggregate_metrics.rpc_wall_ms += d;
        has_chain_rpc = true;
    }

    if (extract_number_field(metadata_json, "chain_true_comm_ms", d)) {
        result.aggregate_metrics.true_comm_ms += d;
        has_chain_true_comm = true;
    }

    if (extract_u64_field(metadata_json, "chain_transport_payload_bytes", u)) {
        result.aggregate_metrics.transport_payload_bytes += u;
    }

    if (!has_chain_compute && extract_number_field(metadata_json, "compute_time_ms", d)) {
        result.aggregate_metrics.compute_ms += d;
    }

    if (!has_chain_rpc && extract_number_field(metadata_json, "rpc_wall_time_ms", d)) {
        result.aggregate_metrics.rpc_wall_ms += d;
    }

    if (!has_chain_true_comm && extract_number_field(metadata_json, "true_comm_ms", d)) {
        result.aggregate_metrics.true_comm_ms += d;
    }

    if (extract_u64_field(metadata_json, "input_tensor_bytes", u)) {
        result.aggregate_metrics.tensor_bytes_in += u;
    }

    if (extract_u64_field(metadata_json, "output_tensor_bytes", u)) {
        result.aggregate_metrics.tensor_bytes_out += u;
    }

    if (extract_number_field(metadata_json, "model_load_ms", d)) {
        result.aggregate_metrics.model_load_ms += d;
    }

    if (extract_u64_field(metadata_json, "kv_cache_bytes", u)) {
        result.aggregate_metrics.kv_cache_bytes =
            std::max(result.aggregate_metrics.kv_cache_bytes, u);
    }

    if (extract_u64_field(metadata_json, "memory_rss_mb", u)) {
        result.aggregate_metrics.memory_rss_mb =
            std::max(result.aggregate_metrics.memory_rss_mb, u);
    }

    if (extract_u64_field(metadata_json, "memory_cgroup_current_mb", u)) {
        result.aggregate_metrics.memory_cgroup_current_mb =
            std::max(result.aggregate_metrics.memory_cgroup_current_mb, u);
    }

    if (extract_u64_field(metadata_json, "memory_cgroup_limit_mb", u)) {
        result.aggregate_metrics.memory_cgroup_limit_mb =
            std::max(result.aggregate_metrics.memory_cgroup_limit_mb, u);
    }

    if (extract_number_field(metadata_json, "memory_cgroup_percent", d)) {
        result.aggregate_metrics.memory_cgroup_percent =
            std::max(result.aggregate_metrics.memory_cgroup_percent, d);
    }

    if (extract_u64_field(metadata_json, "model_file_size_mb", u)) {
        result.aggregate_metrics.model_file_size_mb =
            std::max(result.aggregate_metrics.model_file_size_mb, u);
    }

    if (extract_u64_field(metadata_json, "session_count", u)) {
        result.aggregate_metrics.session_count =
            std::max(result.aggregate_metrics.session_count, u);
    }

    if (extract_u64_field(metadata_json, "session_kv_cache_bytes", u)) {
        result.aggregate_metrics.session_kv_cache_bytes =
            std::max(result.aggregate_metrics.session_kv_cache_bytes, u);
    }
}
    
std::string make_request_id() {
    static std::atomic<std::uint64_t> counter{0};

    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

    return
        "gateway-loop-native-routing-request-" +
        std::to_string(static_cast<long long>(nanos)) +
        "-" +
        std::to_string(++counter);
}


double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
) {
    const auto duration = std::chrono::duration<double, std::milli>(end - start);
    return duration.count();
}

std::string shape_json(const std::vector<std::int64_t>& shape) {
    std::ostringstream out;
    out << "[";

    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << shape[i];
    }

    out << "]";
    return out.str();
}

std::string int_vector_json(const std::vector<int>& values) {
    std::ostringstream out;
    out << "[";

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }

    out << "]";
    return out.str();
}

std::string i64_vector_json(const std::vector<std::int64_t>& values) {
    std::ostringstream out;
    out << "[";

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }

    out << "]";
    return out.str();
}

std::vector<std::uint8_t> int64_tokens_to_bytes(
    const std::vector<std::int64_t>& tokens
) {
    std::vector<std::uint8_t> bytes(tokens.size() * sizeof(std::int64_t));

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const std::uint64_t value = static_cast<std::uint64_t>(tokens[i]);

        for (int b = 0; b < 8; ++b) {
            bytes[i * 8 + static_cast<std::size_t>(b)] =
                static_cast<std::uint8_t>((value >> static_cast<unsigned>(b * 8)) & 0xffu);
        }
    }

    return bytes;
}

std::string request_metadata_json(
    const std::string& request_id,
    int token_index,
    const std::string& generation_mode,
    const std::string& dtype,
    const std::vector<std::int64_t>& shape,
    const std::string& activation_precision,
    bool persistent_sessions_enabled,
    bool native_stage_chaining_enabled,
    int min_new_tokens,
    bool apply_chat_template,
    const std::string& metadata_level
) {
    std::ostringstream out;

    const std::int64_t sequence_length =
        shape.size() >= 2 ? shape[1] : 0;

    if (metadata_level_is_compact(metadata_level)) {
        out
            << "{"
            << "\"rid\":\"" << dli::common::json_escape(request_id) << "\","
            << "\"ti\":" << token_index << ","
            << "\"gm\":\"" << dli::common::json_escape(generation_mode) << "\","
            << "\"dt\":\"" << dli::common::json_escape(dtype) << "\","
            << "\"sh\":" << shape_json(shape) << ","
            << "\"bo\":\"little\","
            << "\"ap\":\"" << dli::common::json_escape(activation_precision) << "\","
            << "\"kv\":true,"
            << "\"ps\":" << (persistent_sessions_enabled ? "true" : "false") << ","
            << "\"nsc\":" << (native_stage_chaining_enabled ? "true" : "false") << ","
            << "\"m\":" << min_new_tokens
            << "}";
        return out.str();
    }

    out
        << "{"
        << "\"request_id\":\"" << dli::common::json_escape(request_id) << "\","
        << "\"token_index\":" << token_index << ","
        << "\"generation_mode\":\"" << dli::common::json_escape(generation_mode) << "\","
        << "\"sequence_length\":" << sequence_length << ","
        << "\"tensor\":{"
        << "\"dtype\":\"" << dli::common::json_escape(dtype) << "\","
        << "\"shape\":" << shape_json(shape) << ","
        << "\"byte_order\":\"little\""
        << "},"
        << "\"feature_flags\":{"
        << "\"kv_cache_enabled\":true,"
        << "\"transport_mode\":\"binary_octet_stream\","
        << "\"activation_precision\":\"" << dli::common::json_escape(activation_precision) << "\","
        << "\"persistent_sessions_enabled\":" << (persistent_sessions_enabled ? "true" : "false") << ","
        << "\"native_stage_chaining_enabled\":" << (native_stage_chaining_enabled ? "true" : "false") << ","
        << "\"metadata_level\":\"" << dli::common::json_escape(metadata_level) << "\","
        << "\"apply_chat_template\":" << (apply_chat_template ? "true" : "false")
        << "},"
        << "\"sampling\":{"
        << "\"temperature\":0.0,"
        << "\"top_k\":1,"
        << "\"top_p\":null,"
        << "\"min_new_tokens\":" << min_new_tokens
        << "}"
        << "}";

    return out.str();
}


bool is_terminal_partition(const PartitionNodeConfig& partition) {
    return partition.next_stage_url.empty() || partition.components.lm_head;
}

std::string stage_url_for_partition(
    const GenerationLoopConfig& config,
    std::size_t index
) {
    if (index == 0) {
        return config.first_stage_url;
    }

    if (index >= config.partitions.size()) {
        throw std::runtime_error("partition index out of range");
    }

    const std::string& previous_next_url =
        config.partitions[index - 1].next_stage_url;

    if (previous_next_url.empty()) {
        throw std::runtime_error(
            "cannot route to partition " +
            config.partitions[index].partition_id +
            " because previous partition has empty next_stage_url"
        );
    }

    return previous_next_url;
}

GenerationStepTrace make_step_trace(
    int token_index,
    const std::string& mode,
    const PartitionNodeConfig& partition,
    const std::string& stage_url,
    const StageClientResult& stage_result
) {
    GenerationStepTrace trace;
    trace.token_index = token_index;
    trace.generation_mode = mode;
    trace.partition_id = partition.partition_id;
    trace.stage_id = partition.stage_id;
    trace.stage_url = stage_url;
    trace.stage_http_status = stage_result.http_status;
    trace.stage_http_reason = stage_result.http_reason;
    trace.stage_http_elapsed_ms = stage_result.elapsed_ms;
    trace.stage_metadata_json = stage_result.response_frame.metadata_json;
    trace.stage_tensor_bytes = stage_result.response_frame.tensor_bytes.size();
    trace.request_body_bytes = stage_result.request_body_bytes;
    trace.response_body_bytes = stage_result.response_body_bytes;
    trace.transport_payload_bytes = stage_result.transport_payload_bytes;
    trace.error_body = stage_result.error_body;

    double compute_ms = 0.0;
    if (extract_number_field(trace.stage_metadata_json, "compute_time_ms", compute_ms)) {
        trace.estimated_true_comm_ms = std::max(0.0, trace.stage_http_elapsed_ms - compute_ms);
    }

    return trace;
}


int next_token_id_from_terminal_response(
    const StageClientResult& stage_result
) {
    if (stage_result.response_frame.metadata_json.empty()) {
        throw std::runtime_error("terminal partition returned empty metadata");
    }

    const dli::common::ParsedRequestMetadata parsed =
        dli::common::parse_request_metadata(stage_result.response_frame.metadata_json);

    if (!parsed.has_next_token_id || parsed.next_token_id < 0) {
        throw std::runtime_error(
            "terminal partition did not return a valid next_token_id"
        );
    }

    return parsed.next_token_id;
}

StageClientResult forward_through_partition_graph(
    const GenerationLoopConfig& config,
    const StageClient& client,
    const dli::common::DliFrame& input_frame,
    int token_index,
    const std::string& generation_mode,
    GenerationLoopResult& result
) {
    if (config.partitions.empty()) {
        throw std::runtime_error("gateway partition graph is empty");
    }


    if (config.native_stage_chaining_enabled) {
        const PartitionNodeConfig& first_partition = config.partitions.front();
        const std::string stage_url = config.first_stage_url;

        StageClientResult stage_result = client.forward_frame(stage_url, input_frame);

        result.steps.push_back(
            make_step_trace(
                token_index,
                generation_mode,
                first_partition,
                stage_url,
                stage_result
            )
        );

        if (stage_result.http_status < 200 || stage_result.http_status >= 300) {
            throw std::runtime_error(
                "native chained stage call failed with HTTP " +
                std::to_string(stage_result.http_status) +
                ": " +
                stage_result.error_body
            );
        }

        aggregate_stage_metrics(result, stage_result.response_frame.metadata_json);
        return stage_result;
    }

    dli::common::DliFrame current_frame = input_frame;
    StageClientResult last_result;

    for (std::size_t i = 0; i < config.partitions.size(); ++i) {
        const PartitionNodeConfig& partition = config.partitions[i];
        const std::string stage_url = stage_url_for_partition(config, i);

        last_result = client.forward_frame(stage_url, current_frame);

        result.steps.push_back(
            make_step_trace(
                token_index,
                generation_mode,
                partition,
                stage_url,
                last_result
            )
        );

        if (last_result.http_status < 200 || last_result.http_status >= 300) {
            throw std::runtime_error(
                "partition " +
                partition.partition_id +
                " returned HTTP " +
                std::to_string(last_result.http_status) +
                ": " +
                last_result.error_body
            );
        }

        current_frame = last_result.response_frame;

        aggregate_stage_metrics(result, current_frame.metadata_json);
        result.aggregate_metrics.rpc_wall_ms += last_result.elapsed_ms;
        result.aggregate_metrics.transport_payload_bytes += last_result.transport_payload_bytes;

        double stage_compute_ms = 0.0;
        if (extract_number_field(current_frame.metadata_json, "compute_time_ms", stage_compute_ms)) {
            result.aggregate_metrics.true_comm_ms +=
                std::max(0.0, last_result.elapsed_ms - stage_compute_ms);
        }

        if (is_terminal_partition(partition)) {
            return last_result;
        }
    }

    throw std::runtime_error("partition graph has no terminal partition");
}

std::string step_json(const GenerationStepTrace& step) {
    std::ostringstream out;
    out
        << "{"
        << "\"token_index\":" << step.token_index << ","
        << "\"generation_mode\":\"" << dli::common::json_escape(step.generation_mode) << "\","
        << "\"partition_id\":\"" << dli::common::json_escape(step.partition_id) << "\","
        << "\"stage_id\":" << step.stage_id << ","
        << "\"stage_url\":\"" << dli::common::json_escape(step.stage_url) << "\","
        << "\"stage_http_status\":" << step.stage_http_status << ","
        << "\"stage_http_reason\":\"" << dli::common::json_escape(step.stage_http_reason) << "\","
        << "\"stage_http_elapsed_ms\":" << step.stage_http_elapsed_ms << ","
        << "\"stage_metadata\":\"" << dli::common::json_escape(step.stage_metadata_json) << "\","
        << "\"stage_tensor_bytes\":" << step.stage_tensor_bytes << ","
        << "\"request_body_bytes\":" << step.request_body_bytes << ","
        << "\"response_body_bytes\":" << step.response_body_bytes << ","
        << "\"transport_payload_bytes\":" << step.transport_payload_bytes << ","
        << "\"estimated_true_comm_ms\":" << step.estimated_true_comm_ms << ","
        << "\"error_body\":\"" << dli::common::json_escape(step.error_body) << "\""
        << "}";

    return out.str();
}

} // namespace

GenerationLoop::GenerationLoop(
    GenerationLoopConfig config,
    const Tokenizer& tokenizer
)
    : config_(std::move(config)),
      tokenizer_(tokenizer) {}

GenerationLoopResult GenerationLoop::run_stub_generation(
    const std::string& prompt,
    int max_new_tokens
) const {
    const auto start = std::chrono::steady_clock::now();

    GenerationLoopResult result;
    result.request_id = make_request_id();
    result.prompt = prompt;
    result.tokenizer_backend = tokenizer_.backend_name();

    try {
        const std::string prompt_for_model =
            format_prompt_for_generation(prompt, config_.apply_chat_template);

        const TokenizedPrompt tokenized = tokenizer_.tokenize(prompt_for_model);
        result.prompt_token_ids = tokenized.token_ids;

        const int decode_steps = std::max(0, max_new_tokens);

        StageClient client(config_.first_stage_url);

        dli::common::DliFrame prefill;
        prefill.metadata_json = request_metadata_json(
            result.request_id,
            0,
            "prefill",
            "int64",
            {1, static_cast<std::int64_t>(tokenized.token_ids.size())},
            config_.activation_precision,
            config_.persistent_sessions_enabled,
            config_.native_stage_chaining_enabled,
            config_.min_new_tokens,
            config_.apply_chat_template,
            config_.metadata_level
        );
        prefill.tensor_bytes = int64_tokens_to_bytes(tokenized.token_ids);

        StageClientResult terminal_prefill_response =
            forward_through_partition_graph(
                config_,
                client,
                prefill,
                0,
                "prefill",
                result
            );

        int last_token_id =
            tokenized.token_ids.empty()
                ? 1
                : static_cast<int>(tokenized.token_ids.back());

        int decode_start = 0;
        result.termination_reason = "max_new_tokens";

        auto should_stop_after = [&](int token_id) -> bool {
            if (static_cast<int>(result.generated_token_ids.size()) < config_.min_new_tokens) {
                return false;
            }

            const std::string token_text =
                tokenizer_.detokenize(std::vector<int>{token_id});

            return token_text_is_stop_marker(token_text);
        };

        if (decode_steps > 0) {
            const int first_generated_token =
                next_token_id_from_terminal_response(terminal_prefill_response);

            result.generated_token_ids.push_back(first_generated_token);
            last_token_id = first_generated_token;
            decode_start = 1;

            if (should_stop_after(first_generated_token)) {
                result.termination_reason = "stop_marker";
                decode_start = decode_steps;
            }
        }

        for (int i = decode_start; i < decode_steps; ++i) {
            dli::common::DliFrame decode;

            decode.metadata_json = request_metadata_json(
                result.request_id,
                i,
                "decode",
                "int64",
                {1, 1},
                config_.activation_precision,
                config_.persistent_sessions_enabled,
                config_.native_stage_chaining_enabled,
                config_.min_new_tokens,
                config_.apply_chat_template,
                config_.metadata_level
            );

            decode.tensor_bytes = int64_tokens_to_bytes(
                {static_cast<std::int64_t>(last_token_id)}
            );

            const StageClientResult terminal_decode_response =
                forward_through_partition_graph(
                    config_,
                    client,
                    decode,
                    i,
                    "decode",
                    result
                );

            const int next_token_id =
                next_token_id_from_terminal_response(terminal_decode_response);

            result.generated_token_ids.push_back(next_token_id);
            last_token_id = next_token_id;

            if (should_stop_after(next_token_id)) {
                result.termination_reason = "stop_marker";
                break;
            }
        }

        result.generated_text = cleanup_generated_text(
            tokenizer_.detokenize(result.generated_token_ids)
        );
        result.generated_token_count = static_cast<int>(result.generated_token_ids.size());
    } catch (const std::exception& exc) {
        result.ok = false;
        result.error = exc.what();
        result.termination_reason = "error";
    }

    const auto end = std::chrono::steady_clock::now();

    result.total_latency_ms = elapsed_ms(start, end);

    if (result.total_latency_ms > 0.0) {
        result.tokens_per_second =
            static_cast<double>(result.generated_token_count) /
            (result.total_latency_ms / 1000.0);
    }

    return result;
}


std::string generation_loop_result_json(
    const GenerationLoopResult& result,
    std::size_t request_body_bytes
) {
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":" << (result.ok ? "true" : "false") << ","
        << "\"service\":\"dli-gateway-cpp\","
        << "\"runtime\":\"cpp-native-routing\","
        << "\"status\":\"" << (result.ok ? "partition_graph_complete" : "partition_graph_error") << "\","
        << "\"request_id\":\"" << dli::common::json_escape(result.request_id) << "\","
        << "\"prompt\":\"" << dli::common::json_escape(result.prompt) << "\","
        << "\"tokenizer_backend\":\"" << dli::common::json_escape(result.tokenizer_backend) << "\","
        << "\"prompt_token_count\":" << result.prompt_token_ids.size() << ","
        << "\"prompt_token_ids\":" << i64_vector_json(result.prompt_token_ids) << ","
        << "\"generated_text\":\"" << dli::common::json_escape(result.generated_text) << "\","
        << "\"assistant_message\":\"" << dli::common::json_escape(result.generated_text) << "\","
        << "\"generated_token_ids\":" << int_vector_json(result.generated_token_ids) << ","
        << "\"generated_token_count\":" << result.generated_token_count << ","
        << "\"total_latency_ms\":" << result.total_latency_ms << ","
        << "\"tokens_per_second\":" << result.tokens_per_second << ","
        << "\"aggregate_metrics\":{"
        << "\"compute_ms\":" << result.aggregate_metrics.compute_ms << ","
        << "\"rpc_wall_ms\":" << result.aggregate_metrics.rpc_wall_ms << ","
        << "\"true_comm_ms\":" << result.aggregate_metrics.true_comm_ms << ","
        << "\"tensor_bytes_in\":" << result.aggregate_metrics.tensor_bytes_in << ","
        << "\"tensor_bytes_out\":" << result.aggregate_metrics.tensor_bytes_out << ","
        << "\"transport_payload_bytes\":" << result.aggregate_metrics.transport_payload_bytes << ","
        << "\"transport_payload_mebibytes\":" << (static_cast<double>(result.aggregate_metrics.transport_payload_bytes) / (1024.0 * 1024.0)) << ","
        << "\"model_load_ms\":" << result.aggregate_metrics.model_load_ms << ","
        << "\"kv_cache_bytes\":" << result.aggregate_metrics.kv_cache_bytes << ","
        << "\"memory_rss_mb\":" << result.aggregate_metrics.memory_rss_mb << ","
        << "\"memory_cgroup_current_mb\":" << result.aggregate_metrics.memory_cgroup_current_mb << ","
        << "\"memory_cgroup_limit_mb\":" << result.aggregate_metrics.memory_cgroup_limit_mb << ","
        << "\"memory_cgroup_percent\":" << result.aggregate_metrics.memory_cgroup_percent << ","
        << "\"model_file_size_mb\":" << result.aggregate_metrics.model_file_size_mb << ","
        << "\"session_count\":" << result.aggregate_metrics.session_count << ","
        << "\"session_kv_cache_bytes\":" << result.aggregate_metrics.session_kv_cache_bytes
        << "},"
        << "\"termination_reason\":\"" << dli::common::json_escape(result.termination_reason) << "\","
        << "\"error\":\"" << dli::common::json_escape(result.error) << "\","
        << "\"request_body_bytes\":" << request_body_bytes << ","
        << "\"step_count\":" << result.steps.size() << ","
        << "\"steps\":[";

    for (std::size_t i = 0; i < result.steps.size(); ++i) {
        if (i > 0) {
            out << ",";
        }

        out << step_json(result.steps[i]);
    }

    out << "]}";
    return out.str();
}

} // namespace dli::gateway
