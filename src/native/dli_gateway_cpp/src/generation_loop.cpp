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

namespace dli::gateway {

namespace {

std::string make_request_id() {
    return "gateway-loop-native-routing-request";
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
    const std::vector<std::int64_t>& shape
) {
    std::ostringstream out;

    out
        << "{"
        << "\"request_id\":\"" << dli::common::json_escape(request_id) << "\","
        << "\"token_index\":" << token_index << ","
        << "\"generation_mode\":\"" << dli::common::json_escape(generation_mode) << "\","
        << "\"tensor\":{"
        << "\"dtype\":\"" << dli::common::json_escape(dtype) << "\","
        << "\"shape\":" << shape_json(shape) << ","
        << "\"byte_order\":\"little\""
        << "},"
        << "\"feature_flags\":{"
        << "\"kv_cache_enabled\":true"
        << "},"
        << "\"sampling\":{"
        << "\"temperature\":0.0,"
        << "\"top_k\":1,"
        << "\"top_p\":null"
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
    trace.stage_metadata_json = stage_result.response_frame.metadata_json;
    trace.stage_tensor_bytes = stage_result.response_frame.tensor_bytes.size();
    trace.error_body = stage_result.error_body;
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
        << "\"stage_metadata\":\"" << dli::common::json_escape(step.stage_metadata_json) << "\","
        << "\"stage_tensor_bytes\":" << step.stage_tensor_bytes << ","
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
        const TokenizedPrompt tokenized = tokenizer_.tokenize(prompt);
        result.prompt_token_ids = tokenized.token_ids;

        const int decode_steps = std::max(0, max_new_tokens);

        StageClient client(config_.first_stage_url);

        dli::common::DliFrame prefill;
        prefill.metadata_json = request_metadata_json(
            result.request_id,
            0,
            "prefill",
            "int64",
            {1, static_cast<std::int64_t>(tokenized.token_ids.size())}
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

        // Prefill may or may not return a token depending on final runtime policy.
        // Decode below is authoritative for generated tokens.
        (void) terminal_prefill_response;

        int last_token_id =
            tokenized.token_ids.empty()
                ? 1
                : static_cast<int>(tokenized.token_ids.back());

        for (int i = 0; i < decode_steps; ++i) {
            dli::common::DliFrame decode;

            decode.metadata_json = request_metadata_json(
                result.request_id,
                i,
                "decode",
                "int64",
                {1, 1}
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
        }

        result.generated_text = tokenizer_.detokenize(result.generated_token_ids);
        result.generated_token_count = static_cast<int>(result.generated_token_ids.size());
        result.termination_reason = "terminal_next_token";
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