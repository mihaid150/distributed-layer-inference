#include "dli/gateway/generation_loop.hpp"

#include "dli/common/json_escape.hpp"
#include "dli/common/metadata.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <utility>

namespace dli::gateway {

namespace {

int next_token_id_from_stage_response(
    const StageClientResult& stage_result,
    int fallback_token_id
) {
    if (stage_result.response_frame.metadata_json.empty()) {
        return fallback_token_id;
    }

    const dli::common::ParsedRequestMetadata parsed =
        dli::common::parse_request_metadata(stage_result.response_frame.metadata_json);

    if (parsed.has_next_token_id && parsed.next_token_id >= 0) {
        return parsed.next_token_id;
    }

    return fallback_token_id;
}

std::string make_request_id() {
    return "gateway-loop-stub-request";
}

double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
) {
    const auto duration = std::chrono::duration<double, std::milli>(end - start);
    return duration.count();
}

GenerationStepTrace make_step_trace(
    int token_index,
    const std::string& mode,
    const StageClientResult& stage_result
) {
    GenerationStepTrace trace;
    trace.token_index = token_index;
    trace.generation_mode = mode;
    trace.stage_http_status = stage_result.http_status;
    trace.stage_http_reason = stage_result.http_reason;
    trace.stage_metadata_json = stage_result.response_frame.metadata_json;
    trace.stage_tensor_bytes = stage_result.response_frame.tensor_bytes.size();
    return trace;
}

std::string step_json(const GenerationStepTrace& step) {
    std::ostringstream out;
    out
        << "{"
        << "\"token_index\":" << step.token_index << ","
        << "\"generation_mode\":\"" << dli::common::json_escape(step.generation_mode) << "\","
        << "\"stage_http_status\":" << step.stage_http_status << ","
        << "\"stage_http_reason\":\"" << dli::common::json_escape(step.stage_http_reason) << "\","
        << "\"stage_metadata\":\"" << dli::common::json_escape(step.stage_metadata_json) << "\","
        << "\"stage_tensor_bytes\":" << step.stage_tensor_bytes
        << "}";

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

    const TokenizedPrompt tokenized = tokenizer_.tokenize(prompt);
    result.prompt_token_ids = tokenized.token_ids;

    const int decode_steps = std::max(0, max_new_tokens);

    StageClient client(config_.first_stage_url);

    StageForwardStubRequest prefill;
    prefill.request_id = result.request_id;
    prefill.token_index = 0;
    prefill.generation_mode = "prefill";
    prefill.dtype = "int64";
    prefill.shape = {1, static_cast<std::int64_t>(tokenized.token_ids.size())};
    prefill.kv_cache_enabled = true;
    prefill.tensor_bytes = int64_tokens_to_little_endian_bytes(tokenized.token_ids);

    const StageClientResult prefill_response = client.forward_stub_frame(prefill);
    result.steps.push_back(make_step_trace(0, "prefill", prefill_response));

    for (int i = 0; i < decode_steps; ++i) {
        StageForwardStubRequest decode;
        decode.request_id = result.request_id;
        decode.token_index = i;
        decode.generation_mode = "decode";
        decode.dtype = "float16";
        decode.shape = {1, 1, 4};
        decode.kv_cache_enabled = true;
        decode.tensor_bytes = {
            static_cast<unsigned char>(1 + i),
            static_cast<unsigned char>(2 + i),
            static_cast<unsigned char>(3 + i),
            static_cast<unsigned char>(4 + i),
            static_cast<unsigned char>(5 + i),
            static_cast<unsigned char>(6 + i),
            static_cast<unsigned char>(7 + i),
            static_cast<unsigned char>(8 + i),
        };

        const StageClientResult decode_response = client.forward_stub_frame(decode);

        const int fallback_token_id = 1000 + i;
        const int next_token_id = next_token_id_from_stage_response(
            decode_response,
            fallback_token_id
        );

        result.generated_token_ids.push_back(next_token_id);
        result.steps.push_back(make_step_trace(i, "decode", decode_response));
    }

    result.generated_text = tokenizer_.detokenize(result.generated_token_ids);

    const auto end = std::chrono::steady_clock::now();

    result.generated_token_count = decode_steps;
    result.total_latency_ms = elapsed_ms(start, end);

    if (result.total_latency_ms > 0.0) {
        result.tokens_per_second =
            static_cast<double>(result.generated_token_count) / (result.total_latency_ms / 1000.0);
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
        << "\"ok\":true,"
        << "\"service\":\"dli-gateway-cpp\","
        << "\"runtime\":\"cpp-native-stub\","
        << "\"status\":\"stub_generate_loop\","
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
        << "\"termination_reason\":\"stub\","
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