#include "dli/gateway/generation_loop.hpp"

#include "dli/common/json_escape.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <utility>

namespace dli::gateway {

namespace {

std::string make_request_id() {
    // Deterministic for now. Later this should become UUID-like.
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

} // namespace

GenerationLoop::GenerationLoop(GenerationLoopConfig config)
    : config_(std::move(config)) {}

GenerationLoopResult GenerationLoop::run_stub_generation(int max_new_tokens) const {
    const auto start = std::chrono::steady_clock::now();

    GenerationLoopResult result;
    result.request_id = make_request_id();

    const int decode_steps = std::max(0, max_new_tokens);

    StageClient client(config_.first_stage_url);

    // Prefill step: fake token-id placeholder with shape [1, 4].
    StageForwardStubRequest prefill;
    prefill.request_id = result.request_id;
    prefill.token_index = 0;
    prefill.generation_mode = "prefill";
    prefill.dtype = "int64";
    prefill.shape = {1, 4};
    prefill.kv_cache_enabled = true;
    prefill.tensor_bytes = {
        1, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0,
        3, 0, 0, 0, 0, 0, 0, 0,
        4, 0, 0, 0, 0, 0, 0, 0,
    };

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
        result.steps.push_back(make_step_trace(i, "decode", decode_response));
    }

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
        << "\"prompt\":\"\","
        << "\"generated_text\":\"\","
        << "\"assistant_message\":\"\","
        << "\"generated_token_ids\":[],"
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