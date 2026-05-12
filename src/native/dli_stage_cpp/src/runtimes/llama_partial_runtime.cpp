#include "dli_stage/runtimes/llama_partial_runtime.hpp"

#include "dli/common/tensor.hpp"

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace dli_stage {

namespace {

double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
) {
    const auto duration = std::chrono::duration<double, std::milli>(end - start);
    return duration.count();
}

void validate_model_path_if_present(const std::string& model_path) {
    if (model_path.empty()) {
        return;
    }

    const std::filesystem::path path(model_path);

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            "LlamaPartialRuntime model path does not exist: " + model_path
        );
    }

    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "LlamaPartialRuntime model path is not a regular file: " + model_path
        );
    }
}

} // namespace

LlamaPartialRuntime::LlamaPartialRuntime(LlamaPartialRuntimeConfig config)
    : config_(std::move(config)) {
    validate_model_path_if_present(config_.model_path);
}

RuntimeResponse LlamaPartialRuntime::forward(const RuntimeRequest& request) {
    const auto start = std::chrono::steady_clock::now();

    RuntimeResponse response;
    response.is_final_stage = false;

    // Skeleton phase:
    // This does not execute llama.cpp layers yet. It only proves that the
    // HTTP server can switch to a llama-backed runtime without changing routes.
    if (request.generation_mode == "decode") {
        response.next_token_id = 2000 + request.token_index;
    } else {
        response.next_token_id = -1;
    }

    response.output_metadata_json = request.input_metadata_json;
    response.output_tensor = request.input_tensor;

    const auto end = std::chrono::steady_clock::now();

    const int input_seq = dli::common::infer_sequence_length_from_shape(
        request.input_tensor.metadata.shape
    );

    response.metrics.backend = backend_name();
    response.metrics.status = "llama_partial_skeleton_echo";
    response.metrics.compute_time_ms = elapsed_ms(start, end);
    response.metrics.true_comm_ms = 0.0;
    response.metrics.rpc_wall_time_ms = 0.0;
    response.metrics.input_tensor_bytes = request.input_tensor.bytes.size();
    response.metrics.output_tensor_bytes = response.output_tensor.bytes.size();
    response.metrics.stage_input_token_count = input_seq;
    response.metrics.stage_output_token_count = input_seq;
    response.metrics.kv_cache_step_valid = true;

    return response;
}

std::string LlamaPartialRuntime::backend_name() const {
    return "llama.cpp-partial-skeleton";
}

const LlamaPartialRuntimeConfig& LlamaPartialRuntime::config() const {
    return config_;
}

} // namespace dli_stage