#include "dli_stage/runtimes/stub_runtime.hpp"

#include "dli/common/tensor.hpp"

#include <chrono>

namespace dli_stage {

namespace {

double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
) {
    const auto duration = std::chrono::duration<double, std::milli>(end - start);
    return duration.count();
}

} // namespace

RuntimeResponse StubRuntime::forward(const RuntimeRequest& request) {
    const auto start = std::chrono::steady_clock::now();

    RuntimeResponse response;
    response.is_final_stage = false;

    if (request.generation_mode == "decode") {
        response.next_token_id = 1000 + request.token_index;
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
    response.metrics.status = "stub_echo";
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

std::string StubRuntime::backend_name() const {
    return "stub";
}

} // namespace dli_stage