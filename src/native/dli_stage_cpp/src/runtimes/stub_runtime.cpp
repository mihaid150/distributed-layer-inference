#include "dli_stage/runtimes/stub_runtime.hpp"

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
    response.next_token_id = -1;
    response.output_metadata_json = request.input_metadata_json;
    response.output_tensor = request.input_tensor;

    const auto end = std::chrono::steady_clock::now();

    response.metrics.backend = backend_name();
    response.metrics.status = "stub_echo";
    response.metrics.compute_time_ms = elapsed_ms(start, end);
    response.metrics.input_tensor_bytes = request.input_tensor.bytes.size();
    response.metrics.output_tensor_bytes = response.output_tensor.bytes.size();

    return response;
}

std::string StubRuntime::backend_name() const {
    return "stub";
}

} // namespace dli_stage