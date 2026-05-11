#include "dli/gateway/stage_client.hpp"

#include "dli/common/http_client.hpp"
#include "dli/common/protocol.hpp"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dli::gateway {

namespace {

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

std::string build_stub_metadata_json(const StageForwardStubRequest& request) {
    std::ostringstream out;
    out
        << "{"
        << "\"request_id\":\"" << request.request_id << "\","
        << "\"token_index\":" << request.token_index << ","
        << "\"generation_mode\":\"" << request.generation_mode << "\","
        << "\"tensor\":{"
        << "\"dtype\":\"" << request.dtype << "\","
        << "\"shape\":" << shape_json(request.shape) << ","
        << "\"byte_order\":\"little\""
        << "},"
        << "\"feature_flags\":{"
        << "\"kv_cache_enabled\":" << (request.kv_cache_enabled ? "true" : "false")
        << "},"
        << "\"sampling\":{"
        << "\"temperature\":0.0,"
        << "\"top_k\":null,"
        << "\"top_p\":null"
        << "}"
        << "}";

    return out.str();
}

} // namespace

StageClient::StageClient(std::string first_stage_url)
    : first_stage_url_(std::move(first_stage_url)) {}

StageClientResult StageClient::forward_stub_frame(
    const StageForwardStubRequest& request
) const {
    dli::common::DliFrame request_frame;
    request_frame.metadata_json = build_stub_metadata_json(request);
    request_frame.tensor_bytes = request.tensor_bytes;

    const std::vector<std::uint8_t> request_body =
        dli::common::encode_frame(request_frame);

    const dli::common::HttpClientResponse http_response =
        dli::common::http_post_binary(first_stage_url_, request_body);

    StageClientResult result;
    result.http_status = http_response.status_code;
    result.http_reason = http_response.reason;

    if (http_response.status_code >= 200 && http_response.status_code < 300) {
        result.response_frame = dli::common::decode_frame(http_response.body);
    }

    return result;
}

} // namespace dli::gateway