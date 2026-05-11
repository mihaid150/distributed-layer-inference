#include "dli/gateway/stage_client.hpp"

#include "dli/common/http_client.hpp"
#include "dli/common/protocol.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dli::gateway {

StageClient::StageClient(std::string first_stage_url)
    : first_stage_url_(std::move(first_stage_url)) {}

StageClientResult StageClient::forward_stub_frame() const {
    dli::common::DliFrame request_frame;

    request_frame.metadata_json =
        R"({"request_id":"gateway-stub-request","token_index":0,"generation_mode":"decode","tensor":{"dtype":"float16","shape":[1,1,4],"byte_order":"little"},"feature_flags":{"kv_cache_enabled":true},"sampling":{"temperature":0.0,"top_k":null,"top_p":null}})";

    request_frame.tensor_bytes = {
        static_cast<std::uint8_t>(1),
        static_cast<std::uint8_t>(2),
        static_cast<std::uint8_t>(3),
        static_cast<std::uint8_t>(4),
        static_cast<std::uint8_t>(5),
        static_cast<std::uint8_t>(6),
        static_cast<std::uint8_t>(7),
        static_cast<std::uint8_t>(8),
    };

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