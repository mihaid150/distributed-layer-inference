#pragma once

#include "dli/common/protocol.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dli::gateway {

struct StageClientResult {
    int http_status = 0;
    std::string http_reason;
    dli::common::DliFrame response_frame;
    std::string error_body;
};

struct StageForwardStubRequest {
    std::string request_id = "gateway-stub-request";
    int token_index = 0;
    std::string generation_mode = "decode";
    std::string dtype = "float16";
    std::vector<std::int64_t> shape = {1, 1, 4};
    bool kv_cache_enabled = true;
    std::vector<std::uint8_t> tensor_bytes = {1, 2, 3, 4, 5, 6, 7, 8};
};

class StageClient {
public:
    explicit StageClient(std::string first_stage_url);

    StageClientResult forward_stub_frame(const StageForwardStubRequest& request) const;

    StageClientResult forward_frame(
        const std::string& stage_url,
        const dli::common::DliFrame& frame
    ) const;

private:
    std::string first_stage_url_;
};

} // namespace dli::gateway