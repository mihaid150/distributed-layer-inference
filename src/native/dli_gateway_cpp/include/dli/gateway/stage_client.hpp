#pragma once

#include "dli/common/protocol.hpp"

#include <string>

namespace dli::gateway {

struct StageClientResult {
    int http_status = 0;
    std::string http_reason;
    dli::common::DliFrame response_frame;
};

class StageClient {
public:
    explicit StageClient(std::string first_stage_url);

    StageClientResult forward_stub_frame() const;

private:
    std::string first_stage_url_;
};

} // namespace dli::gateway