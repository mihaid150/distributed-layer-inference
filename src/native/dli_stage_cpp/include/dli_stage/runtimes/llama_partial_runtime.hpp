#pragma once

#include "dli_stage/runtime.hpp"

#include <string>

namespace dli_stage {

struct LlamaPartialRuntimeConfig {
    std::string model_path;
    int stage_id = 0;
};

class LlamaPartialRuntime final : public StageRuntime {
public:
    explicit LlamaPartialRuntime(LlamaPartialRuntimeConfig config);

    RuntimeResponse forward(const RuntimeRequest& request) override;

    std::string backend_name() const override;

    const LlamaPartialRuntimeConfig& config() const;

private:
    LlamaPartialRuntimeConfig config_;
};

} // namespace dli_stage