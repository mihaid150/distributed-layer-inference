#pragma once

#include "dli_stage/runtime.hpp"

#include <string>

struct llama_model;
struct llama_vocab;

namespace dli_stage {

struct LlamaPartialRuntimeConfig {
    std::string model_path;
    int stage_id = 0;
};

class LlamaPartialRuntime final : public StageRuntime {
public:
    explicit LlamaPartialRuntime(LlamaPartialRuntimeConfig config);

    ~LlamaPartialRuntime() override;

    LlamaPartialRuntime(const LlamaPartialRuntime&) = delete;
    LlamaPartialRuntime& operator=(const LlamaPartialRuntime&) = delete;

    RuntimeResponse forward(const RuntimeRequest& request) override;

    std::string backend_name() const override;

    const LlamaPartialRuntimeConfig& config() const;

private:
    LlamaPartialRuntimeConfig config_;

    llama_model* model_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    bool model_loaded_ = false;
};

} // namespace dli_stage