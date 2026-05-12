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

struct LlamaModelMetadata {
    bool model_loaded = false;

    std::string model_path;
    std::string architecture;
    std::string name;

    int n_ctx_train = 0;
    int n_embd = 0;
    int n_layer = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int n_vocab = 0;
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

    const LlamaModelMetadata& model_metadata() const;

private:
    LlamaPartialRuntimeConfig config_;
    LlamaModelMetadata model_metadata_;

    llama_model* model_ = nullptr;
    const llama_vocab* vocab_ = nullptr;

    std::string backend_metadata_json() const;
};

} // namespace dli_stage