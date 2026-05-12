#pragma once

#include "dli_stage/runtime.hpp"

#include <string>
#include <vector>

struct llama_model;
struct llama_vocab;

namespace dli_stage {

struct LlamaPartialRuntimeConfig {
    std::string model_path;
    int stage_id = 0;

    std::vector<int> expected_layers;
    bool expected_owns_embedding = false;
    bool expected_owns_norm = false;
    bool expected_owns_lm_head = false;
    std::string expected_next_stage_url;
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

struct LlamaShardMetadata {
    bool shard_loaded = false;

    std::string format;
    int format_version = 0;

    std::string partition_id;
    int stage_id = 0;

    std::vector<int> layers;

    bool owns_embedding = false;
    bool owns_norm = false;
    bool owns_lm_head = false;

    std::string next_partition_id;
    std::string next_stage_url;

    int hidden_size = -1;
    std::string source_model;
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

    const LlamaShardMetadata& shard_metadata() const;

private:
    LlamaPartialRuntimeConfig config_;
    LlamaModelMetadata model_metadata_;
    LlamaShardMetadata shard_metadata_;

    llama_model* model_ = nullptr;
    const llama_vocab* vocab_ = nullptr;

    std::string backend_metadata_json() const;

    RuntimeResponse forward_terminal_partition(const RuntimeRequest& request);
    RuntimeResponse forward_non_terminal_partition(const RuntimeRequest& request);
};

} // namespace dli_stage