#pragma once

#include "dli_stage/runtime.hpp"

#include <memory>
#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct llama_model;
struct llama_vocab;

namespace dli_stage {

class LlamaCpuExecutor;

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

struct PartitionKvCacheState {
    int seq_len = 0;
    bool valid = true;
};

struct PartitionKvCacheStep {
    int seq_before = 0;
    int seq_after = 0;
    std::uint64_t bytes = 0;
    bool valid = true;
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

    PartitionKvCacheState kv_cache_;

    std::unique_ptr<LlamaCpuExecutor> executor_;

    llama_model* model_ = nullptr;
    const llama_vocab* vocab_ = nullptr;

    gguf_context* tensor_gguf_ctx_ = nullptr;
    ggml_context* tensor_data_ctx_ = nullptr;

    double model_load_ms_ = 0.0;

    std::string backend_metadata_json() const;

    RuntimeResponse forward_source_partition(const RuntimeRequest& request);
    RuntimeResponse forward_intermediate_partition(const RuntimeRequest& request);
    RuntimeResponse forward_terminal_partition(const RuntimeRequest& request);

    PartitionKvCacheStep update_kv_cache_for_request(const RuntimeRequest& request);
    std::uint64_t estimate_kv_cache_bytes(int seq_len) const;

    void load_raw_tensor_context();
    dli::common::TensorBuffer execute_token_embedding_only(
        const dli::common::TensorBuffer& token_ids
    ) const;
};

} // namespace dli_stage