#pragma once

#include "dli/common/tensor.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace dli_stage {

struct LlamaExecutorHyperparams {
    std::string architecture = "llama";

    int hidden_size = 0;
    int ffn_size = 0;

    int n_head = 0;
    int n_head_kv = 0;
    int head_dim = 0;

    int vocab_size = 0;

    float rms_eps = 1.0e-5f;
    float rope_theta = 10000.0f;
};

struct LlamaExecutorResult {
    dli::common::TensorBuffer hidden_states;
    int next_token_id = -1;
};

struct LayerKvCache {
    int seq_len = 0;
    std::vector<float> k;
    std::vector<float> v;
};

class LlamaCpuExecutor {
public:
    LlamaCpuExecutor(
        ggml_context* tensor_context,
        LlamaExecutorHyperparams hyperparams
    );

    dli::common::TensorBuffer execute_source(
        const dli::common::TensorBuffer& token_ids,
        const std::vector<int>& layers,
        const std::string& generation_mode,
        int kv_seq_before,
        const std::string& output_activation_precision
    );

    dli::common::TensorBuffer execute_intermediate(
        const dli::common::TensorBuffer& hidden_states,
        const std::vector<int>& layers,
        const std::string& generation_mode,
        int kv_seq_before,
        const std::string& output_activation_precision
    );

    int execute_terminal(
        const dli::common::TensorBuffer& hidden_states,
        const std::vector<int>& layers,
        const std::string& generation_mode,
        int kv_seq_before,
        bool owns_norm,
        bool owns_lm_head
    );

    std::uint64_t kv_cache_bytes() const;

private:
    ggml_context* tensor_context_ = nullptr;
    LlamaExecutorHyperparams hp_;

    std::unordered_map<int, LayerKvCache> layer_cache_;

    ggml_tensor* require_tensor(const std::string& name) const;

    std::vector<float> tensor_row_to_float(
        const ggml_tensor* tensor,
        int64_t row
    ) const;

    std::vector<float> tensor_vector_to_float(
        const std::string& name,
        int64_t expected_size
    ) const;

    std::vector<float> matvec(
        const std::string& tensor_name,
        const std::vector<float>& x
    ) const;

    std::vector<float> rms_norm(
        const std::vector<float>& x,
        const std::vector<float>& weight
    ) const;

    void apply_rope(
        std::vector<float>& q,
        std::vector<float>& k,
        int position
    ) const;

    std::vector<float> attention(
        int layer_id,
        const std::vector<float>& q,
        const std::vector<float>& k,
        const std::vector<float>& v,
        int position
    );

    std::vector<float> run_layer(
        int layer_id,
        const std::vector<float>& input,
        int seq_len,
        int kv_seq_before
    );

    dli::common::TensorBuffer float_hidden_to_tensor(
        const std::vector<float>& hidden,
        int batch,
        int seq_len,
        const std::string& activation_precision
    ) const;

    std::vector<float> tensor_to_hidden_f32(
        const dli::common::TensorBuffer& tensor
    ) const;

    std::vector<std::int64_t> tensor_to_token_ids(
        const dli::common::TensorBuffer& tensor
    ) const;

    int argmax_lm_head(const std::vector<float>& last_hidden) const;
};

} // namespace dli_stage
