#pragma once

#include "dli/common/tensor.hpp"

#include <cstdint>
#include <random>
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

// Token sampling configuration for the terminal partition. When `enabled` is
// false (or temperature <= 0) the executor falls back to deterministic greedy
// argmax decoding, preserving the previous behaviour.
struct LlamaSamplingParams {
    bool enabled = false;
    float temperature = 1.0f;
    int top_k = 0;       // <= 0 disables top-k filtering
    float top_p = 1.0f;  // >= 1.0 disables nucleus filtering

    // Optional RNG seed for reproducible sampling. Applied once at the start of a
    // sequence (kv_seq_before == 0). has_seed=false keeps the rolling random
    // stream; seed >= 0 gives reproducible output; seed < 0 reseeds from a random
    // device each sequence (explicit diversity).
    bool has_seed = false;
    long long seed = 0;
};

struct LlamaExecutorResult {
    dli::common::TensorBuffer hidden_states;
    int next_token_id = -1;
};

struct LayerKvCache {
    int seq_len = 0;
    std::vector<float> k;
    std::vector<float> v;

    // Transposed V (positions contiguous in dim 0), kept persistent for the ggml
    // views attention path and appended one column per token, so attention no
    // longer re-gathers the whole cache every step. v_t_capacity is the allocated
    // position stride used by the tensor view (>= seq_len).
    std::vector<float> v_t;
    int v_t_capacity = 0;
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
        bool owns_lm_head,
        const LlamaSamplingParams& sampling
    );

    std::uint64_t kv_cache_bytes() const;

    // Per-forward compute profiling. reset_profile() zeroes the accumulators at
    // the start of a forward; the *_ms() getters report how much wall time went
    // to matmuls vs attention so the CPU optimizations can be measured.
    void reset_profile();
    double profile_matmul_ms() const;
    double profile_attention_ms() const;

    // ggml CPU worker-thread count actually in use (shared process-wide).
    static int configured_threads();

    // Attention compute over a layer's cached K/V. Two interchangeable
    // implementations: a scalar reference and a ggml/NEON path. Exposed as
    // static so a parity test can assert they produce identical results before
    // the ggml path (gated by DLI_GGML_ATTENTION) is trusted in production.
    // k_cache/v_cache are laid out [pos * (n_head_kv*head_dim) + kvh*head_dim + d].
    static std::vector<float> compute_attention_scalar(
        const LlamaExecutorHyperparams& hp,
        const std::vector<float>& q,
        const std::vector<float>& k_cache,
        const std::vector<float>& v_cache,
        int required_seq
    );
    static std::vector<float> compute_attention_ggml(
        const LlamaExecutorHyperparams& hp,
        const std::vector<float>& q,
        const std::vector<float>& k_cache,
        const std::vector<float>& v_cache,
        int required_seq
    );

    // Core ggml attention that references the KV cache by *view* (no per-token
    // re-gather): k_src is the K cache in layout
    // [pos*(n_head_kv*head_dim) + kvh*head_dim + d]; v_t_src is the transposed V
    // cache laid out [pos + v_capacity*d + v_capacity*head_dim*kvh] so positions
    // are contiguous in dim 0 as ggml_mul_mat requires. v_capacity is the
    // allocated position stride (>= required_seq). Numerically identical to the
    // scalar/contiguous paths; exposed static so the parity test can pad
    // v_capacity > required_seq and confirm the strided view is correct.
    static std::vector<float> compute_attention_ggml_views(
        const LlamaExecutorHyperparams& hp,
        const std::vector<float>& q,
        const float* k_src,
        const float* v_t_src,
        int v_capacity,
        int required_seq
    );

private:
    ggml_context* tensor_context_ = nullptr;
    LlamaExecutorHyperparams hp_;

    std::unordered_map<int, LayerKvCache> layer_cache_;

    std::mt19937 rng_;

    // Compute profiling accumulators (mutable so const compute helpers can add).
    mutable double matmul_ms_ = 0.0;
    mutable double attention_ms_ = 0.0;

    // Cached RoPE inverse-frequency table (depends only on head_dim/theta), so
    // the rotation loop no longer calls std::pow per element per token.
    mutable std::vector<float> rope_inv_freq_;

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

    int sample_lm_head(
        const std::vector<float>& last_hidden,
        const LlamaSamplingParams& sampling
    );

    int sample_from_logits(
        std::vector<float> logits,
        const LlamaSamplingParams& sampling
    );
};

} // namespace dli_stage
