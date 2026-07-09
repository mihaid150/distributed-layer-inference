// Parity test: the ggml/NEON attention path (DLI_GGML_ATTENTION) must produce
// the same context vectors as the scalar reference. This guards the opt-in fast
// path so it can only be trusted once it matches the proven implementation,
// across plain MHA and grouped-query (GQA) head configurations.

#include "dli_stage/runtimes/llama_cpu_executor.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct Case {
    int n_head;
    int n_head_kv;
    int head_dim;
    int seq;
};

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return worst;
}

bool run_case(const Case& c, std::mt19937& rng) {
    dli_stage::LlamaExecutorHyperparams hp;
    hp.n_head = c.n_head;
    hp.n_head_kv = c.n_head_kv;
    hp.head_dim = c.head_dim;
    hp.hidden_size = c.n_head * c.head_dim;

    const int q_dim = c.n_head * c.head_dim;
    const int kv_dim = c.n_head_kv * c.head_dim;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(static_cast<std::size_t>(q_dim));
    std::vector<float> k_cache(static_cast<std::size_t>(c.seq) * kv_dim);
    std::vector<float> v_cache(static_cast<std::size_t>(c.seq) * kv_dim);
    for (float& x : q) x = dist(rng);
    for (float& x : k_cache) x = dist(rng);
    for (float& x : v_cache) x = dist(rng);

    const std::vector<float> scalar =
        dli_stage::LlamaCpuExecutor::compute_attention_scalar(hp, q, k_cache, v_cache, c.seq);
    const std::vector<float> ggml =
        dli_stage::LlamaCpuExecutor::compute_attention_ggml(hp, q, k_cache, v_cache, c.seq);

    if (scalar.size() != ggml.size()) {
        std::printf("FAIL size mismatch (%zu vs %zu)\n", scalar.size(), ggml.size());
        return false;
    }

    // Also exercise the views path with V capacity > required_seq, i.e. the
    // strided transposed-V layout production decoding keeps persistently. This
    // guards the non-contiguous nb[] strides on the V tensor view.
    const int pad_cap = c.seq + 5;
    std::vector<float> v_t_padded(
        static_cast<std::size_t>(pad_cap) * kv_dim, 0.0f);
    for (int kvh = 0; kvh < c.n_head_kv; ++kvh) {
        for (int d = 0; d < c.head_dim; ++d) {
            for (int pos = 0; pos < c.seq; ++pos) {
                v_t_padded[static_cast<std::size_t>(pos) +
                           static_cast<std::size_t>(pad_cap) * d +
                           static_cast<std::size_t>(pad_cap) * c.head_dim * kvh] =
                    v_cache[static_cast<std::size_t>(pos) * kv_dim +
                            static_cast<std::size_t>(kvh) * c.head_dim + d];
            }
        }
    }
    const std::vector<float> ggml_strided =
        dli_stage::LlamaCpuExecutor::compute_attention_ggml_views(
            hp, q, k_cache.data(), v_t_padded.data(), pad_cap, c.seq);

    const double diff = max_abs_diff(scalar, ggml);
    const double diff_strided = max_abs_diff(scalar, ggml_strided);
    const double worst = std::max(diff, diff_strided);
    const double tol = 2.0e-3;
    std::printf(
        "case n_head=%d n_head_kv=%d head_dim=%d seq=%d -> "
        "max_abs_diff=%.3e (strided=%.3e) %s\n",
        c.n_head, c.n_head_kv, c.head_dim, c.seq, diff, diff_strided,
        worst <= tol ? "OK" : "FAIL"
    );
    return worst <= tol;
}

// Mirrors the persistent transposed-V maintenance in LlamaCpuExecutor::attention
// (grow re-transposes [0, seq) from the source cache; the steady-state path
// appends only the current position) and checks the ggml views output matches
// the scalar reference at every decode step. This guards the stateful append /
// capacity-grow index math that the model-free static tests cannot reach.
bool run_incremental_case(const Case& c, std::mt19937& rng) {
    dli_stage::LlamaExecutorHyperparams hp;
    hp.n_head = c.n_head;
    hp.n_head_kv = c.n_head_kv;
    hp.head_dim = c.head_dim;
    hp.hidden_size = c.n_head * c.head_dim;

    const int q_dim = c.n_head * c.head_dim;
    const int kv_dim = c.n_head_kv * c.head_dim;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> k_cache;  // source layout, grown one position per step
    std::vector<float> v_cache;  // source layout (for scalar reference)
    std::vector<float> v_t;      // transposed, capacity-strided (production layout)
    int v_t_capacity = 0;

    double worst = 0.0;
    for (int step = 0; step < c.seq; ++step) {
        const int position = step;
        const int required_seq = position + 1;

        std::vector<float> q(static_cast<std::size_t>(q_dim));
        std::vector<float> k_new(static_cast<std::size_t>(kv_dim));
        std::vector<float> v_new(static_cast<std::size_t>(kv_dim));
        for (float& x : q) x = dist(rng);
        for (float& x : k_new) x = dist(rng);
        for (float& x : v_new) x = dist(rng);

        k_cache.insert(k_cache.end(), k_new.begin(), k_new.end());
        v_cache.insert(v_cache.end(), v_new.begin(), v_new.end());

        if (v_t_capacity < required_seq) {
            const int new_cap = std::max(required_seq, std::max(v_t_capacity * 2, 16));
            std::vector<float> grown(
                static_cast<std::size_t>(new_cap) * kv_dim, 0.0f);
            for (int kvh = 0; kvh < c.n_head_kv; ++kvh)
                for (int d = 0; d < c.head_dim; ++d)
                    for (int pos = 0; pos < required_seq; ++pos)
                        grown[static_cast<std::size_t>(pos) +
                              static_cast<std::size_t>(new_cap) * d +
                              static_cast<std::size_t>(new_cap) * c.head_dim * kvh] =
                            v_cache[static_cast<std::size_t>(pos) * kv_dim +
                                    static_cast<std::size_t>(kvh) * c.head_dim + d];
            v_t.swap(grown);
            v_t_capacity = new_cap;
        } else {
            for (int kvh = 0; kvh < c.n_head_kv; ++kvh)
                for (int d = 0; d < c.head_dim; ++d)
                    v_t[static_cast<std::size_t>(position) +
                        static_cast<std::size_t>(v_t_capacity) * d +
                        static_cast<std::size_t>(v_t_capacity) * c.head_dim * kvh] =
                        v_new[static_cast<std::size_t>(kvh) * c.head_dim + d];
        }

        const std::vector<float> scalar =
            dli_stage::LlamaCpuExecutor::compute_attention_scalar(
                hp, q, k_cache, v_cache, required_seq);
        const std::vector<float> ggml =
            dli_stage::LlamaCpuExecutor::compute_attention_ggml_views(
                hp, q, k_cache.data(), v_t.data(), v_t_capacity, required_seq);
        worst = std::max(worst, max_abs_diff(scalar, ggml));
    }

    const double tol = 2.0e-3;
    std::printf(
        "incremental n_head=%d n_head_kv=%d head_dim=%d steps=%d -> "
        "max_abs_diff=%.3e %s\n",
        c.n_head, c.n_head_kv, c.head_dim, c.seq, worst, worst <= tol ? "OK" : "FAIL"
    );
    return worst <= tol;
}

} // namespace

int main() {
    std::mt19937 rng(12345);

    const std::vector<Case> cases = {
        {4, 4, 8, 1},    // single position (prefill of length 1 / first decode)
        {4, 4, 16, 7},   // plain multi-head attention
        {8, 2, 16, 5},   // grouped-query attention (4 query heads per kv head)
        {6, 3, 4, 17},   // GQA, odd sequence length
        {2, 1, 32, 64},  // longer sequence
        {32, 4, 64, 1},  // TinyLlama geometry, first token
        {32, 4, 64, 256},// TinyLlama geometry, mid-length decode
    };

    bool ok = true;
    for (const Case& c : cases) {
        ok = run_case(c, rng) && ok;
    }

    // Incremental decode cases: crosses the capacity-grow boundary (cap starts at
    // 16 and doubles), so both the grow and steady-state append paths are hit.
    const std::vector<Case> incremental_cases = {
        {4, 4, 16, 20},   // plain MHA, grows past the initial capacity of 16
        {8, 2, 16, 40},   // GQA across two grows
        {32, 4, 64, 33},  // TinyLlama geometry across a grow
    };
    for (const Case& c : incremental_cases) {
        ok = run_incremental_case(c, rng) && ok;
    }

    if (!ok) {
        std::printf("attention parity test FAILED\n");
        return 1;
    }
    std::printf("attention parity test PASSED\n");
    return 0;
}
