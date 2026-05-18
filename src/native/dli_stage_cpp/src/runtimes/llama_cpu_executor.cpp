#include "dli_stage/runtimes/llama_cpu_executor.hpp"

#include "dli/common/dli2_abi.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace dli_stage {

namespace {

std::int64_t read_i64_le(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset
) {
    if (offset + sizeof(std::int64_t) > bytes.size()) {
        throw std::runtime_error("cannot read int64 past token tensor");
    }

    std::uint64_t value = 0;

    for (int i = 0; i < 8; ++i) {
        value |=
            static_cast<std::uint64_t>(bytes[offset + static_cast<std::size_t>(i)])
            << static_cast<unsigned>(8 * i);
    }

    return static_cast<std::int64_t>(value);
}

float read_f32_le(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset
) {
    if (offset + sizeof(float) > bytes.size()) {
        throw std::runtime_error("cannot read float32 past hidden tensor");
    }

    std::uint32_t raw = 0;
    raw |= static_cast<std::uint32_t>(bytes[offset + 0]);
    raw |= static_cast<std::uint32_t>(bytes[offset + 1]) << 8u;
    raw |= static_cast<std::uint32_t>(bytes[offset + 2]) << 16u;
    raw |= static_cast<std::uint32_t>(bytes[offset + 3]) << 24u;

    float value = 0.0f;
    std::memcpy(&value, &raw, sizeof(float));
    return value;
}

void write_f32_le(
    std::vector<std::uint8_t>& bytes,
    std::size_t offset,
    float value
) {
    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(float));

    bytes[offset + 0] = static_cast<std::uint8_t>(raw & 0xffu);
    bytes[offset + 1] = static_cast<std::uint8_t>((raw >> 8u) & 0xffu);
    bytes[offset + 2] = static_cast<std::uint8_t>((raw >> 16u) & 0xffu);
    bytes[offset + 3] = static_cast<std::uint8_t>((raw >> 24u) & 0xffu);
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

std::string layer_prefix(int layer_id) {
    return "blk." + std::to_string(layer_id) + ".";
}

void compute_single_node_graph(
    ggml_context* ctx,
    ggml_tensor* result
) {
    ggml_cgraph* graph = ggml_new_graph(ctx);

    if (graph == nullptr) {
        throw std::runtime_error("failed to allocate GGML graph");
    }

    ggml_build_forward_expand(graph, result);

    ggml_backend_t backend = ggml_backend_cpu_init();

    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }

    /*
     * In this executor we create temporary tensors with no_alloc=false,
     * so tensor buffers are already present in the GGML context.
     *
     * Newer llama.cpp/ggml versions expose graph execution through
     * ggml_backend_graph_compute instead of ggml_graph_compute_with_ctx
     * or ggml_graph_plan.
     */
    const enum ggml_status status =
        ggml_backend_graph_compute(backend, graph);

    ggml_backend_free(backend);

    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(
            "GGML backend graph compute failed with status=" +
            std::to_string(static_cast<int>(status))
        );
    }
}

std::vector<float> read_f32_tensor_flat(
    const ggml_tensor* tensor,
    std::size_t expected_count
) {
    if (tensor == nullptr || tensor->data == nullptr) {
        throw std::runtime_error("cannot read null GGML tensor as float32");
    }

    if (tensor->type != GGML_TYPE_F32) {
        throw std::runtime_error("expected GGML_TYPE_F32 output tensor");
    }

    const float* data = static_cast<const float*>(tensor->data);

    std::vector<float> out(expected_count, 0.0f);

    std::copy(
        data,
        data + expected_count,
        out.begin()
    );

    return out;
}

} // namespace

LlamaCpuExecutor::LlamaCpuExecutor(
    ggml_context* tensor_context,
    LlamaExecutorHyperparams hyperparams
)
    : tensor_context_(tensor_context),
      hp_(std::move(hyperparams)) {
    if (tensor_context_ == nullptr) {
        throw std::runtime_error("LlamaCpuExecutor requires non-null ggml_context");
    }

    if (hp_.architecture != "llama") {
        throw std::runtime_error(
            "LlamaCpuExecutor currently supports only GGUF architecture=llama, got: " +
            hp_.architecture
        );
    }

    if (
        hp_.hidden_size <= 0 ||
        hp_.ffn_size <= 0 ||
        hp_.n_head <= 0 ||
        hp_.n_head_kv <= 0 ||
        hp_.head_dim <= 0 ||
        hp_.vocab_size <= 0
    ) {
        throw std::runtime_error("LlamaCpuExecutor received incomplete hyperparameters");
    }

    if (hp_.n_head % hp_.n_head_kv != 0) {
        throw std::runtime_error("n_head must be divisible by n_head_kv");
    }

    if (hp_.n_head * hp_.head_dim != hp_.hidden_size) {
        throw std::runtime_error("n_head * head_dim must equal hidden_size");
    }
}

ggml_tensor* LlamaCpuExecutor::require_tensor(const std::string& name) const {
    ggml_tensor* tensor = ggml_get_tensor(tensor_context_, name.c_str());

    if (tensor == nullptr) {
        throw std::runtime_error("required tensor not found in shard: " + name);
    }

    if (tensor->data == nullptr) {
        throw std::runtime_error("required tensor has null data pointer: " + name);
    }

    return tensor;
}

std::vector<float> LlamaCpuExecutor::tensor_row_to_float(
    const ggml_tensor* tensor,
    int64_t row
) const {
    if (tensor == nullptr || tensor->data == nullptr) {
        throw std::runtime_error("cannot read row from null tensor");
    }

    const int64_t row_width = tensor->ne[0];
    const int64_t row_count = tensor->ne[1] > 0 ? tensor->ne[1] : 1;

    if (row_width <= 0 || row_count <= 0) {
        throw std::runtime_error("tensor has invalid row dimensions");
    }

    if (row < 0 || row >= row_count) {
        throw std::runtime_error("tensor row index out of range");
    }

    /*
     * Use GGML itself to convert/gather the row.
     *
     * This is important for quantized tensors such as Q4_K/Q6_K.
     * The previous implementation used type_traits->to_float directly,
     * which is not available for all quantized tensor types in this build.
     */
    ggml_init_params params{};
    params.mem_size = 64ull * 1024ull * 1024ull;
    params.mem_buffer = nullptr;
    params.no_alloc = false;

    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate temporary GGML context for row conversion");
    }

    ggml_tensor* row_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    if (row_id == nullptr || row_id->data == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("failed to allocate GGML row id tensor");
    }

    static_cast<int32_t*>(row_id->data)[0] = static_cast<int32_t>(row);

    const char* type_name = ggml_type_name(tensor->type);
    if (type_name == nullptr) {
        type_name = "unknown";
    }

    ggml_tensor* gathered = ggml_get_rows(
        ctx,
        const_cast<ggml_tensor*>(tensor),
        row_id
    );

    if (gathered == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_get_rows failed for tensor type: " + std::string(type_name));
    }

    ggml_tensor* out_f32 = ggml_new_tensor_2d(
        ctx,
        GGML_TYPE_F32,
        row_width,
        1
    );

    if (out_f32 == nullptr || out_f32->data == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("failed to allocate GGML row output tensor");
    }

    ggml_tensor* copied = ggml_cpy(
        ctx,
        gathered,
        out_f32
    );

    if (copied == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_cpy failed for row conversion");
    }

    compute_single_node_graph(ctx, copied);

    std::vector<float> out =
        read_f32_tensor_flat(
            out_f32,
            static_cast<std::size_t>(row_width)
        );

    ggml_free(ctx);

    return out;
}

std::vector<float> LlamaCpuExecutor::tensor_vector_to_float(
    const std::string& name,
    int64_t expected_size
) const {
    const ggml_tensor* tensor = require_tensor(name);

    if (tensor->ne[0] != expected_size) {
        throw std::runtime_error(
            "tensor " + name +
            " has wrong vector size: expected " +
            std::to_string(expected_size) +
            ", got " +
            std::to_string(tensor->ne[0])
        );
    }

    return tensor_row_to_float(tensor, 0);
}

std::vector<float> LlamaCpuExecutor::matvec(
    const std::string& tensor_name,
    const std::vector<float>& x
) const {
    const ggml_tensor* weight = require_tensor(tensor_name);

    const int64_t input_dim = weight->ne[0];
    const int64_t output_dim = weight->ne[1];

    if (input_dim <= 0 || output_dim <= 0) {
        throw std::runtime_error("linear tensor has invalid shape: " + tensor_name);
    }

    if (static_cast<int64_t>(x.size()) != input_dim) {
        throw std::runtime_error(
            "matvec input size mismatch for " + tensor_name +
            ": expected " + std::to_string(input_dim) +
            ", got " + std::to_string(x.size())
        );
    }

    /*
     * Use GGML matmul instead of manually dequantizing rows.
     *
     * This lets GGML handle Q4_K, Q6_K, Q8_0, F16, F32, etc.,
     * through its backend implementation.
     */
    ggml_init_params params{};
    params.mem_size = 128ull * 1024ull * 1024ull;
    params.mem_buffer = nullptr;
    params.no_alloc = false;

    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate temporary GGML context for matvec");
    }

    ggml_tensor* x_tensor = ggml_new_tensor_2d(
        ctx,
        GGML_TYPE_F32,
        input_dim,
        1
    );

    if (x_tensor == nullptr || x_tensor->data == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("failed to allocate GGML matvec input tensor");
    }

    std::memcpy(
        x_tensor->data,
        x.data(),
        x.size() * sizeof(float)
    );

    const char* type_name = ggml_type_name(weight->type);
    if (type_name == nullptr) {
        type_name = "unknown";
    }

    ggml_tensor* result = ggml_mul_mat(
        ctx,
        const_cast<ggml_tensor*>(weight),
        x_tensor
    );

    if (result == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_mul_mat failed for tensor: " + tensor_name + ", type=" + std::string(type_name));
    }

    ggml_tensor* out_f32 = ggml_new_tensor_2d(
        ctx,
        GGML_TYPE_F32,
        output_dim,
        1
    );

    if (out_f32 == nullptr || out_f32->data == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("failed to allocate GGML matvec output tensor");
    }

    ggml_tensor* copied = ggml_cpy(
        ctx,
        result,
        out_f32
    );

    if (copied == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_cpy failed for matvec output");
    }

    compute_single_node_graph(ctx, copied);

    std::vector<float> y =
        read_f32_tensor_flat(
            out_f32,
            static_cast<std::size_t>(output_dim)
        );

    ggml_free(ctx);

    return y;
}

std::vector<float> LlamaCpuExecutor::rms_norm(
    const std::vector<float>& x,
    const std::vector<float>& weight
) const {
    if (x.size() != weight.size()) {
        throw std::runtime_error("rms_norm input and weight sizes differ");
    }

    double sum_sq = 0.0;

    for (float v : x) {
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }

    const float scale =
        1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(x.size())) + hp_.rms_eps);

    std::vector<float> y(x.size(), 0.0f);

    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] * scale * weight[i];
    }

    return y;
}

void LlamaCpuExecutor::apply_rope(
    std::vector<float>& q,
    std::vector<float>& k,
    int position
) const {
    const int q_heads = hp_.n_head;
    const int kv_heads = hp_.n_head_kv;
    const int dim = hp_.head_dim;

    auto rotate = [&](std::vector<float>& x, int heads) {
        for (int h = 0; h < heads; ++h) {
            const int base = h * dim;

            for (int i = 0; i < dim; i += 2) {
                const float inv_freq =
                    std::pow(hp_.rope_theta, -static_cast<float>(i) / static_cast<float>(dim));

                const float angle = static_cast<float>(position) * inv_freq;
                const float c = std::cos(angle);
                const float s = std::sin(angle);

                const std::size_t i0 = static_cast<std::size_t>(base + i);
                const std::size_t i1 = static_cast<std::size_t>(base + i + 1);

                const float x0 = x[i0];
                const float x1 = x[i1];

                x[i0] = x0 * c - x1 * s;
                x[i1] = x0 * s + x1 * c;
            }
        }
    };

    rotate(q, q_heads);
    rotate(k, kv_heads);
}

std::vector<float> LlamaCpuExecutor::attention(
    int layer_id,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int position
) {
    LayerKvCache& cache = layer_cache_[layer_id];

    const int kv_dim = hp_.n_head_kv * hp_.head_dim;
    const int q_dim = hp_.n_head * hp_.head_dim;

    if (static_cast<int>(q.size()) != q_dim) {
        throw std::runtime_error("Q vector has wrong size");
    }

    if (static_cast<int>(k.size()) != kv_dim || static_cast<int>(v.size()) != kv_dim) {
        throw std::runtime_error("K/V vector has wrong size");
    }

    const int required_seq = position + 1;

    if (cache.seq_len < required_seq) {
        cache.k.resize(static_cast<std::size_t>(required_seq * kv_dim), 0.0f);
        cache.v.resize(static_cast<std::size_t>(required_seq * kv_dim), 0.0f);
        cache.seq_len = required_seq;
    }

    std::copy(
        k.begin(),
        k.end(),
        cache.k.begin() + static_cast<std::ptrdiff_t>(position * kv_dim)
    );

    std::copy(
        v.begin(),
        v.end(),
        cache.v.begin() + static_cast<std::ptrdiff_t>(position * kv_dim)
    );

    std::vector<float> context(static_cast<std::size_t>(q_dim), 0.0f);

    const int groups = hp_.n_head / hp_.n_head_kv;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hp_.head_dim));

    for (int qh = 0; qh < hp_.n_head; ++qh) {
        const int kvh = qh / groups;

        std::vector<float> scores(static_cast<std::size_t>(required_seq), 0.0f);

        float max_score = -std::numeric_limits<float>::infinity();

        for (int pos = 0; pos < required_seq; ++pos) {
            float dot = 0.0f;

            for (int d = 0; d < hp_.head_dim; ++d) {
                const float qv =
                    q[static_cast<std::size_t>(qh * hp_.head_dim + d)];

                const float kv =
                    cache.k[
                        static_cast<std::size_t>(
                            pos * kv_dim + kvh * hp_.head_dim + d
                        )
                    ];

                dot += qv * kv;
            }

            const float score = dot * scale;
            scores[static_cast<std::size_t>(pos)] = score;
            max_score = std::max(max_score, score);
        }

        float denom = 0.0f;

        for (float& score : scores) {
            score = std::exp(score - max_score);
            denom += score;
        }

        if (denom <= 0.0f) {
            throw std::runtime_error("attention softmax denominator is non-positive");
        }

        for (int pos = 0; pos < required_seq; ++pos) {
            const float prob = scores[static_cast<std::size_t>(pos)] / denom;

            for (int d = 0; d < hp_.head_dim; ++d) {
                context[static_cast<std::size_t>(qh * hp_.head_dim + d)] +=
                    prob *
                    cache.v[
                        static_cast<std::size_t>(
                            pos * kv_dim + kvh * hp_.head_dim + d
                        )
                    ];
            }
        }
    }

    return context;
}

std::vector<float> LlamaCpuExecutor::run_layer(
    int layer_id,
    const std::vector<float>& input,
    int seq_len,
    int kv_seq_before
) {
    const std::string prefix = layer_prefix(layer_id);

    const std::vector<float> attn_norm_weight =
        tensor_vector_to_float(prefix + "attn_norm.weight", hp_.hidden_size);

    const std::vector<float> ffn_norm_weight =
        tensor_vector_to_float(prefix + "ffn_norm.weight", hp_.hidden_size);

    std::vector<float> output = input;

    for (int t = 0; t < seq_len; ++t) {
        const int position = kv_seq_before + t;

        const auto row_begin =
            output.begin() + static_cast<std::ptrdiff_t>(t * hp_.hidden_size);

        const std::vector<float> x(
            row_begin,
            row_begin + hp_.hidden_size
        );

        const std::vector<float> xn =
            rms_norm(x, attn_norm_weight);

        std::vector<float> q =
            matvec(prefix + "attn_q.weight", xn);

        std::vector<float> k =
            matvec(prefix + "attn_k.weight", xn);

        std::vector<float> v =
            matvec(prefix + "attn_v.weight", xn);

        apply_rope(q, k, position);

        const std::vector<float> ctx =
            attention(layer_id, q, k, v, position);

        const std::vector<float> attn_out =
            matvec(prefix + "attn_output.weight", ctx);

        std::vector<float> h1(static_cast<std::size_t>(hp_.hidden_size), 0.0f);

        for (int i = 0; i < hp_.hidden_size; ++i) {
            h1[static_cast<std::size_t>(i)] =
                x[static_cast<std::size_t>(i)] +
                attn_out[static_cast<std::size_t>(i)];
        }

        const std::vector<float> fn =
            rms_norm(h1, ffn_norm_weight);

        std::vector<float> gate =
            matvec(prefix + "ffn_gate.weight", fn);

        std::vector<float> up =
            matvec(prefix + "ffn_up.weight", fn);

        if (static_cast<int>(gate.size()) != hp_.ffn_size ||
            static_cast<int>(up.size()) != hp_.ffn_size) {
            throw std::runtime_error("FFN gate/up size mismatch");
        }

        std::vector<float> ffn_intermediate(static_cast<std::size_t>(hp_.ffn_size), 0.0f);

        for (int i = 0; i < hp_.ffn_size; ++i) {
            ffn_intermediate[static_cast<std::size_t>(i)] =
                silu(gate[static_cast<std::size_t>(i)]) *
                up[static_cast<std::size_t>(i)];
        }

        const std::vector<float> down =
            matvec(prefix + "ffn_down.weight", ffn_intermediate);

        for (int i = 0; i < hp_.hidden_size; ++i) {
            output[static_cast<std::size_t>(t * hp_.hidden_size + i)] =
                h1[static_cast<std::size_t>(i)] +
                down[static_cast<std::size_t>(i)];
        }
    }

    return output;
}

std::uint16_t float_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));

    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x007fffffu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }

        mantissa = (mantissa | 0x00800000u) >> static_cast<unsigned>(1 - exponent);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x00001000u) >> 13u));
    }

    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    return static_cast<std::uint16_t>(
        sign |
        (static_cast<std::uint32_t>(exponent) << 10u) |
        ((mantissa + 0x00001000u) >> 13u)
    );
}

float f16_bits_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16u;
    std::uint32_t exponent = (value >> 10u) & 0x1fu;
    std::uint32_t mantissa = value & 0x03ffu;

    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x03ffu;
            exponent = exponent + (127 - 15);
            bits = sign | (exponent << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        exponent = exponent + (127 - 15);
        bits = sign | (exponent << 23u) | (mantissa << 13u);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

float read_f16_le(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset
) {
    if (offset + 2 > bytes.size()) {
        throw std::runtime_error("cannot read float16 past hidden tensor");
    }

    const std::uint16_t raw =
        static_cast<std::uint16_t>(bytes[offset]) |
        static_cast<std::uint16_t>(bytes[offset + 1]) << 8u;

    return f16_bits_to_float(raw);
}

void write_f16_le(
    std::vector<std::uint8_t>& bytes,
    std::size_t offset,
    float value
) {
    if (offset + 2 > bytes.size()) {
        throw std::runtime_error("cannot write float16 past hidden tensor");
    }

    const std::uint16_t raw = float_to_f16_bits(value);
    bytes[offset] = static_cast<std::uint8_t>(raw & 0xffu);
    bytes[offset + 1] = static_cast<std::uint8_t>((raw >> 8u) & 0xffu);
}

std::vector<float> LlamaCpuExecutor::tensor_to_hidden_f32(
    const dli::common::TensorBuffer& tensor
) const {
    dli::common::validate_hidden_states_tensor(
        tensor,
        hp_.hidden_size,
        true
    );

    const std::int64_t batch = tensor.metadata.shape[0];
    const std::int64_t seq = tensor.metadata.shape[1];
    const std::int64_t hidden = tensor.metadata.shape[2];

    if (batch != 1) {
        throw std::runtime_error("CPU reference executor currently supports batch=1");
    }

    const std::size_t count =
        static_cast<std::size_t>(batch * seq * hidden);

    std::vector<float> values(count, 0.0f);

    if (tensor.metadata.dtype == "float32") {
        for (std::size_t i = 0; i < count; ++i) {
            values[i] = read_f32_le(tensor.bytes, i * sizeof(float));
        }
        return values;
    }

    if (tensor.metadata.dtype == "float16") {
        for (std::size_t i = 0; i < count; ++i) {
            values[i] = read_f16_le(tensor.bytes, i * 2);
        }
        return values;
    }

    throw std::runtime_error("unsupported hidden-state dtype: " + tensor.metadata.dtype);
}

std::vector<std::int64_t> LlamaCpuExecutor::tensor_to_token_ids(
    const dli::common::TensorBuffer& tensor
) const {
    dli::common::validate_token_ids_tensor(tensor);

    const std::int64_t batch = tensor.metadata.shape[0];
    const std::int64_t seq = tensor.metadata.shape[1];

    if (batch != 1) {
        throw std::runtime_error("CPU reference executor currently supports batch=1");
    }

    std::vector<std::int64_t> tokens(static_cast<std::size_t>(seq), 0);

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = read_i64_le(tensor.bytes, i * sizeof(std::int64_t));
    }

    return tokens;
}

dli::common::TensorBuffer LlamaCpuExecutor::float_hidden_to_tensor(
    const std::vector<float>& hidden,
    int batch,
    int seq_len,
    const std::string& activation_precision
) const {
    const bool use_f16 =
        activation_precision == "fp16" ||
        activation_precision == "float16";

    dli::common::TensorBuffer tensor;
    tensor.metadata.dtype = use_f16 ? "float16" : "float32";
    tensor.metadata.shape = {
        batch,
        seq_len,
        hp_.hidden_size
    };
    tensor.metadata.byte_order = "little";

    if (use_f16) {
        tensor.bytes.resize(hidden.size() * 2);
        for (std::size_t i = 0; i < hidden.size(); ++i) {
            write_f16_le(tensor.bytes, i * 2, hidden[i]);
        }
    } else {
        tensor.bytes.resize(hidden.size() * sizeof(float));
        for (std::size_t i = 0; i < hidden.size(); ++i) {
            write_f32_le(tensor.bytes, i * sizeof(float), hidden[i]);
        }
    }

    return tensor;
}

dli::common::TensorBuffer LlamaCpuExecutor::execute_source(
    const dli::common::TensorBuffer& token_ids,
    const std::vector<int>& layers,
    const std::string&,
    int kv_seq_before,
    const std::string& output_activation_precision
) {
    const std::vector<std::int64_t> tokens =
        tensor_to_token_ids(token_ids);

    const int seq_len = static_cast<int>(tokens.size());

    const ggml_tensor* embedding =
        require_tensor("token_embd.weight");

    if (embedding->ne[0] != hp_.hidden_size) {
        throw std::runtime_error("token_embd.weight hidden size mismatch");
    }

    if (embedding->ne[1] != hp_.vocab_size) {
        throw std::runtime_error("token_embd.weight vocab size mismatch");
    }

    std::vector<float> hidden(
        static_cast<std::size_t>(seq_len * hp_.hidden_size),
        0.0f
    );

    for (int t = 0; t < seq_len; ++t) {
        const std::int64_t token_id = tokens[static_cast<std::size_t>(t)];

        if (token_id < 0 || token_id >= hp_.vocab_size) {
            throw std::runtime_error("token id out of range: " + std::to_string(token_id));
        }

        const std::vector<float> row =
            tensor_row_to_float(embedding, token_id);

        std::copy(
            row.begin(),
            row.end(),
            hidden.begin() + static_cast<std::ptrdiff_t>(t * hp_.hidden_size)
        );
    }

    for (const int layer_id : layers) {
        hidden = run_layer(layer_id, hidden, seq_len, kv_seq_before);
    }

    return float_hidden_to_tensor(hidden, 1, seq_len, output_activation_precision);
}

dli::common::TensorBuffer LlamaCpuExecutor::execute_intermediate(
    const dli::common::TensorBuffer& hidden_states,
    const std::vector<int>& layers,
    const std::string&,
    int kv_seq_before,
    const std::string& output_activation_precision
){
    const int seq_len =
        static_cast<int>(hidden_states.metadata.shape[1]);

    std::vector<float> hidden =
        tensor_to_hidden_f32(hidden_states);

    for (const int layer_id : layers) {
        hidden = run_layer(layer_id, hidden, seq_len, kv_seq_before);
    }

    return float_hidden_to_tensor(hidden, 1, seq_len, output_activation_precision);
}

int LlamaCpuExecutor::argmax_lm_head(
    const std::vector<float>& last_hidden
) const {
    const std::vector<float> logits =
        matvec("output.weight", last_hidden);

    if (logits.empty()) {
        throw std::runtime_error("empty logits from lm_head");
    }

    int best_id = 0;
    float best_value = logits[0];

    for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
        if (logits[static_cast<std::size_t>(i)] > best_value) {
            best_value = logits[static_cast<std::size_t>(i)];
            best_id = i;
        }
    }

    return best_id;
}

int LlamaCpuExecutor::execute_terminal(
    const dli::common::TensorBuffer& hidden_states,
    const std::vector<int>& layers,
    const std::string&,
    int kv_seq_before,
    bool owns_norm,
    bool owns_lm_head
) {
    const int seq_len =
        static_cast<int>(hidden_states.metadata.shape[1]);

    std::vector<float> hidden =
        tensor_to_hidden_f32(hidden_states);

    for (const int layer_id : layers) {
        hidden = run_layer(layer_id, hidden, seq_len, kv_seq_before);
    }

    if (!owns_norm) {
        throw std::runtime_error("terminal partition must own output norm");
    }

    if (!owns_lm_head) {
        throw std::runtime_error("terminal partition must own lm_head");
    }

    const std::vector<float> output_norm_weight =
        tensor_vector_to_float("output_norm.weight", hp_.hidden_size);

    const int last_t = seq_len - 1;

    const auto last_begin =
        hidden.begin() + static_cast<std::ptrdiff_t>(last_t * hp_.hidden_size);

    const std::vector<float> last_hidden(
        last_begin,
        last_begin + hp_.hidden_size
    );

    const std::vector<float> normed =
        rms_norm(last_hidden, output_norm_weight);

    return argmax_lm_head(normed);
}

std::uint64_t LlamaCpuExecutor::kv_cache_bytes() const {
    std::uint64_t total = 0;

    for (const auto& item : layer_cache_) {
        total +=
            static_cast<std::uint64_t>(item.second.k.size() * sizeof(float)) +
            static_cast<std::uint64_t>(item.second.v.size() * sizeof(float));
    }

    return total;
}

} // namespace dli_stage