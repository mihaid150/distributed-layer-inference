#include "dli_stage/runtimes/llama_partial_runtime.hpp"
#include "dli_stage/runtimes/llama_cpu_executor.hpp"

#include "dli/common/dli2_abi.hpp"
#include "dli/common/json_escape.hpp"
#include "dli/common/tensor.hpp"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dli_stage {

namespace {

std::uint64_t current_rss_mb() {
    std::ifstream input("/proc/self/status");

    if (!input.is_open()) {
        return 0;
    }

    std::string key;
    while (input >> key) {
        if (key == "VmRSS:") {
            std::uint64_t kb = 0;
            std::string unit;
            input >> kb >> unit;
            return (kb + 1023u) / 1024u;
        }

        std::string rest;
        std::getline(input, rest);
    }

    return 0;
}

std::int64_t read_i64_le(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset
) {
    if (offset + sizeof(std::int64_t) > bytes.size()) {
        throw std::runtime_error("cannot read int64 token id past tensor byte buffer");
    }

    std::uint64_t value = 0;

    for (int i = 0; i < 8; ++i) {
        value |=
            static_cast<std::uint64_t>(bytes[offset + static_cast<std::size_t>(i)])
            << static_cast<unsigned>(i * 8);
    }

    return static_cast<std::int64_t>(value);
}

void write_f32_le(
    std::vector<std::uint8_t>& bytes,
    std::size_t offset,
    float value
) {
    static_assert(sizeof(float) == 4);

    if (offset + sizeof(float) > bytes.size()) {
        throw std::runtime_error("cannot write float32 past tensor byte buffer");
    }

    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(float));

    bytes[offset + 0] = static_cast<std::uint8_t>(raw & 0xffu);
    bytes[offset + 1] = static_cast<std::uint8_t>((raw >> 8u) & 0xffu);
    bytes[offset + 2] = static_cast<std::uint8_t>((raw >> 16u) & 0xffu);
    bytes[offset + 3] = static_cast<std::uint8_t>((raw >> 24u) & 0xffu);
}

double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
) {
    const auto duration = std::chrono::duration<double, std::milli>(end - start);
    return duration.count();
}

void validate_model_path_if_present(const std::string& model_path) {
    if (model_path.empty()) {
        return;
    }

    const std::filesystem::path path(model_path);

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            "LlamaPartialRuntime model path does not exist: " + model_path
        );
    }

    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "LlamaPartialRuntime model path is not a regular file: " + model_path
        );
    }
}

std::string model_meta_string(
    const llama_model* model,
    const char* key
) {
    if (model == nullptr || key == nullptr) {
        return "";
    }

    const int32_t required = llama_model_meta_val_str(
        model,
        key,
        nullptr,
        0
    );

    if (required <= 0) {
        return "";
    }

    std::vector<char> buffer(static_cast<std::size_t>(required) + 1, '\0');

    const int32_t actual = llama_model_meta_val_str(
        model,
        key,
        buffer.data(),
        buffer.size()
    );

    if (actual < 0) {
        return "";
    }

    return std::string(buffer.data());
}

int model_meta_int(
    const llama_model* model,
    const char* key
) {
    const std::string value = model_meta_string(model, key);

    if (value.empty()) {
        return 0;
    }

    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        return 0;
    }
}

float model_meta_float(
    const llama_model* model,
    const char* key,
    float default_value
) {
    const std::string value = model_meta_string(model, key);

    if (value.empty()) {
        return default_value;
    }

    try {
        return std::stof(value);
    } catch (const std::exception&) {
        return default_value;
    }
}

int require_gguf_key(const gguf_context* ctx, const std::string& key) {
    const int key_id = gguf_find_key(ctx, key.c_str());

    if (key_id < 0) {
        throw std::runtime_error("missing required shard metadata key: " + key);
    }

    return key_id;
}

std::string gguf_string_required(const gguf_context* ctx, const std::string& key) {
    const int key_id = require_gguf_key(ctx, key);
    const char* value = gguf_get_val_str(ctx, key_id);
    return value != nullptr ? value : "";
}

int gguf_i32_required(const gguf_context* ctx, const std::string& key) {
    return gguf_get_val_i32(ctx, require_gguf_key(ctx, key));
}

bool gguf_bool_required(const gguf_context* ctx, const std::string& key) {
    return gguf_get_val_bool(ctx, require_gguf_key(ctx, key));
}

std::vector<int> gguf_i32_array_required(
    const gguf_context* ctx,
    const std::string& key
) {
    const int key_id = require_gguf_key(ctx, key);

    if (gguf_get_arr_type(ctx, key_id) != GGUF_TYPE_INT32) {
        throw std::runtime_error("shard metadata key is not array<int32>: " + key);
    }

    const int64_t n = gguf_get_arr_n(ctx, key_id);
    const void* raw = gguf_get_arr_data(ctx, key_id);

    if (n < 0 || raw == nullptr) {
        throw std::runtime_error("invalid shard metadata array: " + key);
    }

    const int32_t* data = static_cast<const int32_t*>(raw);

    std::vector<int> values;
    values.reserve(static_cast<std::size_t>(n));

    for (int64_t i = 0; i < n; ++i) {
        values.push_back(static_cast<int>(data[i]));
    }

    return values;
}

LlamaShardMetadata load_shard_metadata_from_gguf(const std::string& model_path) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;

    gguf_context* raw_ctx = gguf_init_from_file(model_path.c_str(), params);
    if (raw_ctx == nullptr) {
        throw std::runtime_error("failed to open GGUF shard for metadata: " + model_path);
    }

    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(raw_ctx, gguf_free);

    LlamaShardMetadata metadata;
    metadata.shard_loaded = true;

    metadata.format = gguf_string_required(ctx.get(), "dli.format");
    metadata.format_version = gguf_i32_required(ctx.get(), "dli.format_version");
    metadata.partition_id = gguf_string_required(ctx.get(), "dli.partition_id");
    metadata.stage_id = gguf_i32_required(ctx.get(), "dli.stage_id");
    metadata.layers = gguf_i32_array_required(ctx.get(), "dli.layers");

    metadata.owns_embedding = gguf_bool_required(ctx.get(), "dli.owns_embedding");
    metadata.owns_norm = gguf_bool_required(ctx.get(), "dli.owns_norm");
    metadata.owns_lm_head = gguf_bool_required(ctx.get(), "dli.owns_lm_head");

    metadata.next_partition_id = gguf_string_required(ctx.get(), "dli.next_partition_id");
    metadata.next_stage_url = gguf_string_required(ctx.get(), "dli.next_stage_url");

    metadata.hidden_size = gguf_i32_required(ctx.get(), "dli.hidden_size");
    metadata.source_model = gguf_string_required(ctx.get(), "dli.source_model");

    if (metadata.format != "dli.gguf.stage_shard") {
        throw std::runtime_error("invalid dli.format in shard: " + metadata.format);
    }

    if (metadata.format_version != 1) {
        throw std::runtime_error(
            "unsupported dli.format_version: " +
            std::to_string(metadata.format_version)
        );
    }

    return metadata;
}

std::string int_vector_json(const std::vector<int>& values) {
    std::ostringstream out;
    out << "[";

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }

        out << values[i];
    }

    out << "]";
    return out.str();
}

void validate_shard_against_stage_config(
    const LlamaShardMetadata& shard,
    const LlamaPartialRuntimeConfig& config
) {
    if (!shard.shard_loaded) {
        throw std::runtime_error("cannot validate unloaded DLI GGUF shard metadata");
    }

    if (shard.stage_id != config.stage_id) {
        throw std::runtime_error(
            "DLI shard stage_id mismatch: CLI/env stage_id=" +
            std::to_string(config.stage_id) +
            ", shard stage_id=" +
            std::to_string(shard.stage_id)
        );
    }

    if (shard.layers != config.expected_layers) {
        throw std::runtime_error(
            "DLI shard layer allocation mismatch: expected " +
            int_vector_json(config.expected_layers) +
            ", got " +
            int_vector_json(shard.layers)
        );
    }

    if (shard.owns_embedding != config.expected_owns_embedding) {
        throw std::runtime_error("DLI shard owns_embedding does not match stage_map.yaml");
    }

    if (shard.owns_norm != config.expected_owns_norm) {
        throw std::runtime_error("DLI shard owns_norm does not match stage_map.yaml");
    }

    if (shard.owns_lm_head != config.expected_owns_lm_head) {
        throw std::runtime_error("DLI shard owns_lm_head does not match stage_map.yaml");
    }

    if (shard.next_stage_url != config.expected_next_stage_url) {
        throw std::runtime_error(
            "DLI shard next_stage_url mismatch: expected '" +
            config.expected_next_stage_url +
            "', got '" +
            shard.next_stage_url +
            "'"
        );
    }
}

void validate_runtime_request_matches_partition(
    const RuntimeRequest& request,
    const LlamaShardMetadata& shard
) {
    const dli::common::Dli2PayloadKind kind =
        dli::common::infer_dli2_payload_kind(request.input_tensor);

    if (shard.owns_embedding) {
        if (kind != dli::common::Dli2PayloadKind::TokenIds) {
            throw std::runtime_error(
                "source partition owns embedding and expects DLI2 token ids input"
            );
        }

        dli::common::validate_token_ids_tensor(request.input_tensor);
        return;
    }

    if (kind != dli::common::Dli2PayloadKind::HiddenStates) {
        throw std::runtime_error(
            "non-source partition expects DLI2 hidden states input"
        );
    }

    dli::common::validate_hidden_states_tensor(
        request.input_tensor,
        shard.hidden_size,
        false
    );
}

} // namespace

LlamaPartialRuntime::LlamaPartialRuntime(LlamaPartialRuntimeConfig config)
    : config_(std::move(config)) {
    validate_model_path_if_present(config_.model_path);

    model_metadata_.model_path = config_.model_path;

    if (!config_.model_path.empty()) {
        const auto load_start = std::chrono::steady_clock::now();

        shard_metadata_ = load_shard_metadata_from_gguf(config_.model_path);
        validate_shard_against_stage_config(shard_metadata_, config_);

        llama_model_params model_params = llama_model_default_params();

        // Still vocab/metadata only. Real partial execution will require a
        // custom GGML graph or llama.cpp internal graph integration.
        model_params.vocab_only = true;
        model_params.n_gpu_layers = 0;

        model_ = llama_model_load_from_file(config_.model_path.c_str(), model_params);
        if (model_ == nullptr) {
            throw std::runtime_error(
                "failed to load GGUF metadata/vocab for LlamaPartialRuntime: " +
                config_.model_path
            );
        }

        vocab_ = llama_model_get_vocab(model_);
        if (vocab_ == nullptr) {
            llama_model_free(model_);
            model_ = nullptr;
            throw std::runtime_error(
                "failed to get vocab from LlamaPartialRuntime model: " +
                config_.model_path
            );
        }

        model_metadata_.model_loaded = true;
        model_metadata_.architecture = model_meta_string(model_, "general.architecture");
        model_metadata_.name = model_meta_string(model_, "general.name");

        model_metadata_.n_ctx_train = model_meta_int(model_, "llama.context_length");
        model_metadata_.n_embd = model_meta_int(model_, "llama.embedding_length");
        model_metadata_.n_layer = model_meta_int(model_, "llama.block_count");
        model_metadata_.n_head = model_meta_int(model_, "llama.attention.head_count");
        model_metadata_.n_head_kv = model_meta_int(model_, "llama.attention.head_count_kv");
        model_metadata_.n_vocab = llama_vocab_n_tokens(vocab_);

        if (shard_metadata_.hidden_size > 0 && model_metadata_.n_embd > 0) {
            if (shard_metadata_.hidden_size != model_metadata_.n_embd) {
                throw std::runtime_error(
                    "DLI shard hidden_size does not match llama.embedding_length"
                );
            }
        }

        load_raw_tensor_context();

        const auto load_end = std::chrono::steady_clock::now();
        model_load_ms_ = elapsed_ms(load_start, load_end);

        LlamaExecutorHyperparams hp;
        hp.architecture = model_metadata_.architecture;
        hp.hidden_size = model_metadata_.n_embd;
        hp.ffn_size = model_meta_int(model_, "llama.feed_forward_length");
        hp.n_head = model_metadata_.n_head;
        hp.n_head_kv = model_metadata_.n_head_kv;
        hp.head_dim = model_meta_int(model_, "llama.rope.dimension_count");
        hp.vocab_size = model_metadata_.n_vocab;
        hp.rms_eps = model_meta_float(
            model_,
            "llama.attention.layer_norm_rms_epsilon",
            1.0e-5f
        );
        hp.rope_theta = model_meta_float(
            model_,
            "llama.rope.freq_base",
            10000.0f
        );

        executor_ = std::make_unique<LlamaCpuExecutor>(
            tensor_data_ctx_,
            hp
        );
    }
}

LlamaPartialRuntime::~LlamaPartialRuntime() {
    executor_.reset();

    if (tensor_data_ctx_ != nullptr) {
        ggml_free(tensor_data_ctx_);
        tensor_data_ctx_ = nullptr;
    }

    if (tensor_gguf_ctx_ != nullptr) {
        gguf_free(tensor_gguf_ctx_);
        tensor_gguf_ctx_ = nullptr;
    }

    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
        vocab_ = nullptr;
    }

    model_metadata_.model_loaded = false;
}

void LlamaPartialRuntime::load_raw_tensor_context() {
    if (config_.model_path.empty()) {
        throw std::runtime_error("cannot load raw tensors without model_path");
    }

    if (tensor_gguf_ctx_ != nullptr || tensor_data_ctx_ != nullptr) {
        return;
    }

    ggml_context* raw_data_ctx = nullptr;

    gguf_init_params params{};
    params.no_alloc = false;
    params.ctx = &raw_data_ctx;

    tensor_gguf_ctx_ = gguf_init_from_file(config_.model_path.c_str(), params);
    tensor_data_ctx_ = raw_data_ctx;

    if (tensor_gguf_ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to load GGUF shard tensor data: " + config_.model_path
        );
    }

    if (tensor_data_ctx_ == nullptr) {
        gguf_free(tensor_gguf_ctx_);
        tensor_gguf_ctx_ = nullptr;

        throw std::runtime_error(
            "GGUF shard did not produce a GGML tensor context: " + config_.model_path
        );
    }
}

dli::common::TensorBuffer LlamaPartialRuntime::execute_token_embedding_only(
    const dli::common::TensorBuffer& token_ids
) const {
    if (tensor_data_ctx_ == nullptr) {
        throw std::runtime_error("raw GGUF tensor context is not loaded");
    }

    if (!shard_metadata_.owns_embedding) {
        throw std::runtime_error("execute_token_embedding_only called on non-source partition");
    }

    dli::common::validate_token_ids_tensor(token_ids);

    if (token_ids.metadata.shape.size() != 2) {
        throw std::runtime_error("token id input must have shape [batch, seq]");
    }

    const std::int64_t batch = token_ids.metadata.shape[0];
    const std::int64_t seq = token_ids.metadata.shape[1];

    if (batch <= 0 || seq <= 0) {
        throw std::runtime_error("token id input has invalid [batch, seq]");
    }

    ggml_tensor* embedding = ggml_get_tensor(tensor_data_ctx_, "token_embd.weight");
    if (embedding == nullptr) {
        throw std::runtime_error("source shard is missing token_embd.weight");
    }

    const int64_t hidden_size = embedding->ne[0];
    const int64_t vocab_size = embedding->ne[1];

    if (hidden_size <= 0 || vocab_size <= 0) {
        throw std::runtime_error("token_embd.weight has invalid dimensions");
    }

    if (
        shard_metadata_.hidden_size > 0 &&
        hidden_size != static_cast<int64_t>(shard_metadata_.hidden_size)
    ) {
        throw std::runtime_error(
            "token_embd.weight hidden size does not match dli.hidden_size"
        );
    }

    const std::size_t token_count =
        static_cast<std::size_t>(batch * seq);

    const std::size_t output_float_count =
        token_count * static_cast<std::size_t>(hidden_size);

    std::vector<float> output_f32(output_float_count, 0.0f);

    const std::size_t row_size =
        ggml_row_size(embedding->type, hidden_size);

    const auto* base =
        static_cast<const std::uint8_t*>(embedding->data);

    if (base == nullptr) {
        throw std::runtime_error("token_embd.weight has null data pointer");
    }

    const ggml_type_traits* traits = ggml_get_type_traits(embedding->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        throw std::runtime_error("token_embd.weight type cannot be converted to float32");
    }

    std::vector<float> row_f32(static_cast<std::size_t>(hidden_size), 0.0f);

    for (std::size_t i = 0; i < token_count; ++i) {
        const std::int64_t token_id =
            read_i64_le(token_ids.bytes, i * sizeof(std::int64_t));

        if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error(
                "token id out of vocabulary range: " + std::to_string(token_id)
            );
        }

        const std::uint8_t* row_ptr =
            base + static_cast<std::size_t>(token_id) * row_size;

        traits->to_float(
            row_ptr,
            row_f32.data(),
            hidden_size
        );

        float* output_row =
            output_f32.data() + i * static_cast<std::size_t>(hidden_size);

        std::copy(
            row_f32.begin(),
            row_f32.end(),
            output_row
        );
    }

    dli::common::TensorBuffer out;
    out.metadata.dtype = "float32";
    out.metadata.shape = {
        batch,
        seq,
        hidden_size
    };
    out.metadata.byte_order = "little";

    out.bytes.resize(output_f32.size() * sizeof(float));

    for (std::size_t i = 0; i < output_f32.size(); ++i) {
        write_f32_le(
            out.bytes,
            i * sizeof(float),
            output_f32[i]
        );
    }

    return out;
}

RuntimeResponse LlamaPartialRuntime::forward(const RuntimeRequest& request) {
    if (!shard_metadata_.shard_loaded) {
        throw std::runtime_error("LlamaPartialRuntime has no loaded DLI GGUF shard");
    }

    validate_runtime_request_matches_partition(request, shard_metadata_);

    if (shard_metadata_.owns_embedding) {
        return forward_source_partition(request);
    }

    if (shard_metadata_.owns_lm_head || shard_metadata_.next_stage_url.empty()) {
        return forward_terminal_partition(request);
    }

    return forward_intermediate_partition(request);
}

std::uint64_t LlamaPartialRuntime::estimate_kv_cache_bytes(int seq_len) const {
    if (seq_len <= 0 || shard_metadata_.layers.empty()) {
        return 0;
    }

    const int hidden_size =
        shard_metadata_.hidden_size > 0
            ? shard_metadata_.hidden_size
            : model_metadata_.n_embd;

    if (hidden_size <= 0) {
        return 0;
    }

    // Conservative accounting for K and V cache as float32 for this correctness phase.
    // This is not the final memory layout of llama.cpp; it is a stable per-partition
    // accounting metric until the real GGML KV cache object is wired.
    return
        static_cast<std::uint64_t>(seq_len) *
        static_cast<std::uint64_t>(shard_metadata_.layers.size()) *
        static_cast<std::uint64_t>(hidden_size) *
        static_cast<std::uint64_t>(2) *
        static_cast<std::uint64_t>(sizeof(float));
}

PartitionKvCacheStep LlamaPartialRuntime::update_kv_cache_for_request(
    const RuntimeRequest& request
) {
    PartitionKvCacheStep step;
    step.seq_before = kv_cache_.seq_len;

    const int input_seq =
        static_cast<int>(dli::common::dli2_sequence_length(request.input_tensor));

    if (!request.kv_cache_enabled) {
        step.seq_after = step.seq_before;
        step.valid = true;
        step.bytes = estimate_kv_cache_bytes(step.seq_after);
        return step;
    }

    if (request.generation_mode == "prefill") {
        if (input_seq <= 0) {
            step.valid = false;
        } else {
            kv_cache_.seq_len += input_seq;
        }
    } else if (request.generation_mode == "decode") {
        if (input_seq != 1) {
            step.valid = false;
        } else {
            kv_cache_.seq_len += 1;
        }
    } else {
        step.valid = false;
    }

    kv_cache_.valid = kv_cache_.valid && step.valid;

    step.seq_after = kv_cache_.seq_len;
    step.bytes = estimate_kv_cache_bytes(step.seq_after);
    step.valid = kv_cache_.valid;

    return step;
}

RuntimeResponse LlamaPartialRuntime::forward_source_partition(
    const RuntimeRequest& request
) {
    if (!executor_) {
        throw std::runtime_error("source execution requested before executor initialization");
    }

    const auto start = std::chrono::steady_clock::now();

    const PartitionKvCacheStep kv_step = update_kv_cache_for_request(request);

    RuntimeResponse response;
    response.is_final_stage = false;
    response.next_token_id = -1;

    response.output_tensor = executor_->execute_source(
        request.input_tensor,
        shard_metadata_.layers,
        request.generation_mode,
        kv_step.seq_before
    );

    response.output_metadata_json = request.input_metadata_json;
    response.backend_metadata_json = backend_metadata_json();

    const auto end = std::chrono::steady_clock::now();

    response.metrics.backend = backend_name();
    response.metrics.status = "source_partition_executed_embedding_and_layers";
    response.metrics.compute_time_ms = elapsed_ms(start, end);
    response.metrics.true_comm_ms = 0.0;
    response.metrics.rpc_wall_time_ms = 0.0;

    response.metrics.input_tensor_bytes = request.input_tensor.bytes.size();
    response.metrics.output_tensor_bytes = response.output_tensor.bytes.size();

    response.metrics.stage_input_token_count =
        static_cast<int>(dli::common::dli2_sequence_length(request.input_tensor));
    response.metrics.stage_output_token_count =
        static_cast<int>(dli::common::dli2_sequence_length(response.output_tensor));

    response.metrics.kv_cache_step_valid = kv_step.valid;
    response.metrics.kv_cache_seq_before = kv_step.seq_before;
    response.metrics.kv_cache_seq_after = kv_step.seq_after;
    response.metrics.kv_cache_bytes = executor_->kv_cache_bytes();
    response.metrics.kv_cache_valid = kv_step.valid;

    response.metrics.model_load_ms = model_load_ms_;
    response.metrics.memory_rss_mb = current_rss_mb();

    return response;
}

RuntimeResponse LlamaPartialRuntime::forward_intermediate_partition(
    const RuntimeRequest& request
) {
    if (!executor_) {
        throw std::runtime_error("intermediate execution requested before executor initialization");
    }

    const auto start = std::chrono::steady_clock::now();

    const PartitionKvCacheStep kv_step = update_kv_cache_for_request(request);

    RuntimeResponse response;
    response.is_final_stage = false;
    response.next_token_id = -1;

    response.output_tensor = executor_->execute_intermediate(
        request.input_tensor,
        shard_metadata_.layers,
        request.generation_mode,
        kv_step.seq_before
    );

    response.output_metadata_json = request.input_metadata_json;
    response.backend_metadata_json = backend_metadata_json();

    const auto end = std::chrono::steady_clock::now();

    response.metrics.backend = backend_name();
    response.metrics.status = "intermediate_partition_executed_layers";
    response.metrics.compute_time_ms = elapsed_ms(start, end);
    response.metrics.true_comm_ms = 0.0;
    response.metrics.rpc_wall_time_ms = 0.0;

    response.metrics.input_tensor_bytes = request.input_tensor.bytes.size();
    response.metrics.output_tensor_bytes = response.output_tensor.bytes.size();

    response.metrics.stage_input_token_count =
        static_cast<int>(dli::common::dli2_sequence_length(request.input_tensor));
    response.metrics.stage_output_token_count =
        static_cast<int>(dli::common::dli2_sequence_length(response.output_tensor));

    response.metrics.kv_cache_step_valid = kv_step.valid;
    response.metrics.kv_cache_seq_before = kv_step.seq_before;
    response.metrics.kv_cache_seq_after = kv_step.seq_after;
    response.metrics.kv_cache_bytes = executor_->kv_cache_bytes();
    response.metrics.kv_cache_valid = kv_step.valid;

    response.metrics.model_load_ms = model_load_ms_;
    response.metrics.memory_rss_mb = current_rss_mb();

    return response;
}

RuntimeResponse LlamaPartialRuntime::forward_terminal_partition(
    const RuntimeRequest& request
) {
    if (!executor_) {
        throw std::runtime_error("terminal execution requested before executor initialization");
    }

    const auto start = std::chrono::steady_clock::now();

    const PartitionKvCacheStep kv_step = update_kv_cache_for_request(request);

    RuntimeResponse response;
    response.is_final_stage = true;
    response.next_token_id = executor_->execute_terminal(
        request.input_tensor,
        shard_metadata_.layers,
        request.generation_mode,
        kv_step.seq_before,
        shard_metadata_.owns_norm,
        shard_metadata_.owns_lm_head
    );

    response.output_tensor = {};
    response.output_metadata_json = request.input_metadata_json;
    response.backend_metadata_json = backend_metadata_json();

    const auto end = std::chrono::steady_clock::now();

    response.metrics.backend = backend_name();
    response.metrics.status = "terminal_partition_executed_layers_norm_lm_head";
    response.metrics.compute_time_ms = elapsed_ms(start, end);
    response.metrics.true_comm_ms = 0.0;
    response.metrics.rpc_wall_time_ms = 0.0;

    response.metrics.input_tensor_bytes = request.input_tensor.bytes.size();
    response.metrics.output_tensor_bytes = 0;

    response.metrics.stage_input_token_count =
        static_cast<int>(dli::common::dli2_sequence_length(request.input_tensor));
    response.metrics.stage_output_token_count = 0;

    response.metrics.kv_cache_step_valid = kv_step.valid;
    response.metrics.kv_cache_seq_before = kv_step.seq_before;
    response.metrics.kv_cache_seq_after = kv_step.seq_after;
    response.metrics.kv_cache_bytes = executor_->kv_cache_bytes();
    response.metrics.kv_cache_valid = kv_step.valid;

    response.metrics.model_load_ms = model_load_ms_;
    response.metrics.memory_rss_mb = current_rss_mb();

    return response;
}


std::string LlamaPartialRuntime::backend_name() const {
    return "llama.cpp-partial-dli-gguf";
}

const LlamaPartialRuntimeConfig& LlamaPartialRuntime::config() const {
    return config_;
}

const LlamaModelMetadata& LlamaPartialRuntime::model_metadata() const {
    return model_metadata_;
}

const LlamaShardMetadata& LlamaPartialRuntime::shard_metadata() const {
    return shard_metadata_;
}

std::string LlamaPartialRuntime::backend_metadata_json() const {
    std::ostringstream out;

    out
        << "{"
        << "\"model_loaded\":" << (model_metadata_.model_loaded ? "true" : "false") << ","
        << "\"model_path\":\"" << dli::common::json_escape(model_metadata_.model_path) << "\","
        << "\"architecture\":\"" << dli::common::json_escape(model_metadata_.architecture) << "\","
        << "\"name\":\"" << dli::common::json_escape(model_metadata_.name) << "\","
        << "\"n_ctx_train\":" << model_metadata_.n_ctx_train << ","
        << "\"n_embd\":" << model_metadata_.n_embd << ","
        << "\"n_layer\":" << model_metadata_.n_layer << ","
        << "\"n_head\":" << model_metadata_.n_head << ","
        << "\"n_head_kv\":" << model_metadata_.n_head_kv << ","
        << "\"n_vocab\":" << model_metadata_.n_vocab << ","
        << "\"shard\":{"
        << "\"loaded\":" << (shard_metadata_.shard_loaded ? "true" : "false") << ","
        << "\"format\":\"" << dli::common::json_escape(shard_metadata_.format) << "\","
        << "\"format_version\":" << shard_metadata_.format_version << ","
        << "\"partition_id\":\"" << dli::common::json_escape(shard_metadata_.partition_id) << "\","
        << "\"stage_id\":" << shard_metadata_.stage_id << ","
        << "\"layers\":" << int_vector_json(shard_metadata_.layers) << ","
        << "\"owns_embedding\":" << (shard_metadata_.owns_embedding ? "true" : "false") << ","
        << "\"owns_norm\":" << (shard_metadata_.owns_norm ? "true" : "false") << ","
        << "\"owns_lm_head\":" << (shard_metadata_.owns_lm_head ? "true" : "false") << ","
        << "\"next_partition_id\":\"" << dli::common::json_escape(shard_metadata_.next_partition_id) << "\","
        << "\"next_stage_url\":\"" << dli::common::json_escape(shard_metadata_.next_stage_url) << "\","
        << "\"hidden_size\":" << shard_metadata_.hidden_size << ","
        << "\"source_model\":\"" << dli::common::json_escape(shard_metadata_.source_model) << "\""
        << "},"
        << "\"kv_cache\":{"
        << "\"seq_len\":" << kv_cache_.seq_len << ","
        << "\"bytes\":" << estimate_kv_cache_bytes(kv_cache_.seq_len) << ","
        << "\"valid\":" << (kv_cache_.valid ? "true" : "false")
        << "}"
        << "}";

    return out.str();
}

} // namespace dli_stage