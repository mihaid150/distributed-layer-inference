#include "dli/tools/manifest_json.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::string model_path;
    std::string manifest_path;
    std::string manifest_dir;
    std::string output_path;
    std::string output_dir = "build/dli-shards";
    std::string source_model;
};

struct GgufContextDeleter {
    void operator()(gguf_context* ctx) const {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }
};

struct GgmlContextDeleter {
    void operator()(ggml_context* ctx) const {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

using GgufContextPtr = std::unique_ptr<gguf_context, GgufContextDeleter>;
using GgmlContextPtr = std::unique_ptr<ggml_context, GgmlContextDeleter>;

void print_usage(const char* program_name) {
    std::cerr
        << "Usage:\n"
        << "  " << program_name << " --model <full.gguf> --manifest <partition.manifest.json> --output <partition.dli.gguf>\n"
        << "  " << program_name << " --model <full.gguf> --manifest-dir <dir> --output-dir <dir>\n\n"
        << "Options:\n"
        << "  --model <path>          Full source GGUF model.\n"
        << "  --manifest <path>       Single partition manifest JSON.\n"
        << "  --manifest-dir <dir>    Directory containing partition-*.manifest.json files.\n"
        << "  --output <path>         Output path for a single shard.\n"
        << "  --output-dir <dir>      Output directory for directory mode. Default: build/dli-shards\n"
        << "  --source-model <value>  Metadata value for dli.source_model. Default: --model path.\n"
        << "  --help                  Show this help message.\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;

    const char* env_model = std::getenv("DLI_TEST_GGUF_MODEL_PATH");
    if (env_model != nullptr && std::string(env_model).size() > 0) {
        options.model_path = env_model;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }

        auto require_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(flag + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--model") {
            options.model_path = require_value(arg);
            continue;
        }

        if (arg == "--manifest") {
            options.manifest_path = require_value(arg);
            continue;
        }

        if (arg == "--manifest-dir") {
            options.manifest_dir = require_value(arg);
            continue;
        }

        if (arg == "--output") {
            options.output_path = require_value(arg);
            continue;
        }

        if (arg == "--output-dir") {
            options.output_dir = require_value(arg);
            continue;
        }

        if (arg == "--source-model") {
            options.source_model = require_value(arg);
            continue;
        }

        throw std::runtime_error("unknown argument: " + arg);
    }

    if (options.model_path.empty()) {
        throw std::runtime_error("--model is required or set DLI_TEST_GGUF_MODEL_PATH");
    }

    const bool single_mode = !options.manifest_path.empty();
    const bool directory_mode = !options.manifest_dir.empty();

    if (single_mode == directory_mode) {
        throw std::runtime_error("pass exactly one of --manifest or --manifest-dir");
    }

    if (single_mode && options.output_path.empty()) {
        throw std::runtime_error("--output is required with --manifest");
    }

    if (directory_mode && options.output_dir.empty()) {
        throw std::runtime_error("--output-dir must not be empty");
    }

    if (options.source_model.empty()) {
        options.source_model = options.model_path;
    }

    return options;
}

std::vector<std::filesystem::path> list_partition_manifest_files(
    const std::filesystem::path& dir
) {
    if (!std::filesystem::is_directory(dir)) {
        throw std::runtime_error("manifest-dir is not a directory: " + dir.string());
    }

    std::vector<std::filesystem::path> manifests;

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string filename = entry.path().filename().string();

        if (
            filename.rfind("partition-", 0) == 0 &&
            filename.size() >= std::string(".manifest.json").size() &&
            filename.substr(filename.size() - std::string(".manifest.json").size()) == ".manifest.json"
        ) {
            manifests.push_back(entry.path());
        }
    }

    std::sort(manifests.begin(), manifests.end());

    if (manifests.empty()) {
        throw std::runtime_error("no partition-*.manifest.json files found in: " + dir.string());
    }

    return manifests;
}

int read_i32_metadata_or_default(
    const gguf_context* ctx,
    const std::vector<std::string>& candidate_keys,
    int default_value
) {
    for (const auto& key : candidate_keys) {
        const int key_id = gguf_find_key(ctx, key.c_str());
        if (key_id < 0) {
            continue;
        }

        const gguf_type type = gguf_get_kv_type(ctx, key_id);

        if (type == GGUF_TYPE_INT32) {
            return gguf_get_val_i32(ctx, key_id);
        }

        if (type == GGUF_TYPE_UINT32) {
            const std::uint32_t value = gguf_get_val_u32(ctx, key_id);

            if (value > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error(
                    "GGUF metadata value is too large for int: " + key
                );
            }

            return static_cast<int>(value);
        }

        throw std::runtime_error(
            "GGUF metadata key has unsupported integer type for this writer: " + key
        );
    }

    return default_value;
}

void add_dli_metadata(
    gguf_context* dst,
    const dli::tools::PartitionManifest& manifest,
    const gguf_context* src,
    const std::string& source_model
) {
    gguf_set_val_str(dst, "dli.format", "dli.gguf.stage_shard");
    gguf_set_val_i32(dst, "dli.format_version", 1);

    gguf_set_val_str(dst, "dli.partition_id", manifest.partition_id.c_str());
    gguf_set_val_i32(dst, "dli.stage_id", manifest.stage_id);

    std::vector<int32_t> layers_i32;
    layers_i32.reserve(manifest.layers.size());
    for (const int layer : manifest.layers) {
        layers_i32.push_back(static_cast<int32_t>(layer));
    }

    gguf_set_arr_data(
        dst,
        "dli.layers",
        GGUF_TYPE_INT32,
        layers_i32.data(),
        layers_i32.size()
    );

    gguf_set_val_bool(dst, "dli.owns_embedding", manifest.owns_embedding);
    gguf_set_val_bool(dst, "dli.owns_norm", manifest.owns_norm);
    gguf_set_val_bool(dst, "dli.owns_lm_head", manifest.owns_lm_head);

    // The manifest currently contains the route URL, not the next static
    // partition id. Keep this key present and empty until the allocation graph
    // explicitly serializes next_partition_id.
    gguf_set_val_str(dst, "dli.next_partition_id", "");
    gguf_set_val_str(dst, "dli.next_stage_url", manifest.next_stage_url.c_str());

    const int hidden_size = read_i32_metadata_or_default(
        src,
        {
            "llama.embedding_length",
            "gptneox.embedding_length",
            "mistral.embedding_length"
        },
        -1
    );

    gguf_set_val_i32(dst, "dli.hidden_size", hidden_size);
    gguf_set_val_str(dst, "dli.source_model", source_model.c_str());
}

void write_one_shard(
    const std::string& model_path,
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& output_path,
    const std::string& source_model
) {
    const dli::tools::PartitionManifest manifest =
        dli::tools::load_partition_manifest(manifest_path);

    ggml_context* raw_data_ctx = nullptr;

    gguf_init_params params{};
    params.no_alloc = false;
    params.ctx = &raw_data_ctx;

    GgufContextPtr src(gguf_init_from_file(model_path.c_str(), params));
    GgmlContextPtr data_ctx(raw_data_ctx);

    if (src == nullptr) {
        throw std::runtime_error("failed to load source GGUF: " + model_path);
    }

    GgufContextPtr dst(gguf_init_empty());
    if (dst == nullptr) {
        throw std::runtime_error("failed to create empty output GGUF context");
    }

    // Preserve architecture, tokenizer, quantization, and all other source metadata.
    // DLI keys are then added/overridden below.
    gguf_set_kv(dst.get(), src.get());

    add_dli_metadata(dst.get(), manifest, src.get(), source_model);

    for (const auto& tensor_name : manifest.tensor_names) {
        ggml_tensor* tensor = ggml_get_tensor(data_ctx.get(), tensor_name.c_str());
        if (tensor == nullptr) {
            throw std::runtime_error(
                "manifest tensor not found in source GGUF data context: " + tensor_name
            );
        }

        gguf_add_tensor(dst.get(), tensor);
        gguf_set_tensor_data(dst.get(), tensor_name.c_str(), tensor->data);
    }

    std::filesystem::create_directories(output_path.parent_path());

    gguf_write_to_file(dst.get(), output_path.string().c_str(), false);

    std::cout
        << "{"
        << "\"ok\":true,"
        << "\"partition_id\":\"" << manifest.partition_id << "\","
        << "\"stage_id\":" << manifest.stage_id << ","
        << "\"tensor_count\":" << manifest.tensor_names.size() << ","
        << "\"output\":\"" << output_path.string() << "\""
        << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);

        if (!options.manifest_path.empty()) {
            write_one_shard(
                options.model_path,
                options.manifest_path,
                options.output_path,
                options.source_model
            );

            return 0;
        }

        const std::filesystem::path manifest_dir(options.manifest_dir);
        const std::filesystem::path output_dir(options.output_dir);

        std::filesystem::create_directories(output_dir);

        for (const auto& manifest_path : list_partition_manifest_files(manifest_dir)) {
            const dli::tools::PartitionManifest manifest =
                dli::tools::load_partition_manifest(manifest_path);

            const std::filesystem::path output_path =
                output_dir / (manifest.partition_id + ".dli.gguf");

            write_one_shard(
                options.model_path,
                manifest_path,
                output_path,
                options.source_model
            );
        }

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dli-gguf-shard-writer failed: " << exc.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
