#include "dli/common/json_escape.hpp"

#include "gguf.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::string model_path;
};

void print_usage(const char* program_name) {
    std::cerr
        << "Usage: " << program_name << " --model <model.gguf>\n\n"
        << "Options:\n"
        << "  --model <path>  Path to a full GGUF model or DLI GGUF shard.\n"
        << "  --help          Show this help message.\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }

        if (arg == "--model") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--model requires a value");
            }
            options.model_path = argv[++i];
            continue;
        }

        throw std::runtime_error("unknown argument: " + arg);
    }

    if (options.model_path.empty()) {
        throw std::runtime_error("--model is required");
    }

    return options;
}

int find_key(const gguf_context* ctx, const std::string& key) {
    return gguf_find_key(ctx, key.c_str());
}

std::string string_key_or_empty(const gguf_context* ctx, const std::string& key) {
    const int key_id = find_key(ctx, key);
    if (key_id < 0) {
        return "";
    }

    const gguf_type type = gguf_get_kv_type(ctx, key_id);
    if (type != GGUF_TYPE_STRING) {
        return "";
    }

    const char* value = gguf_get_val_str(ctx, key_id);
    return value != nullptr ? value : "";
}

int int_key_or_default(
    const gguf_context* ctx,
    const std::vector<std::string>& keys,
    int default_value
) {
    for (const auto& key : keys) {
        const int key_id = find_key(ctx, key);
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
                throw std::runtime_error("GGUF integer metadata too large: " + key);
            }
            return static_cast<int>(value);
        }
    }

    return default_value;
}

std::string json_string(const std::string& value) {
    return "\"" + dli::common::json_escape(value) + "\"";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);

        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = nullptr;

        gguf_context* raw_ctx = gguf_init_from_file(options.model_path.c_str(), params);
        if (raw_ctx == nullptr) {
            throw std::runtime_error("failed to open GGUF file: " + options.model_path);
        }

        std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(raw_ctx, gguf_free);

        const std::string architecture =
            string_key_or_empty(ctx.get(), "general.architecture");
        const std::string name =
            string_key_or_empty(ctx.get(), "general.name");

        const int block_count = int_key_or_default(
            ctx.get(),
            {
                "llama.block_count",
                "mistral.block_count",
                "qwen2.block_count",
                "gptneox.block_count"
            },
            -1
        );

        const int context_length = int_key_or_default(
            ctx.get(),
            {
                "llama.context_length",
                "mistral.context_length",
                "qwen2.context_length",
                "gptneox.context_length"
            },
            -1
        );

        const int embedding_length = int_key_or_default(
            ctx.get(),
            {
                "llama.embedding_length",
                "mistral.embedding_length",
                "qwen2.embedding_length",
                "gptneox.embedding_length"
            },
            -1
        );

        const int head_count = int_key_or_default(
            ctx.get(),
            {
                "llama.attention.head_count",
                "mistral.attention.head_count",
                "qwen2.attention.head_count",
                "gptneox.attention.head_count"
            },
            -1
        );

        const int head_count_kv = int_key_or_default(
            ctx.get(),
            {
                "llama.attention.head_count_kv",
                "mistral.attention.head_count_kv",
                "qwen2.attention.head_count_kv",
                "gptneox.attention.head_count_kv"
            },
            -1
        );

        std::cout
            << "{"
            << "\"ok\":true,"
            << "\"model_path\":" << json_string(options.model_path) << ","
            << "\"architecture\":" << json_string(architecture) << ","
            << "\"name\":" << json_string(name) << ","
            << "\"block_count\":" << block_count << ","
            << "\"context_length\":" << context_length << ","
            << "\"embedding_length\":" << embedding_length << ","
            << "\"head_count\":" << head_count << ","
            << "\"head_count_kv\":" << head_count_kv
            << "}\n";

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dli-gguf-info failed: " << exc.what() << "\n";
        return 1;
    }
}
