#include "dli/common/gguf_inspector.hpp"
#include "dli/common/partition_plan.hpp"
#include "dli/common/partition_tensor_assignment.hpp"
#include "dli/gateway/config.hpp"
#include "dli/tools/manifest_json.hpp"

#include "gguf.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::string config_path = "configs/stage_map.yaml";
    std::string manifest_path;
    std::string manifest_dir;
    std::string shard_path;
    std::string shard_dir;
};

void print_usage(const char* program_name) {
    std::cerr
        << "Usage:\n"
        << "  " << program_name << " --config <stage_map.yaml> --manifest <partition.manifest.json> --shard <partition.dli.gguf>\n"
        << "  " << program_name << " --config <stage_map.yaml> --manifest-dir <dir> --shard-dir <dir>\n\n"
        << "Options:\n"
        << "  --config <path>        Path to stage_map.yaml. Default: configs/stage_map.yaml\n"
        << "  --manifest <path>      Single partition manifest JSON.\n"
        << "  --manifest-dir <dir>   Directory containing partition-*.manifest.json files.\n"
        << "  --shard <path>         Single .dli.gguf shard path.\n"
        << "  --shard-dir <dir>      Directory containing partition-*.dli.gguf files.\n"
        << "  --help                 Show this help message.\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;

    const char* env_config = std::getenv("DLI_STAGE_MAP_PATH");
    if (env_config != nullptr && std::string(env_config).size() > 0) {
        options.config_path = env_config;
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

        if (arg == "--config") {
            options.config_path = require_value(arg);
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

        if (arg == "--shard") {
            options.shard_path = require_value(arg);
            continue;
        }

        if (arg == "--shard-dir") {
            options.shard_dir = require_value(arg);
            continue;
        }

        throw std::runtime_error("unknown argument: " + arg);
    }

    const bool single_mode = !options.manifest_path.empty() || !options.shard_path.empty();
    const bool directory_mode = !options.manifest_dir.empty() || !options.shard_dir.empty();

    if (single_mode == directory_mode) {
        throw std::runtime_error("use either single mode or directory mode");
    }

    if (single_mode && (options.manifest_path.empty() || options.shard_path.empty())) {
        throw std::runtime_error("single mode requires both --manifest and --shard");
    }

    if (directory_mode && (options.manifest_dir.empty() || options.shard_dir.empty())) {
        throw std::runtime_error("directory mode requires both --manifest-dir and --shard-dir");
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

const dli::gateway::PartitionNodeConfig& find_partition_config(
    const dli::gateway::GatewayConfig& config,
    const std::string& partition_id
) {
    for (const auto& partition : config.partitions) {
        if (partition.partition_id == partition_id) {
            return partition;
        }
    }

    throw std::runtime_error("partition not found in stage_map.yaml: " + partition_id);
}

dli::common::PartitionComponentsPlan to_common_components(
    const dli::gateway::PartitionComponents& components
) {
    dli::common::PartitionComponentsPlan common;
    common.embedding = components.embedding;
    common.layers = components.layers;
    common.norm = components.norm;
    common.lm_head = components.lm_head;
    return common;
}

std::string join_strings(const std::vector<std::string>& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << values[i];
    }
    return out.str();
}

std::vector<std::string> sorted_copy(std::vector<std::string> values) {
    std::sort(values.begin(), values.end());
    return values;
}

void require_tensor_sets_equal(
    const std::vector<std::string>& actual,
    const std::vector<std::string>& expected,
    const std::string& label
) {
    const std::vector<std::string> a = sorted_copy(actual);
    const std::vector<std::string> e = sorted_copy(expected);

    std::vector<std::string> missing;
    std::vector<std::string> extra;

    std::set_difference(e.begin(), e.end(), a.begin(), a.end(), std::back_inserter(missing));
    std::set_difference(a.begin(), a.end(), e.begin(), e.end(), std::back_inserter(extra));

    if (!missing.empty() || !extra.empty()) {
        throw std::runtime_error(
            label +
            " tensor set mismatch; missing=[" + join_strings(missing) +
            "], extra=[" + join_strings(extra) + "]"
        );
    }
}

bool has_metadata_key(const gguf_context* ctx, const std::string& key) {
    return gguf_find_key(ctx, key.c_str()) >= 0;
}

int require_key(const gguf_context* ctx, const std::string& key) {
    const int key_id = gguf_find_key(ctx, key.c_str());
    if (key_id < 0) {
        throw std::runtime_error("missing required GGUF metadata key: " + key);
    }
    return key_id;
}

std::string require_string_key(const gguf_context* ctx, const std::string& key) {
    const int key_id = require_key(ctx, key);
    const char* value = gguf_get_val_str(ctx, key_id);
    return value != nullptr ? value : "";
}

int require_i32_key(const gguf_context* ctx, const std::string& key) {
    return gguf_get_val_i32(ctx, require_key(ctx, key));
}

bool require_bool_key(const gguf_context* ctx, const std::string& key) {
    return gguf_get_val_bool(ctx, require_key(ctx, key));
}

std::vector<int> require_i32_array_key(const gguf_context* ctx, const std::string& key) {
    const int key_id = require_key(ctx, key);

    if (gguf_get_arr_type(ctx, key_id) != GGUF_TYPE_INT32) {
        throw std::runtime_error("metadata key is not array<int32>: " + key);
    }

    const int64_t n = gguf_get_arr_n(ctx, key_id);
    const void* raw = gguf_get_arr_data(ctx, key_id);

    if (n < 0 || raw == nullptr) {
        throw std::runtime_error("metadata key has invalid array data: " + key);
    }

    const int32_t* data = static_cast<const int32_t*>(raw);

    std::vector<int> values;
    values.reserve(static_cast<std::size_t>(n));

    for (int64_t i = 0; i < n; ++i) {
        values.push_back(static_cast<int>(data[i]));
    }

    return values;
}

bool has_tokenizer_metadata(const gguf_context* ctx) {
    const int64_t n_kv = gguf_get_n_kv(ctx);

    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(ctx, i);
        if (key == nullptr) {
            continue;
        }

        const std::string key_string(key);
        if (key_string.rfind("tokenizer.ggml.", 0) == 0) {
            return true;
        }
    }

    return false;
}

void validate_metadata(
    const std::filesystem::path& shard_path,
    const dli::tools::PartitionManifest& manifest,
    const dli::gateway::PartitionNodeConfig& partition_config
) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;

    gguf_context* raw_ctx = gguf_init_from_file(shard_path.string().c_str(), params);
    if (raw_ctx == nullptr) {
        throw std::runtime_error("failed to open GGUF shard: " + shard_path.string());
    }

    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(raw_ctx, gguf_free);

    if (require_string_key(ctx.get(), "dli.format") != "dli.gguf.stage_shard") {
        throw std::runtime_error("invalid dli.format");
    }

    if (require_i32_key(ctx.get(), "dli.format_version") != 1) {
        throw std::runtime_error("unsupported dli.format_version");
    }

    if (require_string_key(ctx.get(), "dli.partition_id") != manifest.partition_id) {
        throw std::runtime_error("dli.partition_id does not match manifest");
    }

    if (require_i32_key(ctx.get(), "dli.stage_id") != manifest.stage_id) {
        throw std::runtime_error("dli.stage_id does not match manifest");
    }

    if (manifest.stage_id != partition_config.stage_id) {
        throw std::runtime_error("manifest stage_id does not match stage_map partition");
    }

    if (require_bool_key(ctx.get(), "dli.owns_embedding") != partition_config.components.embedding) {
        throw std::runtime_error("dli.owns_embedding does not match stage_map");
    }

    if (require_bool_key(ctx.get(), "dli.owns_norm") != partition_config.components.norm) {
        throw std::runtime_error("dli.owns_norm does not match stage_map");
    }

    if (require_bool_key(ctx.get(), "dli.owns_lm_head") != partition_config.components.lm_head) {
        throw std::runtime_error("dli.owns_lm_head does not match stage_map");
    }

    if (require_i32_array_key(ctx.get(), "dli.layers") != partition_config.components.layers) {
        throw std::runtime_error("dli.layers does not match stage_map");
    }

    if (require_string_key(ctx.get(), "dli.next_stage_url") != partition_config.next_stage_url) {
        throw std::runtime_error("dli.next_stage_url does not match stage_map");
    }

    require_key(ctx.get(), "dli.next_partition_id");
    require_key(ctx.get(), "dli.hidden_size");
    require_key(ctx.get(), "dli.source_model");

    require_key(ctx.get(), "general.architecture");

    if (!has_tokenizer_metadata(ctx.get())) {
        throw std::runtime_error("shard does not preserve tokenizer.ggml.* metadata");
    }
}

void validate_one_shard(
    const dli::gateway::GatewayConfig& config,
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& shard_path
) {
    const dli::tools::PartitionManifest manifest =
        dli::tools::load_partition_manifest(manifest_path);

    const dli::gateway::PartitionNodeConfig& partition_config =
        find_partition_config(config, manifest.partition_id);

    validate_metadata(shard_path, manifest, partition_config);

    const dli::common::GgufInspection inspection =
        dli::common::inspect_gguf_tensors(shard_path.string());

    require_tensor_sets_equal(
        inspection.tensors.empty()
            ? std::vector<std::string>{}
            : [&]() {
                std::vector<std::string> names;
                names.reserve(inspection.tensors.size());
                for (const auto& tensor : inspection.tensors) {
                    names.push_back(tensor.name);
                }
                return names;
            }(),
        manifest.tensor_names,
        manifest.partition_id + " manifest"
    );

    const dli::common::TensorNamePlan expected_plan =
        dli::common::build_llama_tensor_name_plan(
            manifest.partition_id,
            to_common_components(partition_config.components)
        );

    const dli::common::PartitionTensorAssignment assignment =
        dli::common::assign_tensors_to_partition(inspection, expected_plan);

    if (!assignment.missing_required_names.empty()) {
        throw std::runtime_error(
            manifest.partition_id +
            " missing required tensor names: " +
            join_strings(assignment.missing_required_names)
        );
    }

    if (!assignment.missing_required_prefixes.empty()) {
        throw std::runtime_error(
            manifest.partition_id +
            " missing required tensor prefixes: " +
            join_strings(assignment.missing_required_prefixes)
        );
    }

    require_tensor_sets_equal(
        [&]() {
            std::vector<std::string> names;
            names.reserve(inspection.tensors.size());
            for (const auto& tensor : inspection.tensors) {
                names.push_back(tensor.name);
            }
            return names;
        }(),
        assignment.tensor_names,
        manifest.partition_id + " stage_map"
    );

    std::cout
        << "{"
        << "\"ok\":true,"
        << "\"partition_id\":\"" << manifest.partition_id << "\","
        << "\"stage_id\":" << manifest.stage_id << ","
        << "\"tensor_count\":" << inspection.tensor_count << ","
        << "\"shard\":\"" << shard_path.string() << "\""
        << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);

        const dli::gateway::GatewayConfig config =
            dli::gateway::load_gateway_config_from_file(options.config_path);

        if (!options.manifest_path.empty()) {
            validate_one_shard(config, options.manifest_path, options.shard_path);
            return 0;
        }

        const std::filesystem::path manifest_dir(options.manifest_dir);
        const std::filesystem::path shard_dir(options.shard_dir);

        for (const auto& manifest_path : list_partition_manifest_files(manifest_dir)) {
            const dli::tools::PartitionManifest manifest =
                dli::tools::load_partition_manifest(manifest_path);

            const std::filesystem::path shard_path =
                shard_dir / (manifest.partition_id + ".dli.gguf");

            validate_one_shard(config, manifest_path, shard_path);
        }

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dli-gguf-shard-validate failed: " << exc.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
