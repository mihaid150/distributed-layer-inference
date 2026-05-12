#include "dli/common/gguf_inspector.hpp"
#include "dli/common/json_escape.hpp"
#include "dli/common/partition_plan.hpp"
#include "dli/common/partition_tensor_assignment.hpp"
#include "dli/gateway/config.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::string config_path = "configs/stage_map.yaml";
    std::string model_path;
    std::string output_dir = "build/dli-partitions";
    bool allow_missing = false;
};

void print_usage(const char* program_name) {
    std::cerr
        << "Usage: " << program_name
        << " --config <stage_map.yaml> --model <model.gguf> --output-dir <dir>\n\n"
        << "Options:\n"
        << "  --config <path>      Path to stage_map.yaml. Default: configs/stage_map.yaml\n"
        << "  --model <path>       Path to full GGUF model file.\n"
        << "  --output-dir <dir>   Output directory for generated manifests.\n"
        << "  --allow-missing      Do not fail if required tensors/prefixes are missing.\n"
        << "  --help               Show this help message.\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;

    const char* env_config = std::getenv("DLI_STAGE_MAP_PATH");
    if (env_config != nullptr && std::string(env_config).size() > 0) {
        options.config_path = env_config;
    }

    const char* env_model = std::getenv("DLI_TEST_GGUF_MODEL_PATH");
    if (env_model != nullptr && std::string(env_model).size() > 0) {
        options.model_path = env_model;
    }

    const char* env_output = std::getenv("DLI_PARTITION_OUTPUT_DIR");
    if (env_output != nullptr && std::string(env_output).size() > 0) {
        options.output_dir = env_output;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }

        if (arg == "--config") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--config requires a value");
            }
            options.config_path = argv[++i];
            continue;
        }

        if (arg == "--model") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--model requires a value");
            }
            options.model_path = argv[++i];
            continue;
        }

        if (arg == "--output-dir") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--output-dir requires a value");
            }
            options.output_dir = argv[++i];
            continue;
        }

        if (arg == "--allow-missing") {
            options.allow_missing = true;
            continue;
        }

        throw std::runtime_error("unknown argument: " + arg);
    }

    if (options.model_path.empty()) {
        throw std::runtime_error("--model is required or set DLI_TEST_GGUF_MODEL_PATH");
    }

    if (options.output_dir.empty()) {
        throw std::runtime_error("--output-dir must not be empty");
    }

    return options;
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

std::string string_vector_json(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << "[";

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }

        out << "\"" << dli::common::json_escape(values[i]) << "\"";
    }

    out << "]";
    return out.str();
}

bool has_missing_tensors(
    const dli::common::PartitionTensorAssignment& assignment
) {
    return
        !assignment.missing_required_names.empty() ||
        !assignment.missing_required_prefixes.empty();
}

std::string partition_manifest_json(
    const dli::gateway::PartitionNodeConfig& partition,
    const dli::common::PartitionTensorAssignment& assignment
) {
    std::ostringstream out;

    out
        << "{"
        << "\"partition_id\":\"" << dli::common::json_escape(partition.partition_id) << "\","
        << "\"stage_id\":" << partition.stage_id << ","
        << "\"service_name\":\"" << dli::common::json_escape(partition.service_name) << "\","
        << "\"physical_node\":\"" << dli::common::json_escape(partition.physical_node) << "\","
        << "\"partition_file\":\"" << dli::common::json_escape(partition.partition_file) << "\","
        << "\"next_stage_url\":\"" << dli::common::json_escape(partition.next_stage_url) << "\","
        << "\"components\":{"
        << "\"embedding\":" << (partition.components.embedding ? "true" : "false") << ","
        << "\"layers\":" << int_vector_json(partition.components.layers) << ","
        << "\"norm\":" << (partition.components.norm ? "true" : "false") << ","
        << "\"lm_head\":" << (partition.components.lm_head ? "true" : "false")
        << "},"
        << "\"tensor_count\":" << assignment.tensor_names.size() << ","
        << "\"missing_required_names\":" << string_vector_json(assignment.missing_required_names) << ","
        << "\"missing_required_prefixes\":" << string_vector_json(assignment.missing_required_prefixes) << ","
        << "\"tensor_names\":" << string_vector_json(assignment.tensor_names)
        << "}";

    return out.str();
}

void write_text_file(
    const std::filesystem::path& path,
    const std::string& content
) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }

    out << content;
}

void write_tensor_list(
    const std::filesystem::path& path,
    const std::vector<std::string>& tensor_names
) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open tensor list file: " + path.string());
    }

    for (const auto& name : tensor_names) {
        out << name << "\n";
    }
}

std::string main_manifest_json(
    const CliOptions& options,
    const dli::gateway::GatewayConfig& config,
    const dli::common::GgufInspection& inspection,
    const std::vector<dli::common::PartitionTensorAssignment>& assignments
) {
    std::ostringstream out;

    out
        << "{"
        << "\"ok\":true,"
        << "\"config_path\":\"" << dli::common::json_escape(options.config_path) << "\","
        << "\"model_path\":\"" << dli::common::json_escape(options.model_path) << "\","
        << "\"model_name\":\"" << dli::common::json_escape(config.model_name) << "\","
        << "\"num_layers\":" << config.num_layers << ","
        << "\"gguf_tensor_count\":" << inspection.tensor_count << ","
        << "\"partition_count\":" << config.partitions.size() << ","
        << "\"partitions\":[";

    for (std::size_t i = 0; i < assignments.size(); ++i) {
        if (i > 0) {
            out << ",";
        }

        const auto& partition = config.partitions[i];
        const auto& assignment = assignments[i];

        out
            << "{"
            << "\"partition_id\":\"" << dli::common::json_escape(partition.partition_id) << "\","
            << "\"stage_id\":" << partition.stage_id << ","
            << "\"service_name\":\"" << dli::common::json_escape(partition.service_name) << "\","
            << "\"tensor_count\":" << assignment.tensor_names.size() << ","
            << "\"missing_required_names\":" << string_vector_json(assignment.missing_required_names) << ","
            << "\"missing_required_prefixes\":" << string_vector_json(assignment.missing_required_prefixes) << ","
            << "\"manifest_file\":\"" << dli::common::json_escape(partition.partition_id + ".manifest.json") << "\","
            << "\"tensor_list_file\":\"" << dli::common::json_escape(partition.partition_id + ".tensors.txt") << "\""
            << "}";
    }

    out << "]}";
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);

        const dli::gateway::GatewayConfig config =
            dli::gateway::load_gateway_config_from_file(options.config_path);

        const dli::common::GgufInspection inspection =
            dli::common::inspect_gguf_tensors(options.model_path);

        std::vector<dli::common::TensorNamePlan> plans;
        plans.reserve(config.partitions.size());

        for (const auto& partition : config.partitions) {
            plans.push_back(
                dli::common::build_llama_tensor_name_plan(
                    partition.partition_id,
                    to_common_components(partition.components)
                )
            );
        }

        const std::vector<dli::common::PartitionTensorAssignment> assignments =
            dli::common::assign_tensors_to_partitions(inspection, plans);

        bool has_missing = false;

        for (const auto& assignment : assignments) {
            if (has_missing_tensors(assignment)) {
                has_missing = true;
                break;
            }
        }

        if (has_missing && !options.allow_missing) {
            std::cerr
                << "dli-partition-manifest refused to write because some tensors are missing.\n"
                << "Run dli-partition-plan for details, or pass --allow-missing.\n";
            return 2;
        }

        const std::filesystem::path output_dir(options.output_dir);
        std::filesystem::create_directories(output_dir);

        for (std::size_t i = 0; i < assignments.size(); ++i) {
            const auto& partition = config.partitions[i];
            const auto& assignment = assignments[i];

            write_text_file(
                output_dir / (partition.partition_id + ".manifest.json"),
                partition_manifest_json(partition, assignment) + "\n"
            );

            write_tensor_list(
                output_dir / (partition.partition_id + ".tensors.txt"),
                assignment.tensor_names
            );
        }

        const std::string manifest =
            main_manifest_json(options, config, inspection, assignments);

        write_text_file(output_dir / "manifest.json", manifest + "\n");

        std::cout << manifest << "\n";

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dli-partition-manifest failed: " << exc.what() << "\n";
        return 1;
    }
}