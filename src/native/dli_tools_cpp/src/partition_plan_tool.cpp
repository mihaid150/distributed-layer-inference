#include "dli/common/gguf_inspector.hpp"
#include "dli/common/json_escape.hpp"
#include "dli/common/partition_plan.hpp"
#include "dli/common/partition_tensor_assignment.hpp"
#include "dli/gateway/config.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::string config_path = "configs/stage_map.yaml";
    std::string model_path;
};

void print_usage(const char* program_name) {
    std::cerr
        << "Usage: " << program_name << " --config <stage_map.yaml> --model <model.gguf>\n\n"
        << "Options:\n"
        << "  --config <path>   Path to stage_map.yaml. Default: configs/stage_map.yaml\n"
        << "  --model <path>    Path to full GGUF model file.\n"
        << "  --help            Show this help message.\n";
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

        throw std::runtime_error("unknown argument: " + arg);
    }

    if (options.model_path.empty()) {
        throw std::runtime_error("--model is required or set DLI_TEST_GGUF_MODEL_PATH");
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

std::string partition_json(
    const dli::common::PartitionTensorAssignment& assignment
) {
    std::ostringstream out;

    out
        << "{"
        << "\"partition_id\":\"" << dli::common::json_escape(assignment.partition_id) << "\","
        << "\"tensor_count\":" << assignment.tensor_names.size() << ","
        << "\"missing_required_names\":" << string_vector_json(assignment.missing_required_names) << ","
        << "\"missing_required_prefixes\":" << string_vector_json(assignment.missing_required_prefixes)
        << "}";

    return out.str();
}

std::string report_json(
    const CliOptions& options,
    const dli::gateway::GatewayConfig& config,
    const dli::common::GgufInspection& inspection
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

    for (std::size_t i = 0; i < config.partitions.size(); ++i) {
        if (i > 0) {
            out << ",";
        }

        const auto& partition = config.partitions[i];

        const auto plan = dli::common::build_llama_tensor_name_plan(
            partition.partition_id,
            to_common_components(partition.components)
        );

        const auto assignment = dli::common::assign_tensors_to_partition(
            inspection,
            plan
        );

        out << partition_json(assignment);
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

        std::cout << report_json(options, config, inspection) << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dli-partition-plan failed: " << exc.what() << "\n";
        return 1;
    }
}