#include "dli/gateway/config.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace dli::gateway {

namespace {

std::string trim_copy(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() && (value[begin] == ' ' || value[begin] == '\t')) {
        ++begin;
    }

    std::size_t end = value.size();
    while (
        end > begin &&
        (value[end - 1] == ' ' || value[end - 1] == '\t' || value[end - 1] == '\r')
    ) {
        --end;
    }

    return value.substr(begin, end - begin);
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

int indent_width(const std::string& line) {
    int count = 0;
    for (char c : line) {
        if (c == ' ') {
            ++count;
        } else {
            break;
        }
    }
    return count;
}

std::string value_after_colon(const std::string& line) {
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
        return "";
    }

    std::string value = trim_copy(line.substr(colon + 1));

    if (value == "null") {
        return "";
    }

    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
    }

    if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'') {
        value = value.substr(1, value.size() - 2);
    }

    return value;
}

bool parse_bool_value(const std::string& value) {
    return value == "true" || value == "True" || value == "yes" || value == "1";
}

bool is_stage_list_item(const std::string& trimmed) {
    return starts_with(trimmed, "- stage_id:");
}

int parse_stage_id_from_list_item(const std::string& trimmed) {
    const std::string value = value_after_colon(trimmed);
    if (value.empty()) {
        throw std::runtime_error("stage list item missing stage_id value");
    }
    return std::stoi(value);
}

std::string default_partition_id(int stage_id) {
    return "partition-" + std::to_string(stage_id);
}

void finish_partition_if_valid(
    std::vector<PartitionNodeConfig>& partitions,
    PartitionNodeConfig& current,
    bool& has_current
) {
    if (!has_current) {
        return;
    }

    if (current.stage_id <= 0) {
        throw std::runtime_error("partition entry has invalid stage_id");
    }

    if (current.partition_id.empty()) {
        current.partition_id = default_partition_id(current.stage_id);
    }

    if (current.service_name.empty()) {
        throw std::runtime_error(
            "partition " + current.partition_id + " is missing service_name"
        );
    }

    partitions.push_back(current);

    current = PartitionNodeConfig{};
    has_current = false;
}

} // namespace

GatewayConfig load_gateway_config_from_file(const std::string& config_path) {
    GatewayConfig config;
    config.config_path = config_path;

    std::ifstream input(config_path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open gateway config file: " + config_path);
    }

    bool inside_gateway = false;
    bool inside_inference_stages = false;
    bool inside_current_partition = false;
    bool inside_components = false;
    bool inside_layers = false;

    int gateway_indent = -1;

    PartitionNodeConfig current_partition;

    std::string line;
    while (std::getline(input, line)) {
        const std::string trimmed = trim_copy(line);

        if (trimmed.empty() || starts_with(trimmed, "#")) {
            continue;
        }

        const int indent = indent_width(line);

        if (indent == 0 && starts_with(trimmed, "model_name:")) {
            config.model_name = value_after_colon(trimmed);
            continue;
        }

        if (indent == 0 && starts_with(trimmed, "model_path:")) {
            config.model_path = value_after_colon(trimmed);
            continue;
        }

        if (indent == 0 && starts_with(trimmed, "inference_gateway:")) {
            inside_gateway = true;
            inside_inference_stages = false;
            inside_current_partition = false;
            inside_components = false;
            inside_layers = false;
            gateway_indent = indent;
            continue;
        }

        if (indent == 0 && starts_with(trimmed, "inference_stages:")) {
            finish_partition_if_valid(
                config.partitions,
                current_partition,
                inside_current_partition
            );

            inside_gateway = false;
            inside_inference_stages = true;
            inside_components = false;
            inside_layers = false;
            continue;
        }

        if (
            inside_inference_stages &&
            indent == 0 &&
            !is_stage_list_item(trimmed) &&
            !starts_with(trimmed, "inference_stages:")
        ) {
            finish_partition_if_valid(
                config.partitions,
                current_partition,
                inside_current_partition
            );
            inside_inference_stages = false;
            inside_components = false;
            inside_layers = false;
            continue;
        }

        if (inside_gateway && indent <= gateway_indent && !starts_with(trimmed, "inference_gateway:")) {
            inside_gateway = false;
        }

        if (inside_gateway) {
            if (starts_with(trimmed, "service_name:")) {
                config.service_name = value_after_colon(trimmed);
                continue;
            }

            if (starts_with(trimmed, "first_stage_url:")) {
                config.first_stage_url = value_after_colon(trimmed);
                continue;
            }

            if (starts_with(trimmed, "port:")) {
                const std::string port_value = value_after_colon(trimmed);
                try {
                    config.port = std::stoi(port_value);
                } catch (const std::exception&) {
                    throw std::runtime_error("invalid gateway port in config: " + port_value);
                }
                continue;
            }
        }

        if (!inside_inference_stages) {
            continue;
        }

        if (is_stage_list_item(trimmed)) {
            finish_partition_if_valid(
                config.partitions,
                current_partition,
                inside_current_partition
            );

            const int stage_id = parse_stage_id_from_list_item(trimmed);

            current_partition = PartitionNodeConfig{};
            current_partition.stage_id = stage_id;
            current_partition.partition_id = default_partition_id(stage_id);

            inside_current_partition = true;
            inside_components = false;
            inside_layers = false;

            continue;
        }

        if (!inside_current_partition) {
            continue;
        }

        if (starts_with(trimmed, "components:")) {
            inside_components = true;
            inside_layers = false;
            continue;
        }

        if (inside_components && indent <= 2 && !starts_with(trimmed, "components:")) {
            inside_components = false;
            inside_layers = false;
        }

        if (!inside_components) {
            if (starts_with(trimmed, "partition_id:")) {
                current_partition.partition_id = value_after_colon(trimmed);
                continue;
            }

            if (starts_with(trimmed, "service_name:")) {
                current_partition.service_name = value_after_colon(trimmed);
                continue;
            }

            if (starts_with(trimmed, "physical_node:")) {
                current_partition.physical_node = value_after_colon(trimmed);
                continue;
            }

            if (starts_with(trimmed, "partition_file:")) {
                current_partition.partition_file = value_after_colon(trimmed);
                continue;
            }

            if (starts_with(trimmed, "next_stage_url:")) {
                current_partition.next_stage_url = value_after_colon(trimmed);
                continue;
            }

            continue;
        }

        if (inside_components) {
            if (starts_with(trimmed, "embedding:")) {
                current_partition.components.embedding =
                    parse_bool_value(value_after_colon(trimmed));
                continue;
            }

            if (starts_with(trimmed, "norm:")) {
                current_partition.components.norm =
                    parse_bool_value(value_after_colon(trimmed));
                continue;
            }

            if (starts_with(trimmed, "lm_head:")) {
                current_partition.components.lm_head =
                    parse_bool_value(value_after_colon(trimmed));
                continue;
            }

            if (starts_with(trimmed, "layers:")) {
                inside_layers = true;
                continue;
            }

            if (inside_layers && starts_with(trimmed, "-")) {
                const std::string item = trim_copy(trimmed.substr(1));
                if (!item.empty()) {
                    current_partition.components.layers.push_back(std::stoi(item));
                }
                continue;
            }
        }
    }

    finish_partition_if_valid(
        config.partitions,
        current_partition,
        inside_current_partition
    );

    return config;
}

GatewayConfig merge_gateway_config(
    GatewayConfig file_config,
    const GatewayConfig& override_config
) {
    if (!override_config.config_path.empty()) {
        file_config.config_path = override_config.config_path;
    }

    if (!override_config.service_name.empty()) {
        file_config.service_name = override_config.service_name;
    }

    if (!override_config.model_name.empty()) {
        file_config.model_name = override_config.model_name;
    }

    if (!override_config.model_path.empty()) {
        file_config.model_path = override_config.model_path;
    }

    if (!override_config.first_stage_url.empty()) {
        file_config.first_stage_url = override_config.first_stage_url;
    }

    if (override_config.port > 0) {
        file_config.port = override_config.port;
    }

    return file_config;
}

} // namespace dli::gateway