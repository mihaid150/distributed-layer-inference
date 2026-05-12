#include "dli_stage/config.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace dli_stage {

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

} // namespace

StageConfig load_stage_config_from_file(
    const std::string& config_path,
    int wanted_stage_id
) {
    StageConfig config;
    config.config_path = config_path;
    config.stage_id = wanted_stage_id;
    config.service_name.clear();

    std::ifstream input(config_path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open stage config file: " + config_path);
    }

    bool inside_inference_stages = false;
    bool inside_target_stage = false;
    bool inside_components = false;
    bool inside_layers = false;

    std::string line;
    while (std::getline(input, line)) {
        const std::string trimmed = trim_copy(line);

        if (trimmed.empty() || starts_with(trimmed, "#")) {
            continue;
        }

        const int indent = indent_width(line);

        if (!inside_inference_stages) {
            if (indent == 0 && starts_with(trimmed, "inference_stages:")) {
                inside_inference_stages = true;
            }
            continue;
        }

        // The stage list in your YAML uses top-level "- stage_id:" entries.
        // Only stop when another top-level section starts, e.g. "topology:".
        if (
            indent == 0 &&
            !is_stage_list_item(trimmed) &&
            !starts_with(trimmed, "inference_stages:")
        ) {
            break;
        }

        if (is_stage_list_item(trimmed)) {
            const int current_stage_id = parse_stage_id_from_list_item(trimmed);

            inside_target_stage = current_stage_id == wanted_stage_id;
            inside_components = false;
            inside_layers = false;

            if (inside_target_stage) {
                config.stage_id = current_stage_id;
                config.components.layers.clear();
            }

            continue;
        }

        if (!inside_target_stage) {
            continue;
        }

        if (starts_with(trimmed, "components:")) {
            inside_components = true;
            inside_layers = false;
            continue;
        }

        // Leave components when returning to a normal stage-level field.
        if (inside_components && indent <= 2 && !starts_with(trimmed, "components:")) {
            inside_components = false;
            inside_layers = false;
        }

        if (!inside_components) {
            if (starts_with(trimmed, "service_name:")) {
                config.service_name = value_after_colon(trimmed);
                continue;
            }

            if (starts_with(trimmed, "physical_node:")) {
                config.physical_node = value_after_colon(trimmed);
                continue;
            }

            if (starts_with(trimmed, "partition_file:")) {
                config.partition_file = value_after_colon(trimmed);
                continue;
            }

            if (starts_with(trimmed, "native_partition_file:")) {
                config.native_partition_file = value_after_colon(trimmed);
                continue;
            }

            if (starts_with(trimmed, "backend:")) {
                config.backend = value_after_colon(trimmed);
                continue;
            }

            if (starts_with(trimmed, "next_stage_url:")) {
                config.next_stage_url = value_after_colon(trimmed);
                continue;
            }

            continue;
        }

        if (inside_components) {
            if (starts_with(trimmed, "embedding:")) {
                config.components.embedding = parse_bool_value(value_after_colon(trimmed));
                continue;
            }

            if (starts_with(trimmed, "norm:")) {
                config.components.norm = parse_bool_value(value_after_colon(trimmed));
                continue;
            }

            if (starts_with(trimmed, "lm_head:")) {
                config.components.lm_head = parse_bool_value(value_after_colon(trimmed));
                continue;
            }

            if (starts_with(trimmed, "layers:")) {
                inside_layers = true;
                continue;
            }

            if (inside_layers && starts_with(trimmed, "-")) {
                std::string item = trim_copy(trimmed.substr(1));
                if (!item.empty()) {
                    config.components.layers.push_back(std::stoi(item));
                }
                continue;
            }
        }
    }

    if (config.service_name.empty()) {
        throw std::runtime_error(
            "stage_id=" + std::to_string(wanted_stage_id) +
            " not found or missing service_name in " + config_path
        );
    }

    return config;
}

StageConfig merge_stage_config(
    StageConfig file_config,
    const StageConfig& override_config
) {
    if (!override_config.config_path.empty()) {
        file_config.config_path = override_config.config_path;
    }

    if (override_config.stage_id > 0) {
        file_config.stage_id = override_config.stage_id;
    }

    if (override_config.port > 0) {
        file_config.port = override_config.port;
    }

    if (!override_config.service_name.empty()) {
        file_config.service_name = override_config.service_name;
    }

    if (!override_config.physical_node.empty()) {
        file_config.physical_node = override_config.physical_node;
    }

    if (!override_config.partition_file.empty()) {
        file_config.partition_file = override_config.partition_file;
    }

    if (!override_config.native_partition_file.empty()) {
        file_config.native_partition_file = override_config.native_partition_file;
    }

    if (!override_config.backend.empty()) {
        file_config.backend = override_config.backend;
    }

    if (!override_config.next_stage_url.empty()) {
        file_config.next_stage_url = override_config.next_stage_url;
    }

    return file_config;
}

} // namespace dli_stage