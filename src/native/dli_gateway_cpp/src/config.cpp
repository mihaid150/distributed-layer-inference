#include "dli/gateway/config.hpp"

#include <fstream>
#include <sstream>
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
    while (end > begin && (value[end - 1] == ' ' || value[end - 1] == '\t' || value[end - 1] == '\r')) {
        --end;
    }

    return value.substr(begin, end - begin);
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string value_after_colon(const std::string& line) {
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
        return "";
    }

    std::string value = trim_copy(line.substr(colon + 1));

    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
    }

    if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'') {
        value = value.substr(1, value.size() - 2);
    }

    return value;
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

} // namespace

GatewayConfig load_gateway_config_from_file(const std::string& config_path) {
    GatewayConfig config;
    config.config_path = config_path;

    std::ifstream input(config_path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open gateway config file: " + config_path);
    }

    bool inside_gateway = false;
    int gateway_indent = -1;

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
            gateway_indent = indent;
            continue;
        }

        if (inside_gateway && indent <= gateway_indent && !starts_with(trimmed, "inference_gateway:")) {
            inside_gateway = false;
        }

        if (!inside_gateway) {
            continue;
        }

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