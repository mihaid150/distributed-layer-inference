#include "dli/gateway/config.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

std::string config_path_from_env_or_default() {
    const char* env = std::getenv("DLI_TEST_STAGE_MAP_PATH");
    if (env != nullptr && std::string(env).size() > 0) {
        return env;
    }

    return "configs/stage_map.yaml";
}

void test_gateway_config_loads_expected_fields(const std::string& config_path) {
    const dli::gateway::GatewayConfig config =
        dli::gateway::load_gateway_config_from_file(config_path);

    assert(config.config_path == config_path);
    assert(config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    assert(config.service_name == "inference-gateway");
    assert(config.port == 8000);
    assert(config.first_stage_url == "http://inference-stage-1:8000/forward");
}

void test_gateway_config_override_merge(const std::string& config_path) {
    dli::gateway::GatewayConfig file_config =
        dli::gateway::load_gateway_config_from_file(config_path);

    dli::gateway::GatewayConfig override_config;
    override_config.config_path = config_path;
    override_config.service_name.clear();
    override_config.model_name.clear();
    override_config.port = 8010;
    override_config.first_stage_url = "http://127.0.0.1:8001/forward-binary";

    const dli::gateway::GatewayConfig merged =
        dli::gateway::merge_gateway_config(file_config, override_config);

    assert(merged.config_path == config_path);
    assert(merged.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    assert(merged.service_name == "inference-gateway");
    assert(merged.port == 8010);
    assert(merged.first_stage_url == "http://127.0.0.1:8001/forward-binary");
}

} // namespace

int main() {
    const std::string config_path = config_path_from_env_or_default();

    test_gateway_config_loads_expected_fields(config_path);
    test_gateway_config_override_merge(config_path);

    std::cout << "test_gateway_config: OK\n";
    return 0;
}