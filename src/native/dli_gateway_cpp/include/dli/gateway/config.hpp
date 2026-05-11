#pragma once

#include <string>

namespace dli::gateway {

struct GatewayConfig {
    std::string config_path = "/app/configs/stage_map.yaml";
    std::string service_name = "dli-gateway-cpp";
    std::string model_name = "";
    std::string first_stage_url = "http://inference-stage-1:8000/forward-binary";
    int port = 8000;
};

GatewayConfig load_gateway_config_from_file(const std::string& config_path);

GatewayConfig merge_gateway_config(
    GatewayConfig file_config,
    const GatewayConfig& override_config
);

} // namespace dli::gateway