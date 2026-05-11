#pragma once

#include <atomic>
#include <string>

namespace dli::gateway {

struct GatewayConfig {
    std::string config_path = "/app/configs/stage_map.yaml";
    std::string service_name = "dli-gateway-cpp";
    std::string first_stage_url = "http://inference-stage-1:8000/forward-binary";
    int port = 8000;
};

class GatewayServer {
public:
    explicit GatewayServer(GatewayConfig config);

    GatewayServer(const GatewayServer&) = delete;
    GatewayServer& operator=(const GatewayServer&) = delete;

    int run();

    void stop();

private:
    GatewayConfig config_;
    std::atomic<bool> stop_requested_{false};
};

} // namespace dli::gateway