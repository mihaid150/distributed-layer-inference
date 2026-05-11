#pragma once

#include "dli/gateway/config.hpp"

#include <atomic>

namespace dli::gateway {

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