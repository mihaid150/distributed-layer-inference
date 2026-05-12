#pragma once

#include "dli/gateway/config.hpp"
#include "dli/gateway/tokenizer.hpp"

#include <atomic>
#include <memory>

namespace dli::gateway {

class GatewayServer {
public:
    GatewayServer(
        GatewayConfig config,
        std::unique_ptr<Tokenizer> tokenizer
    );

    GatewayServer(const GatewayServer&) = delete;
    GatewayServer& operator=(const GatewayServer&) = delete;

    int run();

    void stop();

private:
    GatewayConfig config_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::atomic<bool> stop_requested_{false};
};

} // namespace dli::gateway