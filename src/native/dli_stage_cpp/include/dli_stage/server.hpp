#pragma once

#include "dli_stage/runtime.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace dli_stage {

struct ServerConfig {
    std::string config_path = "/app/configs/stage_map.yaml";
    int port = 8000;
    int stage_id = 0;
    std::string runtime = "cpp-native-stub";
};

class HttpServer {
public:
    HttpServer(ServerConfig config, std::unique_ptr<StageRuntime> runtime);

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    int run();

    void stop();

private:
    ServerConfig config_;
    std::unique_ptr<StageRuntime> runtime_;
    std::atomic<bool> stop_requested_{false};
};

} // namespace dli_stage