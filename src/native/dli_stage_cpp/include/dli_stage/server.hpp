#pragma once

#include <atomic>
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
    explicit HttpServer(ServerConfig config);

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    int run();

    void stop();

private:
    ServerConfig config_;
    std::atomic<bool> stop_requested_{false};
};

} // namespace dli_stage