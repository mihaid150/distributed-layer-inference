#pragma once

#include "dli_stage/config.hpp"
#include "dli_stage/runtime.hpp"

#include <atomic>
#include <memory>

namespace dli_stage {

class HttpServer {
public:
    HttpServer(StageConfig config, std::unique_ptr<StageRuntime> runtime);

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    int run();

    void stop();

private:
    StageConfig config_;
    std::unique_ptr<StageRuntime> runtime_;
    std::atomic<bool> stop_requested_{false};
};

} // namespace dli_stage