#include "dli_stage/runtimes/stub_runtime.hpp"
#include "dli_stage/server.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
    std::string config_path = "/app/configs/stage_map.yaml";
    int port = 8000;
    int stage_id = 0;
};

bool parse_int(const std::string& value, int& out) {
    try {
        std::size_t pos = 0;
        int parsed = std::stoi(value, &pos);
        if (pos != value.size()) {
            return false;
        }
        out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

void print_usage(const char* program_name) {
    std::cerr
        << "Usage: " << program_name << " [options]\n\n"
        << "Options:\n"
        << "  --config <path>      Path to stage_map.yaml. Default: /app/configs/stage_map.yaml\n"
        << "  --port <port>        HTTP port for the native stage server. Default: 8000\n"
        << "  --stage-id <id>      Stage id. If omitted, STAGE_ID env is used when available.\n"
        << "  --help               Show this help message.\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;

    const char* env_stage_id = std::getenv("STAGE_ID");
    if (env_stage_id != nullptr) {
        int parsed_stage_id = 0;
        if (parse_int(env_stage_id, parsed_stage_id)) {
            options.stage_id = parsed_stage_id;
        }
    }

    const char* env_port = std::getenv("PORT");
    if (env_port != nullptr) {
        int parsed_port = 0;
        if (parse_int(env_port, parsed_port) && parsed_port > 0 && parsed_port <= 65535) {
            options.port = parsed_port;
        }
    }

    const char* env_config_path = std::getenv("STAGE_MAP_PATH");
    if (env_config_path != nullptr && std::strlen(env_config_path) > 0) {
        options.config_path = env_config_path;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }

        if (arg == "--config") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--config requires a value");
            }
            options.config_path = argv[++i];
            continue;
        }

        if (arg == "--port") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--port requires a value");
            }

            int parsed_port = 0;
            if (!parse_int(argv[++i], parsed_port) || parsed_port <= 0 || parsed_port > 65535) {
                throw std::runtime_error("--port must be an integer in [1, 65535]");
            }

            options.port = parsed_port;
            continue;
        }

        if (arg == "--stage-id") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--stage-id requires a value");
            }

            int parsed_stage_id = 0;
            if (!parse_int(argv[++i], parsed_stage_id) || parsed_stage_id < 0) {
                throw std::runtime_error("--stage-id must be a non-negative integer");
            }

            options.stage_id = parsed_stage_id;
            continue;
        }

        throw std::runtime_error("unknown argument: " + arg);
    }

    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);

        dli_stage::ServerConfig config;
        config.config_path = options.config_path;
        config.port = options.port;
        config.stage_id = options.stage_id;
        config.runtime = "cpp-native-stub";

        std::cerr << "[dli-stage-cpp] native C++ stage runtime\n";
        std::cerr << "[dli-stage-cpp] config_path=" << config.config_path << "\n";
        std::cerr << "[dli-stage-cpp] port=" << config.port << "\n";
        std::cerr << "[dli-stage-cpp] stage_id=" << config.stage_id << "\n";
        std::cerr << "[dli-stage-cpp] runtime=" << config.runtime << "\n";

        auto runtime = std::make_unique<dli_stage::StubRuntime>();
        dli_stage::HttpServer server(config, std::move(runtime));

        return server.run();
    } catch (const std::exception& exc) {
        std::cerr << "[dli-stage-cpp] error: " << exc.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }
}