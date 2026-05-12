#include "llama.h"

#include "dli/gateway/config.hpp"
#include "dli/gateway/server.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
    std::string config_path = "/app/configs/stage_map.yaml";
    std::string first_stage_url;
    std::string service_name;
    std::string model_path;
    int port = 0;
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
        << "  --config <path>             Path to stage_map.yaml. Default: /app/configs/stage_map.yaml\n"
        << "  --port <port>               HTTP port for the native gateway.\n"
        << "  --first-stage-url <url>     Override first stage URL.\n"
        << "  --service-name <name>       Override service name.\n"
        << "  --model <path>              Override GGUF model path for future llama tokenizer/model loading.\n"
        << "  --help                      Show this help message.\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;

    const char* env_config_path = std::getenv("STAGE_MAP_PATH");
    if (env_config_path != nullptr && std::strlen(env_config_path) > 0) {
        options.config_path = env_config_path;
    }

    const char* env_port = std::getenv("PORT");
    if (env_port != nullptr) {
        int parsed_port = 0;
        if (parse_int(env_port, parsed_port) && parsed_port > 0 && parsed_port <= 65535) {
            options.port = parsed_port;
        }
    }

    const char* env_first_stage_url = std::getenv("FIRST_STAGE_URL");
    if (env_first_stage_url != nullptr && std::strlen(env_first_stage_url) > 0) {
        options.first_stage_url = env_first_stage_url;
    }

    const char* env_service_name = std::getenv("SERVICE_NAME");
    if (env_service_name != nullptr && std::strlen(env_service_name) > 0) {
        options.service_name = env_service_name;
    }

    const char* env_model_path = std::getenv("MODEL_PATH");
    if (env_model_path != nullptr && std::strlen(env_model_path) > 0) {
        options.model_path = env_model_path;
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

        if (arg == "--first-stage-url") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--first-stage-url requires a value");
            }
            options.first_stage_url = argv[++i];
            continue;
        }

        if (arg == "--service-name") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--service-name requires a value");
            }
            options.service_name = argv[++i];
            continue;
        }

        if (arg == "--model") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--model requires a value");
            }
            options.model_path = argv[++i];
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

        dli::gateway::GatewayConfig file_config =
            dli::gateway::load_gateway_config_from_file(options.config_path);

        dli::gateway::GatewayConfig override_config;
        override_config.config_path = options.config_path;
        override_config.port = options.port;
        override_config.first_stage_url = options.first_stage_url;
        override_config.service_name = options.service_name;
        override_config.model_path = options.model_path;

        dli::gateway::GatewayConfig config =
            dli::gateway::merge_gateway_config(file_config, override_config);

        std::cerr << "[dli-gateway-cpp] native C++ gateway runtime\n";
        std::cerr << "[dli-gateway-cpp] config_path=" << config.config_path << "\n";
        std::cerr << "[dli-gateway-cpp] service_name=" << config.service_name << "\n";
        std::cerr << "[dli-gateway-cpp] model_name=" << config.model_name << "\n";
        std::cerr << "[dli-gateway-cpp] model_path=" << config.model_path << "\n";
        std::cerr << "[dli-gateway-cpp] port=" << config.port << "\n";
        std::cerr << "[dli-gateway-cpp] first_stage_url=" << config.first_stage_url << "\n";
        std::cerr << "[dli-gateway-cpp] runtime=cpp-native-stub\n";

        llama_backend_init();

        dli::gateway::GatewayServer server(config);
        const int exit_code = server.run();

        llama_backend_free();

        return exit_code;
    } catch (const std::exception& exc) {
        std::cerr << "[dli-gateway-cpp] error: " << exc.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }
}