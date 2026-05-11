#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

namespace {

struct CliOptions {
    std::string config_path = "/app/configs/stage_map.yaml";
    int port = 8000;
    bool keep_alive = false;
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
        << "  --port <port>        HTTP port planned for the native stage server. Default: 8000\n"
        << "  --keep-alive         Keep the stub process alive for container smoke tests.\n"
        << "  --help               Show this help message.\n\n"
        << "Environment:\n"
        << "  DLI_CPP_STUB_KEEP_ALIVE=1   Keep process alive even without --keep-alive.\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;

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

        if (arg == "--keep-alive") {
            options.keep_alive = true;
            continue;
        }

        throw std::runtime_error("unknown argument: " + arg);
    }

    const char* env_keep_alive = std::getenv("DLI_CPP_STUB_KEEP_ALIVE");
    if (env_keep_alive != nullptr && std::strcmp(env_keep_alive, "1") == 0) {
        options.keep_alive = true;
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);

        std::cerr << "[dli-stage-cpp] native C++ stage runtime stub\n";
        std::cerr << "[dli-stage-cpp] config_path=" << options.config_path << "\n";
        std::cerr << "[dli-stage-cpp] port=" << options.port << "\n";
        std::cerr << "[dli-stage-cpp] status=not_implemented\n";
        std::cerr << "[dli-stage-cpp] next step: implement /health, /config, /forward-binary\n";

        if (options.keep_alive) {
            std::cerr << "[dli-stage-cpp] keep-alive mode enabled\n";
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(60));
            }
        }

        return 64;
    } catch (const std::exception& exc) {
        std::cerr << "[dli-stage-cpp] error: " << exc.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }
}