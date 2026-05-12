#include "llama.h"

#include "dli_stage/config.hpp"
#include "dli_stage/runtimes/llama_partial_runtime.hpp"
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
    std::string backend = "stub";
    std::string model_path;
    int port = 0;
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
        << "  --port <port>        HTTP port for the native stage server.\n"
        << "  --stage-id <id>      Stage id. If omitted, STAGE_ID env is used.\n"
        << "  --backend <name>     Runtime backend: stub | llama. Default: stub.\n"
        << "  --model <path>       Optional GGUF model/stage-shard path for llama backend.\n"
        << "  --help               Show this help message.\n\n"
        << "Environment:\n"
        << "  STAGE_ID             Stage id.\n"
        << "  PORT                 HTTP port.\n"
        << "  STAGE_MAP_PATH       Path to stage_map.yaml.\n"
        << "  DLI_STAGE_MODEL_PATH Optional GGUF model/stage-shard path for llama backend.\n"
        << "  DLI_STAGE_BACKEND    Runtime backend: stub | llama.\n";
}

std::unique_ptr<dli_stage::StageRuntime> make_runtime(const CliOptions& options) {
    if (options.backend == "stub") {
        return std::make_unique<dli_stage::StubRuntime>();
    }

    if (options.backend == "llama" || options.backend == "llama-partial") {
        dli_stage::LlamaPartialRuntimeConfig config;
        config.model_path = options.model_path;
        config.stage_id = options.stage_id;

        return std::make_unique<dli_stage::LlamaPartialRuntime>(std::move(config));
    }

    throw std::runtime_error(
        "unknown stage backend '" + options.backend + "'; expected stub or llama"
    );
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

    const char* env_backend = std::getenv("DLI_STAGE_BACKEND");
    if (env_backend != nullptr && std::strlen(env_backend) > 0) {
        options.backend = env_backend;
    }

    const char* env_model_path = std::getenv("DLI_STAGE_MODEL_PATH");
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

        if (arg == "--stage-id") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--stage-id requires a value");
            }

            int parsed_stage_id = 0;
            if (!parse_int(argv[++i], parsed_stage_id) || parsed_stage_id <= 0) {
                throw std::runtime_error("--stage-id must be a positive integer");
            }

            options.stage_id = parsed_stage_id;
            continue;
        }

        if (arg == "--backend") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--backend requires a value");
            }

            options.backend = argv[++i];
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

    if (options.stage_id <= 0) {
        throw std::runtime_error("stage id is required; pass --stage-id or set STAGE_ID");
    }

    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);

        dli_stage::StageConfig file_config =
            dli_stage::load_stage_config_from_file(options.config_path, options.stage_id);

        dli_stage::StageConfig override_config;
        override_config.service_name.clear();
        override_config.config_path = options.config_path;
        override_config.stage_id = options.stage_id;
        override_config.port = options.port;

        dli_stage::StageConfig config =
            dli_stage::merge_stage_config(file_config, override_config);
        
        llama_backend_init();

        CliOptions runtime_options = options;

        if (runtime_options.backend == "stub" && !config.backend.empty()) {
            runtime_options.backend = config.backend;
        }

        if (runtime_options.model_path.empty()) {
            if (
                (runtime_options.backend == "llama" ||
                runtime_options.backend == "llama-partial") &&
                !config.native_partition_file.empty()
            ) {
                runtime_options.model_path = config.native_partition_file;
            } else if (!config.partition_file.empty()) {
                runtime_options.model_path = config.partition_file;
            }
        }

        auto runtime = make_runtime(runtime_options);

        std::cerr << "[dli-stage-cpp] native C++ stage runtime\n";
        std::cerr << "[dli-stage-cpp] config_path=" << config.config_path << "\n";
        std::cerr << "[dli-stage-cpp] service_name=" << config.service_name << "\n";
        std::cerr << "[dli-stage-cpp] physical_node=" << config.physical_node << "\n";
        std::cerr << "[dli-stage-cpp] partition_file=" << config.partition_file << "\n";
        std::cerr << "[dli-stage-cpp] native_partition_file=" << config.native_partition_file << "\n";
        std::cerr << "[dli-stage-cpp] next_stage_url=" << config.next_stage_url << "\n";
        std::cerr << "[dli-stage-cpp] port=" << config.port << "\n";
        std::cerr << "[dli-stage-cpp] stage_id=" << config.stage_id << "\n";
        std::cerr << "[dli-stage-cpp] config_backend=" << config.backend << "\n";
        std::cerr << "[dli-stage-cpp] backend=" << runtime_options.backend << "\n";
        std::cerr << "[dli-stage-cpp] model_path=" << runtime_options.model_path << "\n";
        std::cerr << "[dli-stage-cpp] runtime_backend=" << runtime->backend_name() << "\n";

        dli_stage::HttpServer server(config, std::move(runtime));

        const int exit_code = server.run();

        llama_backend_free();

        return exit_code;
    } catch (const std::exception& exc) {
        std::cerr << "[dli-stage-cpp] error: " << exc.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }
}