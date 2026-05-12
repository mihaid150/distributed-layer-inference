#include "dli/gateway/server.hpp"
#include "dli/gateway/generation_loop.hpp"
#include "dli/gateway/tokenizer.hpp"
#include "dli/common/http.hpp"
#include "dli/common/json_escape.hpp"
#include "dli/common/gguf_inspector.hpp"
#include "dli/common/partition_plan.hpp"
#include "dli/common/partition_tensor_assignment.hpp"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <regex>
#include <optional>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace dli::gateway {

namespace {

int parse_int_field_or_default(
    const std::string& json,
    const std::string& field_name,
    int default_value,
    int min_value,
    int max_value
) {
    const std::regex pattern(
        "\"" + field_name + R"dli("\s*:\s*(-?\d+))dli"
    );

    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return default_value;
    }

    try {
        const int parsed = std::stoi(match[1].str());
        return std::max(min_value, std::min(max_value, parsed));
    } catch (const std::exception&) {
        return default_value;
    }
}

std::optional<std::string> parse_string_field(
    const std::string& json,
    const std::string& field_name
) {
    const std::regex pattern(
        "\"" + field_name + R"dli("\s*:\s*"((?:\\.|[^"\\])*)")dli"
    );

    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return std::nullopt;
    }

    return match[1].str();
}

std::string health_json(const GatewayConfig& config) {
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":true,"
        << "\"service\":\"" << dli::common::json_escape(config.service_name) << "\","
        << "\"runtime\":\"cpp-native-stub\","
        << "\"status\":\"stub\","
        << "\"protocol\":\"http-json+dli2\""
        << "}";

    return out.str();
}

std::string int_vector_json(const std::vector<int>& values) {
    std::ostringstream out;
    out << "[";

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }

    out << "]";
    return out.str();
}

dli::common::PartitionComponentsPlan to_common_components(
    const PartitionComponents& components
) {
    dli::common::PartitionComponentsPlan common;
    common.embedding = components.embedding;
    common.layers = components.layers;
    common.norm = components.norm;
    common.lm_head = components.lm_head;
    return common;
}

std::string string_vector_json(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << "[";

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << "\"" << dli::common::json_escape(values[i]) << "\"";
    }

    out << "]";
    return out.str();
}

std::string partition_tensor_assignment_json(const GatewayConfig& config) {
    if (config.model_path.empty()) {
        return "[]";
    }

    try {
        const dli::common::GgufInspection inspection =
            dli::common::inspect_gguf_tensors(config.model_path);

        std::ostringstream out;
        out << "[";

        for (std::size_t i = 0; i < config.partitions.size(); ++i) {
            if (i > 0) {
                out << ",";
            }

            const auto& partition = config.partitions[i];

            const auto plan = dli::common::build_llama_tensor_name_plan(
                partition.partition_id,
                to_common_components(partition.components)
            );

            const auto assignment = dli::common::assign_tensors_to_partition(
                inspection,
                plan
            );

            out
                << "{"
                << "\"partition_id\":\"" << dli::common::json_escape(assignment.partition_id) << "\","
                << "\"tensor_count\":" << assignment.tensor_names.size() << ","
                << "\"missing_required_names\":" << string_vector_json(assignment.missing_required_names) << ","
                << "\"missing_required_prefixes\":" << string_vector_json(assignment.missing_required_prefixes)
                << "}";
        }

        out << "]";
        return out.str();
    } catch (const std::exception& exc) {
        std::ostringstream out;
        out
            << "[{"
            << "\"error\":\"" << dli::common::json_escape(exc.what()) << "\""
            << "}]";
        return out.str();
    }
}

std::string partition_graph_json(const std::vector<PartitionNodeConfig>& partitions) {
    std::ostringstream out;
    out << "[";

    for (std::size_t i = 0; i < partitions.size(); ++i) {
        if (i > 0) {
            out << ",";
        }

        const PartitionNodeConfig& partition = partitions[i];

        out
            << "{"
            << "\"stage_id\":" << partition.stage_id << ","
            << "\"partition_id\":\"" << dli::common::json_escape(partition.partition_id) << "\","
            << "\"service_name\":\"" << dli::common::json_escape(partition.service_name) << "\","
            << "\"physical_node\":\"" << dli::common::json_escape(partition.physical_node) << "\","
            << "\"partition_file\":\"" << dli::common::json_escape(partition.partition_file) << "\","
            << "\"next_stage_url\":\"" << dli::common::json_escape(partition.next_stage_url) << "\","
            << "\"components\":{"
            << "\"embedding\":" << (partition.components.embedding ? "true" : "false") << ","
            << "\"layers\":" << int_vector_json(partition.components.layers) << ","
            << "\"norm\":" << (partition.components.norm ? "true" : "false") << ","
            << "\"lm_head\":" << (partition.components.lm_head ? "true" : "false")
            << "}"
            << "}";
    }

    out << "]";
    return out.str();
}

std::string config_json(const GatewayConfig& config) {
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":true,"
        << "\"service\":\"" << dli::common::json_escape(config.service_name) << "\","
        << "\"runtime\":\"cpp-native-stub\","
        << "\"status\":\"stub\","
        << "\"port\":" << config.port << ","
        << "\"config_path\":\"" << dli::common::json_escape(config.config_path) << "\","
        << "\"num_layers\":" << config.num_layers << ","
        << "\"model_name\":\"" << dli::common::json_escape(config.model_name) << "\","
        << "\"model_path\":\"" << dli::common::json_escape(config.model_path) << "\","
        << "\"first_stage_url\":\"" << dli::common::json_escape(config.first_stage_url) << "\","
        << "\"partition_validation\":{"
        << "\"valid\":" << (config.partition_validation.valid ? "true" : "false") << ","
        << "\"error\":\"" << dli::common::json_escape(config.partition_validation.error) << "\","
        << "\"assigned_layer_count\":" << config.partition_validation.assigned_layer_count << ","
        << "\"embedding_owner_count\":" << config.partition_validation.embedding_owner_count << ","
        << "\"lm_head_owner_count\":" << config.partition_validation.lm_head_owner_count << ","
        << "\"terminal_partition_id\":\"" << dli::common::json_escape(config.partition_validation.terminal_partition_id) << "\""
        << "},"
        << "\"partition_count\":" << config.partitions.size() << ","
        << "\"partition_graph\":" << partition_graph_json(config.partitions) << ","
        << "\"partition_tensor_assignment\":"
        << partition_tensor_assignment_json(config) << ","
        << "\"routes\":["
        << "\"GET /health\","
        << "\"GET /config\","
        << "\"POST /generate\""
        << "]"
        << "}";

    return out.str();
}

std::string generate_loop_json(
    const dli::common::HttpRequest& request,
    const GatewayConfig& config,
    const Tokenizer& tokenizer
) {
    const std::string body_text(
        reinterpret_cast<const char*>(request.body.data()),
        request.body.size()
    );

    const int max_new_tokens = parse_int_field_or_default(
        body_text,
        "max_new_tokens",
        2,
        0,
        512
    );

    std::string prompt;
    if (const auto parsed_prompt = parse_string_field(body_text, "prompt")) {
        prompt = *parsed_prompt;
    }

    GenerationLoopConfig loop_config;
    loop_config.first_stage_url = config.first_stage_url;
    loop_config.model_path = config.model_path;
    loop_config.max_new_tokens = max_new_tokens;

    GenerationLoop loop(loop_config, tokenizer);

    const GenerationLoopResult result =
        loop.run_stub_generation(prompt, loop_config.max_new_tokens);

    return generation_loop_result_json(result, request.body.size());
}

std::string not_found_json(const dli::common::HttpRequest& request) {
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":false,"
        << "\"error\":\"route not found\","
        << "\"method\":\"" << dli::common::json_escape(request.method) << "\","
        << "\"path\":\"" << dli::common::json_escape(request.path) << "\""
        << "}";

    return out.str();
}

dli::common::HttpResponse handle_request(
    const dli::common::HttpRequest& request,
    const GatewayConfig& config,
    const Tokenizer& tokenizer
){
    if (request.method == "GET" && request.path == "/health") {
        return dli::common::make_json_response(
            200,
            "OK",
            health_json(config)
        );
    }

    if (request.method == "GET" && request.path == "/config") {
        return dli::common::make_json_response(
            200,
            "OK",
            config_json(config)
        );
    }

    if (request.method == "POST" && request.path == "/generate") {
        try {
            return dli::common::make_json_response(
                200,
                "OK",
                generate_loop_json(request, config, tokenizer)
            );
        } catch (const std::exception& exc) {
            return dli::common::make_json_response(
                502,
                "Bad Gateway",
                dli::common::http_error_json(exc.what())
            );
        }
    }

    return dli::common::make_json_response(
        404,
        "Not Found",
        not_found_json(request)
    );
}

void close_fd(int fd) {
    if (fd >= 0) {
        while (::close(fd) < 0 && errno == EINTR) {
        }
    }
}

} // namespace

GatewayServer::GatewayServer(
    GatewayConfig config,
    std::unique_ptr<Tokenizer> tokenizer
)
    : config_(std::move(config)),
      tokenizer_(std::move(tokenizer)) {
    if (!tokenizer_) {
        throw std::runtime_error("GatewayServer requires a tokenizer");
    }
}

void GatewayServer::stop() {
    stop_requested_.store(true);
}

int GatewayServer::run() {
    const int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "[dli-gateway-cpp] socket failed: " << std::strerror(errno) << "\n";
        return 1;
    }

    int reuse = 1;
    if (::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        std::cerr << "[dli-gateway-cpp] setsockopt(SO_REUSEADDR) failed: "
                  << std::strerror(errno) << "\n";
        close_fd(server_fd);
        return 1;
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<std::uint16_t>(config_.port));

    if (::bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        std::cerr << "[dli-gateway-cpp] bind failed on port " << config_.port << ": "
                  << std::strerror(errno) << "\n";
        close_fd(server_fd);
        return 1;
    }

    if (::listen(server_fd, 64) < 0) {
        std::cerr << "[dli-gateway-cpp] listen failed: " << std::strerror(errno) << "\n";
        close_fd(server_fd);
        return 1;
    }

    std::cerr << "[dli-gateway-cpp] listening on 0.0.0.0:" << config_.port << "\n";

    while (!stop_requested_.load()) {
        sockaddr_in client_address{};
        socklen_t client_len = sizeof(client_address);

        const int client_fd = ::accept(
            server_fd,
            reinterpret_cast<sockaddr*>(&client_address),
            &client_len
        );

        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }

            std::cerr << "[dli-gateway-cpp] accept failed: " << std::strerror(errno) << "\n";
            continue;
        }

        try {
            const dli::common::HttpRequest request =
                dli::common::read_http_request(client_fd);

            const dli::common::HttpResponse response = 
                handle_request(request, config_, *tokenizer_);

            dli::common::send_http_response(client_fd, response);
        } catch (const std::exception& exc) {
            const dli::common::HttpResponse response =
                dli::common::make_json_response(
                    400,
                    "Bad Request",
                    dli::common::http_error_json(exc.what())
                );

            dli::common::send_http_response(client_fd, response);
        }

        close_fd(client_fd);
    }

    close_fd(server_fd);
    return 0;
}

} // namespace dli::gateway