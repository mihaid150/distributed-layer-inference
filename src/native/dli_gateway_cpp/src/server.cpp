#include "dli/gateway/server.hpp"
#include "dli/gateway/generation_loop.hpp"
#include "dli/gateway/tokenizer.hpp"
#include "dli/common/http.hpp"
#include "dli/common/json_escape.hpp"
#include "dli/common/gguf_inspector.hpp"
#include "dli/common/partition_plan.hpp"
#include "dli/common/partition_tensor_assignment.hpp"

#include <cerrno>
#include <thread>
#include <mutex>
#include <functional>
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
            << "\"native_partition_file\":\"" << dli::common::json_escape(partition.native_partition_file) << "\","
            << "\"backend\":\"" << dli::common::json_escape(partition.backend) << "\","
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

bool parse_bool_field_or_default(
    const std::string& json,
    const std::string& field_name,
    bool default_value
) {
    const std::regex pattern(
        "\"" + field_name + R"dli("\s*:\s*(true|false))dli"
    );

    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return default_value;
    }

    return match[1].str() == "true";
}


std::string parse_string_field_or_default(
    const std::string& json,
    const std::string& field_name,
    const std::string& default_value
) {
    if (const auto value = parse_string_field(json, field_name)) {
        return *value;
    }
    return default_value;
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

    const int min_new_tokens = std::min(
        max_new_tokens,
        parse_int_field_or_default(body_text, "min_new_tokens", 0, 0, 512)
    );

    const bool apply_chat_template =
        parse_bool_field_or_default(body_text, "apply_chat_template", true);

    const std::string feature_activation_precision =
        parse_string_field_or_default(body_text, "activation_precision", "fp32");

    const bool feature_persistent_sessions_enabled =
        parse_bool_field_or_default(body_text, "persistent_sessions_enabled", false);

    const bool native_stage_chaining_enabled =
        parse_bool_field_or_default(body_text, "native_stage_chaining_enabled", false);

    const std::string metadata_level =
        parse_string_field_or_default(body_text, "metadata_level", "debug");

    std::string prompt;
    if (const auto parsed_prompt = parse_string_field(body_text, "prompt")) {
        prompt = *parsed_prompt;
    }

    GenerationLoopConfig loop_config;
    loop_config.first_stage_url = config.first_stage_url;
    loop_config.model_path = config.model_path;
    loop_config.max_new_tokens = max_new_tokens;
    loop_config.min_new_tokens = min_new_tokens;
    loop_config.apply_chat_template = apply_chat_template;
    loop_config.partitions = config.partitions;
    loop_config.activation_precision = feature_activation_precision;
    loop_config.persistent_sessions_enabled = feature_persistent_sessions_enabled;
    loop_config.native_stage_chaining_enabled = native_stage_chaining_enabled;
    loop_config.metadata_level = metadata_level;

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
    const Tokenizer& tokenizer,
    std::mutex& generation_mutex
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
            std::lock_guard<std::mutex> lock(generation_mutex);
            const std::string response_json =
                generate_loop_json(request, config, tokenizer);

            if (response_json.find("\"ok\":false") != std::string::npos) {
                return dli::common::make_json_response(
                    502,
                    "Bad Gateway",
                    response_json
                );
            }

            return dli::common::make_json_response(
                200,
                "OK",
                response_json
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

void handle_gateway_client_connection(
    int client_fd,
    const GatewayConfig& config,
    const Tokenizer& tokenizer,
    std::mutex& generation_mutex,
    std::atomic<bool>& stop_requested
) {
    while (!stop_requested.load()) {
        try {
            const dli::common::HttpRequest request =
                dli::common::read_http_request(client_fd);

            dli::common::HttpResponse response =
                handle_request(request, config, tokenizer, generation_mutex);
            response.keep_alive = dli::common::request_wants_keep_alive(request);
            dli::common::send_http_response(client_fd, response);
            if (!response.keep_alive) {
                break;
            }
        } catch (const std::exception& exc) {
            dli::common::HttpResponse response =
                dli::common::make_json_response(
                    400,
                    "Bad Request",
                    dli::common::http_error_json(exc.what())
                );
            response.keep_alive = false;
            dli::common::send_http_response(client_fd, response);
            break;
        } catch (...) {
            dli::common::HttpResponse response =
                dli::common::make_json_response(
                    400,
                    "Bad Request",
                    dli::common::http_error_json("unknown gateway connection failure")
                );
            response.keep_alive = false;
            dli::common::send_http_response(client_fd, response);
            break;
        }
    }

    close_fd(client_fd);
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

    std::mutex generation_mutex;

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

        std::thread(
            handle_gateway_client_connection,
            client_fd,
            std::cref(config_),
            std::cref(*tokenizer_),
            std::ref(generation_mutex),
            std::ref(stop_requested_)
        ).detach();
    }

    close_fd(server_fd);
    return 0;
}

} // namespace dli::gateway