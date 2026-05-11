#include "dli/gateway/server.hpp"
#include "dli/gateway/stage_client.hpp"
#include "dli/common/http.hpp"
#include "dli/common/json_escape.hpp"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace dli::gateway {

namespace {

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
        << "\"first_stage_url\":\"" << dli::common::json_escape(config.first_stage_url) << "\","
        << "\"routes\":["
        << "\"GET /health\","
        << "\"GET /config\","
        << "\"POST /generate\""
        << "]"
        << "}";

    return out.str();
}

std::string generate_stub_json(
    const dli::common::HttpRequest& request,
    const StageClientResult& stage_result
) {
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":true,"
        << "\"service\":\"dli-gateway-cpp\","
        << "\"runtime\":\"cpp-native-stub\","
        << "\"status\":\"stub_generate_with_stage_call\","
        << "\"prompt\":\"\","
        << "\"generated_text\":\"\","
        << "\"assistant_message\":\"\","
        << "\"generated_token_ids\":[],"
        << "\"total_latency_ms\":0.0,"
        << "\"tokens_per_second\":0.0,"
        << "\"termination_reason\":\"stub\","
        << "\"request_body_bytes\":" << request.body.size() << ","
        << "\"stage_http_status\":" << stage_result.http_status << ","
        << "\"stage_http_reason\":\"" << dli::common::json_escape(stage_result.http_reason) << "\","
        << "\"stage_metadata\":\"" << dli::common::json_escape(stage_result.response_frame.metadata_json) << "\","
        << "\"stage_tensor_bytes\":" << stage_result.response_frame.tensor_bytes.size()
        << "}";

    return out.str();
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
    const GatewayConfig& config
) {
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
        StageClient client(config.first_stage_url);
        const StageClientResult result = client.forward_stub_frame();

        return dli::common::make_json_response(
            200,
            "OK",
            generate_stub_json(request, result)
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

GatewayServer::GatewayServer(GatewayConfig config)
    : config_(std::move(config)) {}

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
                handle_request(request, config_);

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