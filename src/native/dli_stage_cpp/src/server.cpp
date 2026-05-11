#include "dli_stage/server.hpp"

#include "dli/common/http.hpp"
#include "dli/common/json_escape.hpp"
#include "dli/common/metadata.hpp"
#include "dli/common/protocol.hpp"
#include "dli/common/tensor.hpp"
#include "dli_stage/runtime.hpp"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace dli_stage {

namespace {

std::string health_json(const StageConfig& config) {
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":true,"
        << "\"service\":\"" << dli::common::json_escape(config.service_name) << "\","
        << "\"runtime\":\"cpp-native-stub\","
        << "\"status\":\"stub\","
        << "\"stage_id\":" << config.stage_id << ","
        << "\"protocol\":\"DLI2\""
        << "}";

    return out.str();
}

std::string metrics_json(const dli::common::StageMetrics& metrics) {
    std::ostringstream out;
    out
        << "{"
        << "\"backend\":\"" << dli::common::json_escape(metrics.backend) << "\","
        << "\"status\":\"" << dli::common::json_escape(metrics.status) << "\","
        << "\"compute_time_ms\":" << metrics.compute_time_ms << ","
        << "\"true_comm_ms\":" << metrics.true_comm_ms << ","
        << "\"rpc_wall_time_ms\":" << metrics.rpc_wall_time_ms << ","
        << "\"input_tensor_bytes\":" << metrics.input_tensor_bytes << ","
        << "\"output_tensor_bytes\":" << metrics.output_tensor_bytes << ","
        << "\"stage_input_token_count\":" << metrics.stage_input_token_count << ","
        << "\"stage_output_token_count\":" << metrics.stage_output_token_count << ","
        << "\"kv_cache_step_valid\":" << (metrics.kv_cache_step_valid ? "true" : "false")
        << "}";

    return out.str();
}

std::string shape_json(const std::vector<std::int64_t>& shape) {
    std::ostringstream out;
    out << "[";

    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << shape[i];
    }

    out << "]";
    return out.str();
}

std::string runtime_response_metadata_json(
    const RuntimeResponse& response,
    const StageConfig& config,
    const RuntimeRequest& request
) {
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":true,"
        << "\"service\":\"dli-stage-cpp\","
        << "\"runtime\":\"cpp-native-stub\","
        << "\"backend\":\"" << dli::common::json_escape(response.metrics.backend) << "\","
        << "\"status\":\"" << dli::common::json_escape(response.metrics.status) << "\","
        << "\"stage_id\":" << config.stage_id << ","
        << "\"request_id\":\"" << dli::common::json_escape(request.request_id) << "\","
        << "\"token_index\":" << request.token_index << ","
        << "\"generation_mode\":\"" << dli::common::json_escape(request.generation_mode) << "\","
        << "\"input_metadata_bytes\":" << request.input_metadata_json.size() << ","
        << "\"input_tensor_bytes\":" << request.input_tensor.bytes.size() << ","
        << "\"output_tensor_bytes\":" << response.output_tensor.bytes.size() << ","
        << "\"is_final_stage\":" << (response.is_final_stage ? "true" : "false") << ","
        << "\"next_token_id\":" << response.next_token_id << ","
        << "\"tensor\":{"
        << "\"dtype\":\"" << dli::common::json_escape(request.input_tensor.metadata.dtype) << "\","
        << "\"shape\":" << shape_json(request.input_tensor.metadata.shape) << ","
        << "\"byte_order\":\"" << dli::common::json_escape(request.input_tensor.metadata.byte_order) << "\""
        << "},"
        << "\"feature_flags\":{"
        << "\"kv_cache_enabled\":" << (request.kv_cache_enabled ? "true" : "false")
        << "},"
        << "\"metrics\":" << metrics_json(response.metrics)
        << "}";

    return out.str();
}

std::string layers_json(const std::vector<int>& layers) {
    std::ostringstream out;
    out << "[";

    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << layers[i];
    }

    out << "]";
    return out.str();
}

std::string config_json(const StageConfig& config) {
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":true,"
        << "\"service\":\"" << dli::common::json_escape(config.service_name) << "\","
        << "\"runtime\":\"cpp-native-stub\","
        << "\"status\":\"stub\","
        << "\"stage_id\":" << config.stage_id << ","
        << "\"port\":" << config.port << ","
        << "\"config_path\":\"" << dli::common::json_escape(config.config_path) << "\","
        << "\"physical_node\":\"" << dli::common::json_escape(config.physical_node) << "\","
        << "\"partition_file\":\"" << dli::common::json_escape(config.partition_file) << "\","
        << "\"next_stage_url\":\"" << dli::common::json_escape(config.next_stage_url) << "\","
        << "\"components\":{"
        << "\"embedding\":" << (config.components.embedding ? "true" : "false") << ","
        << "\"layers\":" << layers_json(config.components.layers) << ","
        << "\"norm\":" << (config.components.norm ? "true" : "false") << ","
        << "\"lm_head\":" << (config.components.lm_head ? "true" : "false")
        << "},"
        << "\"routes\":["
        << "\"GET /health\","
        << "\"GET /config\","
        << "\"POST /forward-binary\""
        << "]"
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

RuntimeRequest make_runtime_request(
    const dli::common::DliFrame& input,
    const StageConfig& config
) {
    const dli::common::ParsedRequestMetadata parsed =
        dli::common::parse_request_metadata(input.metadata_json);

    RuntimeRequest request;
    request.stage_id = config.stage_id;
    request.input_metadata_json = input.metadata_json;
    request.input_tensor.bytes = input.tensor_bytes;

    request.request_id = parsed.request_id;
    request.token_index = parsed.token_index;
    request.generation_mode = parsed.generation_mode;
    request.input_tensor.metadata = parsed.tensor;

    request.kv_cache_enabled = parsed.kv_cache_enabled;

    request.has_temperature = parsed.has_temperature;
    request.temperature = parsed.temperature;

    request.has_top_k = parsed.has_top_k;
    request.top_k = parsed.top_k;

    request.has_top_p = parsed.has_top_p;
    request.top_p = parsed.top_p;

    return request;
}

dli::common::HttpResponse handle_forward_binary(
    const dli::common::HttpRequest& request,
    const StageConfig& config,
    StageRuntime& runtime
) {
    try {
        const dli::common::DliFrame input = dli::common::decode_frame(request.body);
        RuntimeRequest runtime_request = make_runtime_request(input, config);
        RuntimeResponse runtime_response = runtime.forward(runtime_request);

        dli::common::DliFrame output;
        output.metadata_json = runtime_response_metadata_json(
            runtime_response,
            config,
            runtime_request
        );
        output.tensor_bytes = runtime_response.output_tensor.bytes;

        return dli::common::make_binary_response(
            200,
            "OK",
            dli::common::encode_frame(output)
        );
    } catch (const std::exception& exc) {
        return dli::common::make_json_response(
            400,
            "Bad Request",
            dli::common::http_error_json(exc.what())
        );
    }
}

dli::common::HttpResponse handle_request(
    const dli::common::HttpRequest& request,
    const StageConfig& config,
    StageRuntime& runtime
) {
    if (request.method == "GET" && request.path == "/health") {
        return dli::common::make_json_response(200, "OK", health_json(config));
    }

    if (request.method == "GET" && request.path == "/config") {
        return dli::common::make_json_response(200, "OK", config_json(config));
    }

    if (request.method == "POST" && request.path == "/forward-binary") {
        return handle_forward_binary(request, config, runtime);
    }

    return dli::common::make_json_response(404, "Not Found", not_found_json(request));
}

void close_fd(int fd) {
    if (fd >= 0) {
        while (::close(fd) < 0 && errno == EINTR) {
        }
    }
}

} // namespace

HttpServer::HttpServer(StageConfig config, std::unique_ptr<StageRuntime> runtime)
    : config_(std::move(config)),
      runtime_(std::move(runtime)) {
    if (!runtime_) {
        throw std::runtime_error("HttpServer requires a non-null StageRuntime");
    }
}

void HttpServer::stop() {
    stop_requested_.store(true);
}

int HttpServer::run() {
    const int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "[dli-stage-cpp] socket failed: " << std::strerror(errno) << "\n";
        return 1;
    }

    int reuse = 1;
    if (::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        std::cerr << "[dli-stage-cpp] setsockopt(SO_REUSEADDR) failed: " << std::strerror(errno) << "\n";
        close_fd(server_fd);
        return 1;
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<std::uint16_t>(config_.port));

    if (::bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        std::cerr << "[dli-stage-cpp] bind failed on port " << config_.port << ": "
                  << std::strerror(errno) << "\n";
        close_fd(server_fd);
        return 1;
    }

    if (::listen(server_fd, 64) < 0) {
        std::cerr << "[dli-stage-cpp] listen failed: " << std::strerror(errno) << "\n";
        close_fd(server_fd);
        return 1;
    }

    std::cerr << "[dli-stage-cpp] listening on 0.0.0.0:" << config_.port << "\n";

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

            std::cerr << "[dli-stage-cpp] accept failed: " << std::strerror(errno) << "\n";
            continue;
        }

        try {
            const dli::common::HttpRequest request = dli::common::read_http_request(client_fd);
            const dli::common::HttpResponse response = handle_request(
                request,
                config_,
                *runtime_
            );
            dli::common::send_http_response(client_fd, response);
        } catch (const std::exception& exc) {
            const dli::common::HttpResponse response = dli::common::make_json_response(
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

} // namespace dli_stage