#include "dli_stage/server.hpp"

#include "dli/common/json_escape.hpp"
#include "dli/common/metadata.hpp"
#include "dli/common/protocol.hpp"
#include "dli/common/tensor.hpp"
#include "dli_stage/runtime.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
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

constexpr std::size_t MAX_HTTP_HEADER_BYTES = 64u * 1024u;
constexpr std::size_t MAX_HTTP_BODY_BYTES = 512u * 1024u * 1024u;

struct HttpRequest {
    std::string method;
    std::string path;
    std::map<std::string, std::string> headers;
    std::vector<std::uint8_t> body;
};

std::string lower_copy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
    );
    return value;
}

std::string trim_copy(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }

    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return value.substr(begin, end - begin);
}

std::size_t find_header_end(const std::vector<std::uint8_t>& buffer) {
    if (buffer.size() < 4) {
        return std::string::npos;
    }

    for (std::size_t i = 0; i + 3 < buffer.size(); ++i) {
        if (buffer[i] == '\r' &&
            buffer[i + 1] == '\n' &&
            buffer[i + 2] == '\r' &&
            buffer[i + 3] == '\n') {
            return i;
        }
    }

    return std::string::npos;
}

bool send_all(int fd, const std::uint8_t* data, std::size_t size) {
    std::size_t sent = 0;

    while (sent < size) {
        const ssize_t rc = ::send(fd, data + sent, size - sent, 0);
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }

        if (rc == 0) {
            return false;
        }

        sent += static_cast<std::size_t>(rc);
    }

    return true;
}

bool send_all(int fd, const std::string& data) {
    return send_all(
        fd,
        reinterpret_cast<const std::uint8_t*>(data.data()),
        data.size()
    );
}

void send_response(
    int fd,
    int status_code,
    const std::string& reason,
    const std::string& content_type,
    const std::vector<std::uint8_t>& body
) {
    std::ostringstream header;
    header << "HTTP/1.1 " << status_code << " " << reason << "\r\n";
    header << "Content-Type: " << content_type << "\r\n";
    header << "Content-Length: " << body.size() << "\r\n";
    header << "Connection: close\r\n";
    header << "\r\n";

    const std::string header_text = header.str();
    (void)send_all(fd, header_text);

    if (!body.empty()) {
        (void)send_all(fd, body.data(), body.size());
    }
}

void send_json(int fd, int status_code, const std::string& reason, const std::string& json) {
    const auto* begin = reinterpret_cast<const std::uint8_t*>(json.data());
    std::vector<std::uint8_t> body(begin, begin + json.size());
    send_response(fd, status_code, reason, "application/json", body);
}

std::string http_error_json(const std::string& error) {
    std::ostringstream out;
    out << "{\"ok\":false,\"error\":\"" << dli::common::json_escape(error) << "\"}";
    return out.str();
}

HttpRequest parse_http_request(
    const std::vector<std::uint8_t>& header_bytes,
    const std::vector<std::uint8_t>& body
) {
    const std::string header_text(
        reinterpret_cast<const char*>(header_bytes.data()),
        header_bytes.size()
    );

    std::istringstream stream(header_text);

    std::string request_line;
    if (!std::getline(stream, request_line)) {
        throw std::runtime_error("missing HTTP request line");
    }

    if (!request_line.empty() && request_line.back() == '\r') {
        request_line.pop_back();
    }

    std::istringstream request_line_stream(request_line);

    HttpRequest request;
    std::string version;

    request_line_stream >> request.method >> request.path >> version;

    if (request.method.empty() || request.path.empty() || version.empty()) {
        throw std::runtime_error("malformed HTTP request line");
    }

    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.empty()) {
            break;
        }

        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }

        const std::string key = lower_copy(trim_copy(line.substr(0, colon)));
        const std::string value = trim_copy(line.substr(colon + 1));

        request.headers[key] = value;
    }

    request.body = body;
    return request;
}

std::size_t parse_content_length(const std::map<std::string, std::string>& headers) {
    auto it = headers.find("content-length");
    if (it == headers.end()) {
        return 0;
    }

    try {
        std::size_t parsed_chars = 0;
        const unsigned long long value = std::stoull(it->second, &parsed_chars);

        if (parsed_chars != it->second.size()) {
            throw std::runtime_error("invalid Content-Length");
        }

        if (value > MAX_HTTP_BODY_BYTES) {
            throw std::runtime_error("Content-Length exceeds maximum allowed size");
        }

        return static_cast<std::size_t>(value);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid Content-Length");
    }
}

HttpRequest read_request(int fd) {
    std::vector<std::uint8_t> buffer;
    buffer.reserve(8192);

    std::size_t header_end = std::string::npos;
    std::size_t expected_body_size = 0;

    while (true) {
        std::uint8_t temp[8192];
        const ssize_t rc = ::recv(fd, temp, sizeof(temp), 0);

        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("recv failed: ") + std::strerror(errno));
        }

        if (rc == 0) {
            throw std::runtime_error("client closed connection before full request was received");
        }

        buffer.insert(buffer.end(), temp, temp + rc);

        if (header_end == std::string::npos) {
            header_end = find_header_end(buffer);

            if (header_end == std::string::npos && buffer.size() > MAX_HTTP_HEADER_BYTES) {
                throw std::runtime_error("HTTP headers exceed maximum allowed size");
            }

            if (header_end != std::string::npos) {
                std::vector<std::uint8_t> header_bytes(buffer.begin(), buffer.begin() + header_end);
                std::vector<std::uint8_t> partial_body(
                    buffer.begin() + static_cast<std::ptrdiff_t>(header_end + 4),
                    buffer.end()
                );

                HttpRequest partial = parse_http_request(header_bytes, partial_body);
                expected_body_size = parse_content_length(partial.headers);

                if (partial.body.size() >= expected_body_size) {
                    partial.body.resize(expected_body_size);
                    return partial;
                }
            }
        } else {
            const std::size_t body_start = header_end + 4;
            const std::size_t current_body_size = buffer.size() - body_start;

            if (current_body_size >= expected_body_size) {
                std::vector<std::uint8_t> header_bytes(buffer.begin(), buffer.begin() + header_end);
                std::vector<std::uint8_t> body(
                    buffer.begin() + static_cast<std::ptrdiff_t>(body_start),
                    buffer.begin() + static_cast<std::ptrdiff_t>(body_start + expected_body_size)
                );

                return parse_http_request(header_bytes, body);
            }
        }
    }
}

std::string health_json(const ServerConfig& config) {
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":true,"
        << "\"service\":\"dli-stage-cpp\","
        << "\"runtime\":\"" << dli::common::json_escape(config.runtime) << "\","
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
    const ServerConfig& config,
    const RuntimeRequest& request
) {
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":true,"
        << "\"service\":\"dli-stage-cpp\","
        << "\"runtime\":\"" << dli::common::json_escape(config.runtime) << "\","
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

std::string config_json(const ServerConfig& config) {
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":true,"
        << "\"service\":\"dli-stage-cpp\","
        << "\"runtime\":\"" << dli::common::json_escape(config.runtime) << "\","
        << "\"status\":\"stub\","
        << "\"stage_id\":" << config.stage_id << ","
        << "\"port\":" << config.port << ","
        << "\"config_path\":\"" << dli::common::json_escape(config.config_path) << "\","
        << "\"routes\":["
        << "\"GET /health\","
        << "\"GET /config\","
        << "\"POST /forward-binary\""
        << "]"
        << "}";

    return out.str();
}

std::vector<std::uint8_t> frame_response_body(const dli::common::DliFrame& frame) {
    return dli::common::encode_frame(frame);
}

std::string not_found_json(const HttpRequest& request) {
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
    const ServerConfig& config
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

void handle_forward_binary(
    int fd,
    const HttpRequest& request,
    const ServerConfig& config,
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

        send_response(
            fd,
            200,
            "OK",
            "application/octet-stream",
            frame_response_body(output)
        );
    } catch (const std::exception& exc) {
        send_json(fd, 400, "Bad Request", http_error_json(exc.what()));
    }
}

void handle_request(
    int fd,
    const HttpRequest& request,
    const ServerConfig& config,
    StageRuntime& runtime
) {
    if (request.method == "GET" && request.path == "/health") {
        send_json(fd, 200, "OK", health_json(config));
        return;
    }

    if (request.method == "GET" && request.path == "/config") {
        send_json(fd, 200, "OK", config_json(config));
        return;
    }

    if (request.method == "POST" && request.path == "/forward-binary") {
        handle_forward_binary(fd, request, config, runtime);
        return;
    }

    send_json(fd, 404, "Not Found", not_found_json(request));
}

void close_fd(int fd) {
    if (fd >= 0) {
        while (::close(fd) < 0 && errno == EINTR) {
        }
    }
}

} // namespace

HttpServer::HttpServer(ServerConfig config, std::unique_ptr<StageRuntime> runtime)
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
            const HttpRequest request = read_request(client_fd);
            handle_request(client_fd, request, config_, *runtime_);
        } catch (const std::exception& exc) {
            send_json(client_fd, 400, "Bad Request", http_error_json(exc.what()));
        }

        close_fd(client_fd);
    }

    close_fd(server_fd);
    return 0;
}

} // namespace dli_stage