#include "dli/common/http.hpp"

#include "dli/common/json_escape.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <sys/socket.h>

namespace dli::common {

namespace {

constexpr std::size_t MAX_HTTP_HEADER_BYTES = 64u * 1024u;
constexpr std::size_t MAX_HTTP_BODY_BYTES = 512u * 1024u * 1024u;

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

std::vector<std::uint8_t> string_to_bytes(const std::string& text) {
    const auto* begin = reinterpret_cast<const std::uint8_t*>(text.data());
    return std::vector<std::uint8_t>(begin, begin + text.size());
}

} // namespace

HttpRequest read_http_request(int fd) {
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



bool request_wants_keep_alive(const HttpRequest& request) {
    const auto it = request.headers.find("connection");
    if (it == request.headers.end()) {
        return true;
    }

    const std::string value = lower_copy(trim_copy(it->second));
    return value != "close";
}

void send_http_response(int fd, const HttpResponse& response) {
    std::ostringstream header;
    header << "HTTP/1.1 " << response.status_code << " " << response.reason << "\r\n";
    header << "Content-Type: " << response.content_type << "\r\n";
    header << "Content-Length: " << response.body.size() << "\r\n";
    header << "Connection: "
           << (response.keep_alive ? "keep-alive" : "close")
           << "\r\n";
    header << "\r\n";

    const std::string header_text = header.str();

    (void)send_all(fd, header_text);

    if (!response.body.empty()) {
        (void)send_all(fd, response.body.data(), response.body.size());
    }
}

HttpResponse make_json_response(
    int status_code,
    const std::string& reason,
    const std::string& json
) {
    HttpResponse response;
    response.status_code = status_code;
    response.reason = reason;
    response.content_type = "application/json";
    response.body = string_to_bytes(json);
    return response;
}

HttpResponse make_text_response(
    int status_code,
    const std::string& reason,
    const std::string& text
) {
    HttpResponse response;
    response.status_code = status_code;
    response.reason = reason;
    response.content_type = "text/plain; charset=utf-8";
    response.body = string_to_bytes(text);
    return response;
}

HttpResponse make_binary_response(
    int status_code,
    const std::string& reason,
    const std::vector<std::uint8_t>& body
) {
    HttpResponse response;
    response.status_code = status_code;
    response.reason = reason;
    response.content_type = "application/octet-stream";
    response.body = body;
    return response;
}

std::string http_error_json(const std::string& error) {
    std::ostringstream out;
    out << "{\"ok\":false,\"error\":\"" << json_escape(error) << "\"}";
    return out.str();
}

} // namespace dli::common