#include "dli/common/http_client.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace dli::common {

namespace {

constexpr std::size_t MAX_HTTP_RESPONSE_HEADER_BYTES = 64u * 1024u;
constexpr std::size_t MAX_HTTP_RESPONSE_BODY_BYTES = 512u * 1024u * 1024u;

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

void close_fd(int fd) {
    if (fd >= 0) {
        while (::close(fd) < 0 && errno == EINTR) {
        }
    }
}

int connect_tcp(const ParsedUrl& url, int timeout_seconds) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* result = nullptr;
    const std::string port_str = std::to_string(url.port);

    const int gai_rc = ::getaddrinfo(url.host.c_str(), port_str.c_str(), &hints, &result);
    if (gai_rc != 0) {
        throw std::runtime_error(
            "getaddrinfo failed for host " + url.host + ": " + gai_strerror(gai_rc)
        );
    }

    int connected_fd = -1;

    for (addrinfo* item = result; item != nullptr; item = item->ai_next) {
        const int fd = ::socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (fd < 0) {
            continue;
        }

        timeval timeout{};
        timeout.tv_sec = timeout_seconds;
        timeout.tv_usec = 0;

        (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        if (::connect(fd, item->ai_addr, item->ai_addrlen) == 0) {
            connected_fd = fd;
            break;
        }

        close_fd(fd);
    }

    ::freeaddrinfo(result);

    if (connected_fd < 0) {
        throw std::runtime_error(
            "failed to connect to " + url.host + ":" + std::to_string(url.port)
        );
    }

    return connected_fd;
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
            throw std::runtime_error("invalid Content-Length in HTTP response");
        }

        if (value > MAX_HTTP_RESPONSE_BODY_BYTES) {
            throw std::runtime_error("HTTP response body exceeds maximum allowed size");
        }

        return static_cast<std::size_t>(value);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid Content-Length in HTTP response");
    }
}

HttpClientResponse parse_http_response(
    const std::vector<std::uint8_t>& header_bytes,
    const std::vector<std::uint8_t>& body
) {
    const std::string header_text(
        reinterpret_cast<const char*>(header_bytes.data()),
        header_bytes.size()
    );

    std::istringstream stream(header_text);

    std::string status_line;
    if (!std::getline(stream, status_line)) {
        throw std::runtime_error("missing HTTP response status line");
    }

    if (!status_line.empty() && status_line.back() == '\r') {
        status_line.pop_back();
    }

    std::istringstream status_stream(status_line);

    std::string version;
    HttpClientResponse response;

    status_stream >> version >> response.status_code;
    std::getline(status_stream, response.reason);
    response.reason = trim_copy(response.reason);

    if (version.empty() || response.status_code <= 0) {
        throw std::runtime_error("malformed HTTP response status line");
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

        response.headers[key] = value;
    }

    response.body = body;
    return response;
}

HttpClientResponse read_http_response(int fd) {
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
            throw std::runtime_error(
                std::string("recv failed while reading HTTP response: ") + std::strerror(errno)
            );
        }

        if (rc == 0) {
            if (header_end == std::string::npos) {
                throw std::runtime_error("connection closed before HTTP response header was received");
            }
            break;
        }

        buffer.insert(buffer.end(), temp, temp + rc);

        if (header_end == std::string::npos) {
            header_end = find_header_end(buffer);

            if (header_end == std::string::npos && buffer.size() > MAX_HTTP_RESPONSE_HEADER_BYTES) {
                throw std::runtime_error("HTTP response headers exceed maximum allowed size");
            }

            if (header_end != std::string::npos) {
                std::vector<std::uint8_t> header_bytes(buffer.begin(), buffer.begin() + header_end);
                std::vector<std::uint8_t> partial_body(
                    buffer.begin() + static_cast<std::ptrdiff_t>(header_end + 4),
                    buffer.end()
                );

                HttpClientResponse partial = parse_http_response(header_bytes, partial_body);
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

                return parse_http_response(header_bytes, body);
            }
        }
    }

    if (header_end == std::string::npos) {
        throw std::runtime_error("HTTP response missing header terminator");
    }

    std::vector<std::uint8_t> header_bytes(buffer.begin(), buffer.begin() + header_end);
    std::vector<std::uint8_t> body(
        buffer.begin() + static_cast<std::ptrdiff_t>(header_end + 4),
        buffer.end()
    );

    return parse_http_response(header_bytes, body);
}

} // namespace

ParsedUrl parse_http_url(const std::string& url) {
    constexpr const char* prefix = "http://";

    if (url.rfind(prefix, 0) != 0) {
        throw std::runtime_error("only http:// URLs are supported: " + url);
    }

    const std::string rest = url.substr(std::strlen(prefix));

    const std::size_t slash_pos = rest.find('/');
    const std::string host_port = slash_pos == std::string::npos
        ? rest
        : rest.substr(0, slash_pos);

    ParsedUrl parsed;
    parsed.path = slash_pos == std::string::npos
        ? "/"
        : rest.substr(slash_pos);

    const std::size_t colon_pos = host_port.rfind(':');

    if (colon_pos == std::string::npos) {
        parsed.host = host_port;
        parsed.port = 80;
    } else {
        parsed.host = host_port.substr(0, colon_pos);
        const std::string port_str = host_port.substr(colon_pos + 1);

        try {
            parsed.port = std::stoi(port_str);
        } catch (const std::exception&) {
            throw std::runtime_error("invalid URL port: " + port_str);
        }
    }

    if (parsed.host.empty()) {
        throw std::runtime_error("URL host is empty: " + url);
    }

    if (parsed.port <= 0 || parsed.port > 65535) {
        throw std::runtime_error("URL port out of range: " + std::to_string(parsed.port));
    }

    if (parsed.path.empty()) {
        parsed.path = "/";
    }

    return parsed;
}

HttpClientResponse http_post_binary(
    const std::string& url,
    const std::vector<std::uint8_t>& body,
    int timeout_seconds
) {
    const ParsedUrl parsed = parse_http_url(url);
    const int fd = connect_tcp(parsed, timeout_seconds);

    try {
        std::ostringstream request_header;
        request_header << "POST " << parsed.path << " HTTP/1.1\r\n";
        request_header << "Host: " << parsed.host << ":" << parsed.port << "\r\n";
        request_header << "Content-Type: application/octet-stream\r\n";
        request_header << "Content-Length: " << body.size() << "\r\n";
        request_header << "Connection: close\r\n";
        request_header << "\r\n";

        const std::string header_text = request_header.str();

        if (!send_all(fd, header_text)) {
            throw std::runtime_error("failed to send HTTP request header");
        }

        if (!body.empty() && !send_all(fd, body.data(), body.size())) {
            throw std::runtime_error("failed to send HTTP request body");
        }

        HttpClientResponse response = read_http_response(fd);
        close_fd(fd);
        return response;
    } catch (...) {
        close_fd(fd);
        throw;
    }
}

} // namespace dli::common