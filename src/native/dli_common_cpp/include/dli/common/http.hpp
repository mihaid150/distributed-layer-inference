#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dli::common {

struct HttpRequest {
    std::string method;
    std::string path;
    std::map<std::string, std::string> headers;
    std::vector<std::uint8_t> body;
};

struct HttpResponse {
    int status_code = 200;
    std::string reason = "OK";
    std::string content_type = "application/json";
    std::vector<std::uint8_t> body;
};

HttpRequest read_http_request(int fd);

void send_http_response(int fd, const HttpResponse& response);

HttpResponse make_json_response(
    int status_code,
    const std::string& reason,
    const std::string& json
);

HttpResponse make_text_response(
    int status_code,
    const std::string& reason,
    const std::string& text
);

HttpResponse make_binary_response(
    int status_code,
    const std::string& reason,
    const std::vector<std::uint8_t>& body
);

std::string http_error_json(const std::string& error);

} // namespace dli::common