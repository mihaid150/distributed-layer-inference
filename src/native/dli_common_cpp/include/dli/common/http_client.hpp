#pragma once

#include "dli/common/http.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dli::common {

struct ParsedUrl {
    std::string host;
    int port = 80;
    std::string path = "/";
};

struct HttpClientResponse {
    int status_code = 0;
    std::string reason;
    std::map<std::string, std::string> headers;
    std::vector<std::uint8_t> body;
};

ParsedUrl parse_http_url(const std::string& url);

HttpClientResponse http_post_binary(
    const std::string& url,
    const std::vector<std::uint8_t>& body,
    int timeout_seconds = 120
);



class PersistentHttpClient {
public:
    explicit PersistentHttpClient(std::string url, int timeout_seconds = 120);
    ~PersistentHttpClient();

    PersistentHttpClient(const PersistentHttpClient&) = delete;
    PersistentHttpClient& operator=(const PersistentHttpClient&) = delete;

    HttpClientResponse post_binary(const std::vector<std::uint8_t>& body);

private:
    ParsedUrl parsed_;
    int timeout_seconds_ = 120;
    int fd_ = -1;

    void ensure_connected();
    void close_connection();
    HttpClientResponse post_binary_once(const std::vector<std::uint8_t>& body);
};

} // namespace dli::common