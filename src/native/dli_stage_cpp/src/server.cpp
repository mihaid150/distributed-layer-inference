#include "dli_stage/server.hpp"

#include "dli/common/http.hpp"
#include "dli/common/http_client.hpp"
#include "dli/common/json_escape.hpp"
#include "dli/common/metadata.hpp"
#include "dli/common/protocol.hpp"
#include "dli/common/metrics.hpp"
#include "dli/common/tensor.hpp"
#include "dli_stage/runtime.hpp"

#include <algorithm>
#include <thread>
#include <future>
#include <deque>
#include <condition_variable>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace dli_stage {

namespace {


bool env_bool(const char* name, bool default_value = false) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

int env_int(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    try {
        return std::stoi(raw);
    } catch (...) {
        return default_value;
    }
}

bool native_stage_chaining_enabled(const RuntimeRequest& request) {
    return request.native_stage_chaining_enabled || env_bool("DLI_NATIVE_STAGE_CHAINING", false);
}

double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

dli::common::PersistentHttpClient& persistent_client_for_url(const std::string& url) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::unique_ptr<dli::common::PersistentHttpClient>> clients;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = clients.find(url);
    if (it == clients.end()) {
        it = clients.emplace(
            url,
            std::make_unique<dli::common::PersistentHttpClient>(
                url,
                env_int("DLI_NATIVE_STAGE_FORWARD_TIMEOUT_SECONDS", 120)
            )
        ).first;
    }
    return *it->second;
}

dli::common::HttpResponse handle_forward_binary(
    const dli::common::HttpRequest& request,
    const StageConfig& config,
    StageRuntime& runtime
);


class ForwardBroker {
public:
    ForwardBroker(StageConfig config, StageRuntime& runtime)
        : config_(std::move(config)),
          runtime_(runtime),
          max_queue_size_(std::max(1, env_int("DLI_NATIVE_FORWARD_QUEUE_SIZE", 1))),
          worker_([this] { run(); }) {}

    ~ForwardBroker() {
        stop();
    }

    ForwardBroker(const ForwardBroker&) = delete;
    ForwardBroker& operator=(const ForwardBroker&) = delete;

    std::future<dli::common::HttpResponse> submit(
        dli::common::HttpRequest request
    ) {
        ForwardJob job;
        job.request = std::move(request);
        auto future = job.promise.get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_requested_) {
                job.promise.set_value(
                    dli::common::make_json_response(
                        503,
                        "Service Unavailable",
                        dli::common::http_error_json("native forward broker is stopping")
                    )
                );
                return future;
            }

            if (queue_.size() >= static_cast<std::size_t>(max_queue_size_)) {
                job.promise.set_value(
                    dli::common::make_json_response(
                        429,
                        "Too Many Requests",
                        dli::common::http_error_json("native forward queue is full")
                    )
                );
                return future;
            }

            queue_.push_back(std::move(job));
        }

        cv_.notify_one();
        return future;
    }

    std::size_t queue_depth() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    struct ForwardJob {
        dli::common::HttpRequest request;
        std::promise<dli::common::HttpResponse> promise;
    };

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_requested_) {
                return;
            }
            stop_requested_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void run() {
        while (true) {
            ForwardJob job;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_requested_ || !queue_.empty(); });

                if (stop_requested_ && queue_.empty()) {
                    return;
                }

                job = std::move(queue_.front());
                queue_.pop_front();
            }

            try {
                job.promise.set_value(
                    handle_forward_binary(job.request, config_, runtime_)
                );
            } catch (const std::exception& exc) {
                job.promise.set_value(
                    dli::common::make_json_response(
                        500,
                        "Internal Server Error",
                        dli::common::http_error_json(exc.what())
                    )
                );
            } catch (...) {
                job.promise.set_value(
                    dli::common::make_json_response(
                        500,
                        "Internal Server Error",
                        dli::common::http_error_json("unknown native forward broker failure")
                    )
                );
            }
        }
    }

    StageConfig config_;
    StageRuntime& runtime_;
    int max_queue_size_ = 1;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<ForwardJob> queue_;
    std::thread worker_;
    bool stop_requested_ = false;
};

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
        << "\"kv_cache_step_valid\":" << (metrics.kv_cache_step_valid ? "true" : "false") << ","
        << "\"kv_cache_seq_before\":" << metrics.kv_cache_seq_before << ","
        << "\"kv_cache_seq_after\":" << metrics.kv_cache_seq_after << ","
        << "\"kv_cache_bytes\":" << metrics.kv_cache_bytes << ","
        << "\"kv_cache_valid\":" << (metrics.kv_cache_valid ? "true" : "false") << ","
        << "\"model_load_ms\":" << metrics.model_load_ms << ","
        << "\"memory_rss_mb\":" << metrics.memory_rss_mb << ","
        << "\"memory_cgroup_current_mb\":" << metrics.memory_cgroup_current_mb << ","
        << "\"memory_cgroup_limit_mb\":" << metrics.memory_cgroup_limit_mb << ","
        << "\"memory_cgroup_percent\":" << metrics.memory_cgroup_percent << ","
        << "\"model_file_size_mb\":" << metrics.model_file_size_mb << ","
        << "\"session_count\":" << metrics.session_count << ","
        << "\"session_kv_cache_bytes\":" << metrics.session_kv_cache_bytes
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
    const dli::common::TensorMetadata& tensor_metadata =
    response.output_tensor.metadata.shape.empty()
        ? request.input_tensor.metadata
        : response.output_tensor.metadata;
    std::ostringstream out;
    out
        << "{"
        << "\"ok\":true,"
        << "\"service\":\"" << dli::common::json_escape(config.service_name) << "\","
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
        << "\"dtype\":\"" << dli::common::json_escape(tensor_metadata.dtype) << "\","
        << "\"shape\":" << shape_json(tensor_metadata.shape) << ","
        << "\"byte_order\":\"" << dli::common::json_escape(tensor_metadata.byte_order) << "\""
        << "},"
        << "\"feature_flags\":{"
        << "\"kv_cache_enabled\":" << (request.kv_cache_enabled ? "true" : "false") << ","
        << "\"native_stage_chaining_enabled\":" << (request.native_stage_chaining_enabled ? "true" : "false") << ","
        << "\"transport_mode\":\"" << dli::common::json_escape(request.transport_mode) << "\","
        << "\"activation_precision\":\"" << dli::common::json_escape(request.activation_precision) << "\","
        << "\"persistent_sessions_enabled\":" << (request.persistent_sessions_enabled ? "true" : "false") << ","
        << "\"metadata_level\":\"" << dli::common::json_escape(request.metadata_level) << "\""
        << "},"
        << "\"metrics\":" << metrics_json(response.metrics) << ","
        << "\"backend_metadata\":"
        << (response.backend_metadata_json.empty() ? "{}" : response.backend_metadata_json)
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
        << "\"native_partition_file\":\"" << dli::common::json_escape(config.native_partition_file) << "\","
        << "\"backend\":\"" << dli::common::json_escape(config.backend) << "\","
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

    request.activation_precision = parsed.activation_precision;
    request.transport_mode = parsed.transport_mode;
    request.rebalance_profile = parsed.rebalance_profile;
    request.persistent_sessions_enabled = parsed.persistent_sessions_enabled;
    request.topology_aware_routing = parsed.topology_aware_routing;
    request.native_stage_chaining_enabled = parsed.native_stage_chaining_enabled;
    request.metadata_level = parsed.metadata_level;

    request.has_temperature = parsed.has_temperature;
    request.temperature = parsed.temperature;

    request.has_top_k = parsed.has_top_k;
    request.top_k = parsed.top_k;

    request.has_top_p = parsed.has_top_p;
    request.top_p = parsed.top_p;

    request.has_next_token_id = parsed.has_next_token_id;
    request.next_token_id = parsed.next_token_id;
    request.stage_input_token_count = parsed.stage_input_token_count;

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



        if (
            native_stage_chaining_enabled(runtime_request) &&
            !runtime_response.is_final_stage &&
            !config.next_stage_url.empty()
        ) {
            const std::vector<std::uint8_t> next_body =
                dli::common::encode_frame(output);

            const auto rpc_start = std::chrono::steady_clock::now();
            const dli::common::HttpClientResponse next_response =
                persistent_client_for_url(config.next_stage_url).post_binary(next_body);
            const auto rpc_end = std::chrono::steady_clock::now();
            const double next_rpc_wall_ms = elapsed_ms(rpc_start, rpc_end);

            if (next_response.status_code < 200 || next_response.status_code >= 300) {
                return dli::common::make_json_response(
                    502,
                    "Bad Gateway",
                    dli::common::http_error_json(
                        "next native stage failed with HTTP " +
                        std::to_string(next_response.status_code)
                    )
                );
            }

            dli::common::DliFrame downstream =
                dli::common::decode_frame(next_response.body);

            downstream.metadata_json = dli::common::merge_chain_metrics(
                downstream.metadata_json,
                runtime_response.metrics,
                next_rpc_wall_ms,
                static_cast<std::uint64_t>(next_body.size()),
                static_cast<std::uint64_t>(next_response.body.size())
            );

            return dli::common::make_binary_response(
                200,
                "OK",
                dli::common::encode_frame(downstream)
            );
        }
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
    StageRuntime& runtime,
    ForwardBroker& forward_broker
) {
    if (request.method == "GET" && request.path == "/health") {
        return dli::common::make_json_response(200, "OK", health_json(config));
    }

    if (request.method == "GET" && request.path == "/config") {
        return dli::common::make_json_response(200, "OK", config_json(config));
    }

    if (request.method == "POST" && request.path == "/forward-binary") {
        auto future = forward_broker.submit(request);
        return future.get();
    }

    return dli::common::make_json_response(404, "Not Found", not_found_json(request));
}

void close_fd(int fd) {
    if (fd >= 0) {
        while (::close(fd) < 0 && errno == EINTR) {
        }
    }
}

void handle_stage_client_connection(
    int client_fd,
    const StageConfig& config,
    StageRuntime& runtime,
    ForwardBroker& forward_broker,
    std::atomic<bool>& stop_requested
) {
    while (!stop_requested.load()) {
        try {
            const dli::common::HttpRequest request = dli::common::read_http_request(client_fd);
            dli::common::HttpResponse response = handle_request(
                request,
                config,
                runtime,
                forward_broker
            );
            response.keep_alive = dli::common::request_wants_keep_alive(request);
            dli::common::send_http_response(client_fd, response);
            if (!response.keep_alive) {
                break;
            }
        } catch (const std::exception& exc) {
            dli::common::HttpResponse response = dli::common::make_json_response(
                400,
                "Bad Request",
                dli::common::http_error_json(exc.what())
            );
            response.keep_alive = false;
            dli::common::send_http_response(client_fd, response);
            break;
        } catch (...) {
            dli::common::HttpResponse response = dli::common::make_json_response(
                400,
                "Bad Request",
                dli::common::http_error_json("unknown stage connection failure")
            );
            response.keep_alive = false;
            dli::common::send_http_response(client_fd, response);
            break;
        }
    }

    close_fd(client_fd);
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

    ForwardBroker forward_broker(config_, *runtime_);

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

        std::thread(
            handle_stage_client_connection,
            client_fd,
            std::cref(config_),
            std::ref(*runtime_),
            std::ref(forward_broker),
            std::ref(stop_requested_)
        ).detach();
    }

    close_fd(server_fd);
    return 0;
}

} // namespace dli_stage