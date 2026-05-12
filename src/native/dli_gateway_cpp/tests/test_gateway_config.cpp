#include "dli/gateway/config.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string config_path_from_env_or_default() {
    const char* env = std::getenv("DLI_TEST_STAGE_MAP_PATH");
    if (env != nullptr && std::string(env).size() > 0) {
        return env;
    }

    return "configs/stage_map.yaml";
}

void test_gateway_config_loads_expected_fields(const std::string& config_path) {
    const dli::gateway::GatewayConfig config =
        dli::gateway::load_gateway_config_from_file(config_path);

    assert(config.config_path == config_path);
    assert(config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    assert(config.model_path.empty());
    assert(config.service_name == "inference-gateway");
    assert(config.port == 8000);
    assert(config.first_stage_url == "http://inference-stage-1:8000/forward");

    assert(config.partitions.size() == 4);

    const auto& p0 = config.partitions[0];
    assert(p0.stage_id == 1);
    assert(p0.partition_id == "partition-1");
    assert(p0.service_name == "inference-stage-1");
    assert(p0.physical_node == "pinode7");
    assert(p0.partition_file == "/app/models/stage_1.pt");
    assert(p0.next_stage_url == "http://inference-stage-2:8000/forward");
    assert(p0.components.embedding);
    assert(!p0.components.norm);
    assert(!p0.components.lm_head);
    assert((p0.components.layers == std::vector<int>{0, 1, 2, 3, 4, 5}));

    const auto& p3 = config.partitions[3];
    assert(p3.stage_id == 4);
    assert(p3.partition_id == "partition-4");
    assert(p3.service_name == "inference-stage-4");
    assert(p3.physical_node == "pinode10");
    assert(p3.partition_file == "/app/models/stage_4.pt");
    assert(p3.next_stage_url.empty());
    assert(!p3.components.embedding);
    assert(p3.components.norm);
    assert(p3.components.lm_head);
    assert((p3.components.layers == std::vector<int>{17, 18, 19, 20, 21}));

    assert(config.num_layers == 22);
    assert(config.partition_validation.valid);
    assert(config.partition_validation.error.empty());
    assert(config.partition_validation.assigned_layer_count == 22);
    assert(config.partition_validation.embedding_owner_count == 1);
    assert(config.partition_validation.lm_head_owner_count == 1);
    assert(config.partition_validation.terminal_partition_id == "partition-4");
}

void test_gateway_config_override_merge(const std::string& config_path) {
    dli::gateway::GatewayConfig file_config =
        dli::gateway::load_gateway_config_from_file(config_path);

    dli::gateway::GatewayConfig override_config;
    override_config.config_path = config_path;
    override_config.service_name.clear();
    override_config.model_name.clear();
    override_config.port = 8010;
    override_config.first_stage_url = "http://127.0.0.1:8001/forward-binary";
    override_config.model_path = "/tmp/tinyllama-q8_0.gguf";

    const dli::gateway::GatewayConfig merged =
        dli::gateway::merge_gateway_config(file_config, override_config);

    assert(merged.config_path == config_path);
    assert(merged.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    assert(merged.service_name == "inference-gateway");
    assert(merged.port == 8010);
    assert(merged.first_stage_url == "http://127.0.0.1:8001/forward-binary");
    assert(merged.model_path == "/tmp/tinyllama-q8_0.gguf");
}

} // namespace

int main() {
    const std::string config_path = config_path_from_env_or_default();

    test_gateway_config_loads_expected_fields(config_path);
    test_gateway_config_override_merge(config_path);

    std::cout << "test_gateway_config: OK\n";
    return 0;
}