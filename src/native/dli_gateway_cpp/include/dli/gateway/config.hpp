#pragma once

#include <string>
#include <vector>

namespace dli::gateway {

struct PartitionComponents {
    bool embedding = false;
    std::vector<int> layers;
    bool norm = false;
    bool lm_head = false;
};

struct PartitionNodeConfig {
    int stage_id = 0;

    // Generic static-allocation name derived from current stage_map.
    // Later this can become explicit in YAML as partition_id.
    std::string partition_id;

    std::string service_name;
    std::string physical_node;
    std::string partition_file;
    std::string next_stage_url;

    PartitionComponents components;
};

struct PartitionGraphValidation {
    bool valid = false;
    std::string error;
    int assigned_layer_count = 0;
    int embedding_owner_count = 0;
    int lm_head_owner_count = 0;
    std::string terminal_partition_id;
};

struct GatewayConfig {
    std::string config_path = "/app/configs/stage_map.yaml";
    std::string service_name = "dli-gateway-cpp";
    std::string model_name = "";
    std::string model_path = "";
    std::string first_stage_url = "";
    int port = 8000;
    int num_layers = 0;

    std::vector<PartitionNodeConfig> partitions;
    PartitionGraphValidation partition_validation;
};

GatewayConfig load_gateway_config_from_file(const std::string& config_path);

GatewayConfig merge_gateway_config(
    GatewayConfig file_config,
    const GatewayConfig& override_config
);

PartitionGraphValidation validate_partition_graph(
    const std::vector<PartitionNodeConfig>& partitions,
    int num_layers
);

} // namespace dli::gateway