#pragma once

#include <string>
#include <vector>

namespace dli_stage {

struct StageComponents {
    bool embedding = false;
    std::vector<int> layers;
    bool norm = false;
    bool lm_head = false;
};

struct StageConfig {
    int stage_id = 0;
    int port = 8000;

    std::string config_path = "/app/configs/stage_map.yaml";
    std::string service_name = "dli-stage-cpp";
    std::string physical_node;
    std::string partition_file;
    std::string next_stage_url;

    StageComponents components;
};

StageConfig load_stage_config_from_file(
    const std::string& config_path,
    int wanted_stage_id
);

StageConfig merge_stage_config(
    StageConfig file_config,
    const StageConfig& override_config
);

} // namespace dli_stage