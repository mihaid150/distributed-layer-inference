#include "dli_stage/config.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

std::string config_path_from_env_or_default() {
    const char* env = std::getenv("DLI_TEST_STAGE_MAP_PATH");
    if (env != nullptr && std::string(env).size() > 0) {
        return env;
    }

    return "configs/stage_map.yaml";
}

void test_stage_1_config(const std::string& config_path) {
    const dli_stage::StageConfig config =
        dli_stage::load_stage_config_from_file(config_path, 1);

    assert(config.stage_id == 1);
    assert(config.service_name == "inference-stage-1");
    assert(config.physical_node == "pinode7");
    assert(config.partition_file == "/app/models/stage_1.pt");
    assert(config.next_stage_url == "http://inference-stage-2:8000/forward");

    assert(config.components.embedding);
    assert(!config.components.norm);
    assert(!config.components.lm_head);

    const std::vector<int> expected_layers = {0, 1, 2, 3, 4, 5};
    assert(config.components.layers == expected_layers);
}

void test_stage_4_config(const std::string& config_path) {
    const dli_stage::StageConfig config =
        dli_stage::load_stage_config_from_file(config_path, 4);

    assert(config.stage_id == 4);
    assert(config.service_name == "inference-stage-4");
    assert(config.physical_node == "pinode10");
    assert(config.partition_file == "/app/models/stage_4.pt");
    assert(config.next_stage_url.empty());

    assert(!config.components.embedding);
    assert(config.components.norm);
    assert(config.components.lm_head);

    const std::vector<int> expected_layers = {17, 18, 19, 20, 21};
    assert(config.components.layers == expected_layers);
}

void test_merge_preserves_loaded_service_name(const std::string& config_path) {
    dli_stage::StageConfig file_config =
        dli_stage::load_stage_config_from_file(config_path, 1);

    dli_stage::StageConfig override_config;
    override_config.service_name.clear();
    override_config.config_path = config_path;
    override_config.stage_id = 1;
    override_config.port = 8001;

    const dli_stage::StageConfig merged =
        dli_stage::merge_stage_config(file_config, override_config);

    assert(merged.service_name == "inference-stage-1");
    assert(merged.port == 8001);
    assert(merged.stage_id == 1);
}

} // namespace

int main() {
    const std::string config_path = config_path_from_env_or_default();

    test_stage_1_config(config_path);
    test_stage_4_config(config_path);
    test_merge_preserves_loaded_service_name(config_path);

    std::cout << "test_stage_config: OK\n";
    return 0;
}