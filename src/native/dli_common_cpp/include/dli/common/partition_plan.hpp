#pragma once

#include <string>
#include <vector>

namespace dli::common {

struct PartitionComponentsPlan {
    bool embedding = false;
    std::vector<int> layers;
    bool norm = false;
    bool lm_head = false;
};

struct TensorNamePlan {
    std::string partition_id;
    std::vector<std::string> required_tensor_prefixes;
    std::vector<std::string> required_tensor_names;
};

TensorNamePlan build_llama_tensor_name_plan(
    const std::string& partition_id,
    const PartitionComponentsPlan& components
);

bool tensor_matches_plan(
    const std::string& tensor_name,
    const TensorNamePlan& plan
);

} // namespace dli::common