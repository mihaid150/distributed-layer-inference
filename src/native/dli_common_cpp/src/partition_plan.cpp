#include "dli/common/partition_plan.hpp"

#include <string>

namespace dli::common {

namespace {

void add_layer_tensor_prefixes(
    TensorNamePlan& plan,
    int layer_id
) {
    const std::string prefix = "blk." + std::to_string(layer_id) + ".";

    plan.required_tensor_prefixes.push_back(prefix);
}

} // namespace

TensorNamePlan build_llama_tensor_name_plan(
    const std::string& partition_id,
    const PartitionComponentsPlan& components
) {
    TensorNamePlan plan;
    plan.partition_id = partition_id;

    if (components.embedding) {
        plan.required_tensor_names.push_back("token_embd.weight");
    }

    for (const int layer_id : components.layers) {
        add_layer_tensor_prefixes(plan, layer_id);
    }

    if (components.norm) {
        plan.required_tensor_names.push_back("output_norm.weight");
    }

    if (components.lm_head) {
        plan.required_tensor_names.push_back("output.weight");
    }

    return plan;
}

} // namespace dli::common