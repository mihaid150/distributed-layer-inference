#include "dli/common/partition_plan.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool contains(
    const std::vector<std::string>& values,
    const std::string& wanted
) {
    for (const auto& value : values) {
        if (value == wanted) {
            return true;
        }
    }

    return false;
}

void test_embedding_partition_plan() {
    dli::common::PartitionComponentsPlan components;
    components.embedding = true;
    components.layers = {0, 1, 2, 3, 4, 5};
    components.norm = false;
    components.lm_head = false;

    const auto plan = dli::common::build_llama_tensor_name_plan(
        "partition-1",
        components
    );

    assert(plan.partition_id == "partition-1");

    assert(contains(plan.required_tensor_names, "token_embd.weight"));
    assert(!contains(plan.required_tensor_names, "output_norm.weight"));
    assert(!contains(plan.required_tensor_names, "output.weight"));

    assert(contains(plan.required_tensor_prefixes, "blk.0."));
    assert(contains(plan.required_tensor_prefixes, "blk.5."));
    assert(!contains(plan.required_tensor_prefixes, "blk.6."));
}

void test_terminal_partition_plan() {
    dli::common::PartitionComponentsPlan components;
    components.embedding = false;
    components.layers = {17, 18, 19, 20, 21};
    components.norm = true;
    components.lm_head = true;

    const auto plan = dli::common::build_llama_tensor_name_plan(
        "partition-4",
        components
    );

    assert(plan.partition_id == "partition-4");

    assert(!contains(plan.required_tensor_names, "token_embd.weight"));
    assert(contains(plan.required_tensor_names, "output_norm.weight"));
    assert(contains(plan.required_tensor_names, "output.weight"));

    assert(contains(plan.required_tensor_prefixes, "blk.17."));
    assert(contains(plan.required_tensor_prefixes, "blk.21."));
    assert(!contains(plan.required_tensor_prefixes, "blk.16."));
}

} // namespace

int main() {
    test_embedding_partition_plan();
    test_terminal_partition_plan();

    std::cout << "test_partition_plan: OK\n";
    return 0;
}