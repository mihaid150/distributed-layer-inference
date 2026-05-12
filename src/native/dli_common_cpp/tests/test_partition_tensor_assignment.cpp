#include "dli/common/partition_tensor_assignment.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace {

dli::common::GgufInspection fake_inspection() {
    dli::common::GgufInspection inspection;
    inspection.path = "fake.gguf";

    const std::vector<std::string> names = {
        "token_embd.weight",

        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_up.weight",

        "blk.1.attn_norm.weight",
        "blk.1.attn_q.weight",

        "blk.17.attn_norm.weight",
        "blk.17.attn_q.weight",
        "blk.21.ffn_up.weight",

        "output_norm.weight",
        "output.weight"
    };

    for (const auto& name : names) {
        dli::common::GgufTensorInfo info;
        info.name = name;
        info.type_name = "q4_K";
        info.shape = {1, 1};
        info.byte_size = 16;
        info.offset = 0;
        inspection.tensors.push_back(info);
    }

    inspection.tensor_count = static_cast<int>(inspection.tensors.size());
    return inspection;
}

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

void test_assignment_for_embedding_partition() {
    const auto inspection = fake_inspection();

    dli::common::PartitionComponentsPlan components;
    components.embedding = true;
    components.layers = {0, 1};
    components.norm = false;
    components.lm_head = false;

    const auto plan = dli::common::build_llama_tensor_name_plan(
        "partition-1",
        components
    );

    const auto assignment = dli::common::assign_tensors_to_partition(
        inspection,
        plan
    );

    assert(assignment.partition_id == "partition-1");
    assert(contains(assignment.tensor_names, "token_embd.weight"));
    assert(contains(assignment.tensor_names, "blk.0.attn_q.weight"));
    assert(contains(assignment.tensor_names, "blk.1.attn_q.weight"));
    assert(!contains(assignment.tensor_names, "blk.17.attn_q.weight"));
    assert(!contains(assignment.tensor_names, "output.weight"));

    assert(assignment.missing_required_names.empty());
    assert(assignment.missing_required_prefixes.empty());
}

void test_assignment_for_terminal_partition() {
    const auto inspection = fake_inspection();

    dli::common::PartitionComponentsPlan components;
    components.embedding = false;
    components.layers = {17, 21};
    components.norm = true;
    components.lm_head = true;

    const auto plan = dli::common::build_llama_tensor_name_plan(
        "partition-4",
        components
    );

    const auto assignment = dli::common::assign_tensors_to_partition(
        inspection,
        plan
    );

    assert(assignment.partition_id == "partition-4");
    assert(contains(assignment.tensor_names, "blk.17.attn_q.weight"));
    assert(contains(assignment.tensor_names, "blk.21.ffn_up.weight"));
    assert(contains(assignment.tensor_names, "output_norm.weight"));
    assert(contains(assignment.tensor_names, "output.weight"));
    assert(!contains(assignment.tensor_names, "token_embd.weight"));

    assert(assignment.missing_required_names.empty());
    assert(assignment.missing_required_prefixes.empty());
}

void test_missing_required_name_is_reported() {
    auto inspection = fake_inspection();

    inspection.tensors.erase(
        std::remove_if(
            inspection.tensors.begin(),
            inspection.tensors.end(),
            [](const dli::common::GgufTensorInfo& info) {
                return info.name == "output.weight";
            }
        ),
        inspection.tensors.end()
    );

    dli::common::PartitionComponentsPlan components;
    components.lm_head = true;

    const auto plan = dli::common::build_llama_tensor_name_plan(
        "partition-terminal",
        components
    );

    const auto assignment = dli::common::assign_tensors_to_partition(
        inspection,
        plan
    );

    assert(contains(assignment.missing_required_names, "output.weight"));
}

} // namespace

int main() {
    test_assignment_for_embedding_partition();
    test_assignment_for_terminal_partition();
    test_missing_required_name_is_reported();

    std::cout << "test_partition_tensor_assignment: OK\n";
    return 0;
}