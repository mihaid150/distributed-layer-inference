#include "dli/common/partition_tensor_assignment.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace dli::common {

PartitionTensorAssignment assign_tensors_to_partition(
    const GgufInspection& inspection,
    const TensorNamePlan& plan
) {
    PartitionTensorAssignment assignment;
    assignment.partition_id = plan.partition_id;

    for (const auto& required_name : plan.required_tensor_names) {
        if (!gguf_contains_tensor(inspection, required_name)) {
            assignment.missing_required_names.push_back(required_name);
        }
    }

    for (const auto& required_prefix : plan.required_tensor_prefixes) {
        if (!gguf_contains_tensor_with_prefix(inspection, required_prefix)) {
            assignment.missing_required_prefixes.push_back(required_prefix);
        }
    }

    for (const auto& tensor : inspection.tensors) {
        if (tensor_matches_plan(tensor.name, plan)) {
            assignment.tensor_names.push_back(tensor.name);
        }
    }

    std::sort(assignment.tensor_names.begin(), assignment.tensor_names.end());
    std::sort(
        assignment.missing_required_names.begin(),
        assignment.missing_required_names.end()
    );
    std::sort(
        assignment.missing_required_prefixes.begin(),
        assignment.missing_required_prefixes.end()
    );

    return assignment;
}

std::vector<PartitionTensorAssignment> assign_tensors_to_partitions(
    const GgufInspection& inspection,
    const std::vector<TensorNamePlan>& plans
) {
    std::vector<PartitionTensorAssignment> assignments;
    assignments.reserve(plans.size());

    for (const auto& plan : plans) {
        assignments.push_back(assign_tensors_to_partition(inspection, plan));
    }

    return assignments;
}

} // namespace dli::common