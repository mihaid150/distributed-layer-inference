#pragma once

#include "dli/common/gguf_inspector.hpp"
#include "dli/common/partition_plan.hpp"

#include <string>
#include <vector>

namespace dli::common {

struct PartitionTensorAssignment {
    std::string partition_id;

    std::vector<std::string> tensor_names;
    std::vector<std::string> missing_required_names;
    std::vector<std::string> missing_required_prefixes;
};

PartitionTensorAssignment assign_tensors_to_partition(
    const GgufInspection& inspection,
    const TensorNamePlan& plan
);

std::vector<PartitionTensorAssignment> assign_tensors_to_partitions(
    const GgufInspection& inspection,
    const std::vector<TensorNamePlan>& plans
);

} // namespace dli::common