#pragma once

#include "dli_stage/runtime.hpp"

namespace dli_stage {

class StubRuntime final : public StageRuntime {
public:
    RuntimeResponse forward(const RuntimeRequest& request) override;

    std::string backend_name() const override;
};

} // namespace dli_stage