#include "dli/common/gguf_inspector.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

std::string optional_model_path() {
    const char* env = std::getenv("DLI_TEST_GGUF_MODEL_PATH");
    if (env == nullptr) {
        return "";
    }
    return env;
}

} // namespace

int main() {
    const std::string model_path = optional_model_path();

    if (model_path.empty()) {
        std::cout << "test_gguf_inspector_optional: SKIPPED "
                  << "(DLI_TEST_GGUF_MODEL_PATH not set)\n";
        return 0;
    }

    try {
        const auto inspection = dli::common::inspect_gguf_tensors(model_path);

        if (inspection.tensor_count <= 0) {
            std::cerr << "expected GGUF tensor_count > 0\n";
            return 1;
        }

        if (!dli::common::gguf_contains_tensor(inspection, "token_embd.weight")) {
            std::cerr << "expected token_embd.weight tensor\n";
            return 1;
        }

        if (!dli::common::gguf_contains_tensor(inspection, "output.weight")) {
            std::cerr << "expected output.weight tensor\n";
            return 1;
        }

        if (!dli::common::gguf_contains_tensor_with_prefix(inspection, "blk.0.")) {
            std::cerr << "expected at least one blk.0.* tensor\n";
            return 1;
        }

        std::cout << "tensor_count=" << inspection.tensor_count << "\n";
        std::cout << "test_gguf_inspector_optional: OK\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "test_gguf_inspector_optional failed: " << exc.what() << "\n";
        return 1;
    }
}