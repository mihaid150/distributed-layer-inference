#include "dli/gateway/tokenizer.hpp"

#include "llama.h"

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

void test_real_llama_tokenizer(const std::string& model_path) {
    const auto tokenizer = dli::gateway::make_tokenizer_for_model_path(model_path);

    if (tokenizer->backend_name() != "llama.cpp") {
        std::cerr << "expected tokenizer backend llama.cpp, got "
                  << tokenizer->backend_name() << "\n";
        std::exit(1);
    }

    const auto result = tokenizer->tokenize("Write one short sentence about distributed inference.");

    if (result.token_ids.empty()) {
        std::cerr << "expected non-empty token list\n";
        std::exit(1);
    }

    const std::vector<int> first_tokens = {
        static_cast<int>(result.token_ids.front())
    };

    const std::string text = tokenizer->detokenize(first_tokens);

    if (text.empty()) {
        std::cerr << "expected non-empty detokenized text for first token\n";
        std::exit(1);
    }
    
    std::cout << "tokenizer_backend=" << tokenizer->backend_name() << "\n";
    std::cout << "token_count=" << result.token_ids.size() << "\n";
    std::cout << "tokens=";

    for (std::size_t i = 0; i < result.token_ids.size(); ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << result.token_ids[i];
    }

    std::cout << "\n";
}

} // namespace

int main() {
    const std::string model_path = optional_model_path();

    if (model_path.empty()) {
        std::cout << "test_llama_tokenizer_optional: SKIPPED "
                  << "(DLI_TEST_GGUF_MODEL_PATH not set)\n";
        return 0;
    }

    llama_backend_init();

    try {
        test_real_llama_tokenizer(model_path);
    } catch (const std::exception& exc) {
        llama_backend_free();
        std::cerr << "test_llama_tokenizer_optional failed: " << exc.what() << "\n";
        return 1;
    }

    llama_backend_free();

    std::cout << "test_llama_tokenizer_optional: OK\n";
    return 0;
}