#include "dli/gateway/tokenizer.hpp"

#include "llama.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void test_stub_tokenizer() {
    const auto tokenizer = dli::gateway::make_stub_tokenizer();

    const auto result = tokenizer->tokenize("hello distributed inference");

    assert(tokenizer->backend_name() == "stub");
    assert(result.prompt == "hello distributed inference");
    assert(result.token_ids.size() == 4);
    assert(result.token_ids[0] == 1);
    assert(result.token_ids[1] == 100);
    assert(result.token_ids[2] == 101);
    assert(result.token_ids[3] == 102);
}

void test_factory_returns_stub_for_empty_model_path() {
    const auto tokenizer = dli::gateway::make_tokenizer_for_model_path("");

    assert(tokenizer->backend_name() == "stub");
}

void test_llama_tokenizer_rejects_missing_file() {
    bool rejected = false;

    try {
        const auto tokenizer =
            dli::gateway::make_tokenizer_for_model_path("/tmp/does-not-exist-dli-tokenizer-test.gguf");
        (void)tokenizer;
    } catch (const std::exception&) {
        rejected = true;
    }

    if (!rejected) {
        std::cerr << "expected missing model path to be rejected\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    llama_backend_init();

    test_stub_tokenizer();
    test_factory_returns_stub_for_empty_model_path();
    test_llama_tokenizer_rejects_missing_file();

    llama_backend_free();

    std::cout << "test_tokenizer: OK\n";
    return 0;
}