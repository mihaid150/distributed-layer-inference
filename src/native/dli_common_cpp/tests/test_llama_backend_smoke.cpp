#include "llama.h"

#include <iostream>

int main() {
    std::cout << "test_llama_backend_smoke: initializing llama backend\n";

    llama_backend_init();

    std::cout << "test_llama_backend_smoke: freeing llama backend\n";

    llama_backend_free();

    std::cout << "test_llama_backend_smoke: OK\n";
    return 0;
}