cmake_minimum_required(VERSION 3.20)

project(dli_common_cpp LANGUAGES CXX)

enable_testing()

add_library(dli-common-cpp STATIC
    src/protocol.cpp
    src/metadata.cpp
    src/tensor.cpp
    src/metrics.cpp
    src/json_escape.cpp
    src/http.cpp
    src/http_client.cpp
)

target_include_directories(dli-common-cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(dli-common-cpp PUBLIC cxx_std_17)

target_compile_options(dli-common-cpp PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(dli-common-cpp PRIVATE -O3)
endif()

add_executable(test-protocol
    tests/test_protocol.cpp
)

target_link_libraries(test-protocol PRIVATE
    dli-common-cpp
)

target_compile_features(test-protocol PRIVATE cxx_std_17)

target_compile_options(test-protocol PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(test-metadata
    tests/test_metadata.cpp
)

target_link_libraries(test-metadata PRIVATE
    dli-common-cpp
)

target_compile_features(test-metadata PRIVATE cxx_std_17)

target_compile_options(test-metadata PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(test-protocol PRIVATE -O3)
    target_compile_options(test-metadata PRIVATE -O3)
endif()


add_test(NAME test-protocol COMMAND test-protocol)
add_test(NAME test-metadata COMMAND test-metadata)

add_executable(test-llama-backend-smoke
    tests/test_llama_backend_smoke.cpp
)

target_link_libraries(test-llama-backend-smoke PRIVATE
    llama
)

target_compile_features(test-llama-backend-smoke PRIVATE cxx_std_17)

target_compile_options(test-llama-backend-smoke PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(test-llama-backend-smoke PRIVATE -O3)
endif()

add_test(NAME test-llama-backend-smoke COMMAND test-llama-backend-smoke)