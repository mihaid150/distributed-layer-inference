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
    src/partition_plan.cpp
    src/partition_tensor_assignment.cpp
    src/gguf_inspector.cpp
)

target_include_directories(dli-common-cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)

target_link_libraries(dli-common-cpp PUBLIC
    ggml
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

add_executable(test-metadata
    tests/test_metadata.cpp
)

target_link_libraries(test-metadata PRIVATE
    dli-common-cpp
)

add_executable(test-llama-backend-smoke
    tests/test_llama_backend_smoke.cpp
)

target_link_libraries(test-llama-backend-smoke PRIVATE
    llama
)

add_executable(test-partition-plan
    tests/test_partition_plan.cpp
)

target_link_libraries(test-partition-plan PRIVATE
    dli-common-cpp
)

add_executable(test-partition-tensor-assignment
    tests/test_partition_tensor_assignment.cpp
)

target_link_libraries(test-partition-tensor-assignment PRIVATE
    dli-common-cpp
)

add_executable(test-gguf-inspector-optional
    tests/test_gguf_inspector_optional.cpp
)

target_link_libraries(test-gguf-inspector-optional PRIVATE
    dli-common-cpp
)

add_executable(test-chain-metrics
    tests/test_chain_metrics.cpp
)

target_link_libraries(test-chain-metrics PRIVATE
    dli-common-cpp
)

foreach(target_name
    test-protocol
    test-metadata
    test-llama-backend-smoke
    test-partition-plan
    test-partition-tensor-assignment
    test-gguf-inspector-optional
)
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_compile_options(${target_name} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${target_name} PRIVATE -O3)
    endif()
endforeach()

add_test(NAME test-protocol COMMAND test-protocol)
add_test(NAME test-metadata COMMAND test-metadata)
add_test(NAME test-llama-backend-smoke COMMAND test-llama-backend-smoke)
add_test(NAME test-partition-plan COMMAND test-partition-plan)
add_test(NAME test-partition-tensor-assignment COMMAND test-partition-tensor-assignment)
add_test(NAME test-gguf-inspector-optional COMMAND test-gguf-inspector-optional)
add_test(NAME test-chain-metrics COMMAND test-chain-metrics)