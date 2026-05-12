cmake_minimum_required(VERSION 3.20)

project(dli_tools_cpp LANGUAGES CXX)

add_executable(dli-gguf-shard-writer
    src/gguf_shard_writer_tool.cpp
)

target_include_directories(dli-gguf-shard-writer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/dli_common_cpp/include
    ${CMAKE_SOURCE_DIR}/dli_gateway_cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)

target_link_libraries(dli-gguf-shard-writer PRIVATE
    dli-common-cpp
    dli-gateway-config
    ggml
)

add_executable(dli-gguf-shard-validate
    src/gguf_shard_validate_tool.cpp
)

target_include_directories(dli-gguf-shard-validate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/dli_common_cpp/include
    ${CMAKE_SOURCE_DIR}/dli_gateway_cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)

target_link_libraries(dli-gguf-shard-validate PRIVATE
    dli-common-cpp
    dli-gateway-config
    ggml
)

foreach(target_name
    dli-gguf-shard-writer
    dli-gguf-shard-validate
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

    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()

# Optional: if the previous manifest-writer step has already added this file,
# build it from the same tools module. This keeps this bundle compatible with
# both states of the branch.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/partition_manifest_tool.cpp)
    add_executable(dli-partition-manifest
        src/partition_manifest_tool.cpp
    )

    target_include_directories(dli-partition-manifest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/dli_common_cpp/include
        ${CMAKE_SOURCE_DIR}/dli_gateway_cpp/include
    )

    target_link_libraries(dli-partition-manifest PRIVATE
        dli-common-cpp
        dli-gateway-config
    )

    target_compile_features(dli-partition-manifest PRIVATE cxx_std_17)
    target_compile_options(dli-partition-manifest PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(dli-partition-manifest PRIVATE -O3)
    endif()

    set_target_properties(dli-partition-manifest PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()