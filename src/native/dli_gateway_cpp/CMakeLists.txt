add_library(dli-gateway-config STATIC
    src/config.cpp
)

target_include_directories(dli-gateway-config PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dli-gateway-config PUBLIC
    dli-common-cpp
)

target_compile_features(dli-gateway-config PUBLIC cxx_std_17)

target_compile_options(dli-gateway-config PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(dli-gateway-config PRIVATE -O3)
endif()

add_executable(dli-gateway-cpp
    src/main.cpp
    src/server.cpp
    src/generation_loop.cpp
    src/stage_client.cpp
    src/tokenizer.cpp
    src/sampler.cpp
)

target_include_directories(dli-gateway-cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dli-gateway-cpp PRIVATE
    dli-common-cpp
    dli-gateway-config
    llama
)

target_compile_features(dli-gateway-cpp PRIVATE cxx_std_17)

target_compile_options(dli-gateway-cpp PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(dli-gateway-cpp PRIVATE -O3)
endif()

set_target_properties(dli-gateway-cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(test-gateway-config
    tests/test_gateway_config.cpp
)

target_include_directories(test-gateway-config PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test-gateway-config PRIVATE
    dli-common-cpp
    dli-gateway-config
)

target_compile_features(test-gateway-config PRIVATE cxx_std_17)

target_compile_options(test-gateway-config PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(test-gateway-config PRIVATE -O3)
endif()

add_test(NAME test-gateway-config COMMAND test-gateway-config)

set_tests_properties(test-gateway-config PROPERTIES
    ENVIRONMENT "DLI_TEST_STAGE_MAP_PATH=${CMAKE_SOURCE_DIR}/../../configs/stage_map.yaml"
)

add_executable(test-tokenizer
    tests/test_tokenizer.cpp
    src/tokenizer.cpp
)

target_include_directories(test-tokenizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test-tokenizer PRIVATE
    dli-common-cpp
    llama
)

target_compile_features(test-tokenizer PRIVATE cxx_std_17)

target_compile_options(test-tokenizer PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(test-tokenizer PRIVATE -O3)
endif()

add_test(NAME test-tokenizer COMMAND test-tokenizer)

add_executable(test-llama-tokenizer-optional
    tests/test_llama_tokenizer_optional.cpp
    src/tokenizer.cpp
)

target_include_directories(test-llama-tokenizer-optional PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test-llama-tokenizer-optional PRIVATE
    dli-common-cpp
    llama
)

target_compile_features(test-llama-tokenizer-optional PRIVATE cxx_std_17)

target_compile_options(test-llama-tokenizer-optional PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(test-llama-tokenizer-optional PRIVATE -O3)
endif()

add_test(NAME test-llama-tokenizer-optional COMMAND test-llama-tokenizer-optional)