add_executable(dli-stage-cpp
    src/main.cpp
    src/config.cpp
    src/server.cpp
    src/runtimes/stub_runtime.cpp
    src/runtimes/llama_partial_runtime.cpp
)

target_include_directories(dli-stage-cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dli-stage-cpp PRIVATE
    dli-common-cpp
)

target_compile_features(dli-stage-cpp PRIVATE cxx_std_17)

target_compile_options(dli-stage-cpp PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(dli-stage-cpp PRIVATE -O3)
endif()

set_target_properties(dli-stage-cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(test-stage-config
    tests/test_stage_config.cpp
    src/config.cpp
)

target_include_directories(test-stage-config PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test-stage-config PRIVATE
    dli-common-cpp
)

target_compile_features(test-stage-config PRIVATE cxx_std_17)

target_compile_options(test-stage-config PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(test-stage-config PRIVATE -O3)
endif()

add_test(NAME test-stage-config COMMAND test-stage-config)

set_tests_properties(test-stage-config PROPERTIES
    ENVIRONMENT "DLI_TEST_STAGE_MAP_PATH=${CMAKE_SOURCE_DIR}/../../configs/stage_map.yaml"
)