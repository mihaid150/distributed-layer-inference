cmake_minimum_required(VERSION 3.20)

project(dli_stage_cpp LANGUAGES CXX)

add_library(dli-stage-core STATIC
    src/protocol.cpp
    src/server.cpp
    src/runtimes/stub_runtime.cpp
)

target_include_directories(dli-stage-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(dli-stage-core PUBLIC cxx_std_17)

target_compile_options(dli-stage-core PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(dli-stage-cpp
    src/main.cpp
)

target_link_libraries(dli-stage-cpp PRIVATE
    dli-stage-core
)

target_compile_features(dli-stage-cpp PRIVATE cxx_std_17)

target_compile_options(dli-stage-cpp PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(test-protocol
    tests/test_protocol.cpp
)

target_link_libraries(test-protocol PRIVATE
    dli-stage-core
)

target_compile_features(test-protocol PRIVATE cxx_std_17)

target_compile_options(test-protocol PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(dli-stage-core PRIVATE -O3)
    target_compile_options(dli-stage-cpp PRIVATE -O3)
    target_compile_options(test-protocol PRIVATE -O3)
endif()

enable_testing()
add_test(NAME test-protocol COMMAND test-protocol)