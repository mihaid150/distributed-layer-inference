cmake_minimum_required(VERSION 3.20)

project(dli_stage_cpp LANGUAGES CXX)

add_executable(dli-stage-cpp
    src/main.cpp
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