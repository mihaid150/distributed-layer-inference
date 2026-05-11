cmake_minimum_required(VERSION 3.20)

project(dli_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_subdirectory(dli_common_cpp)
add_subdirectory(dli_stage_cpp)
add_subdirectory(dli_gateway_cpp)