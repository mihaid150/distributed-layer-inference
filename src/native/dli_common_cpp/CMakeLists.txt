add_library(dli-common-cpp STATIC
    src/protocol.cpp
    src/metadata.cpp
    src/tensor.cpp
    src/metrics.cpp
    src/json_escape.cpp
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

target_link_libraries(test-metadata PRIVATE
    dli-common-cpp
)

add_executable(test-metadata
    tests/test_metadata.cpp
)

target_link_libraries(test-metadata PRIVATE
    dli-common-cpp)

enable_testing()
add_test(NAME test-protocol COMMAND test-protocol)
add_test(NAME test-metadata COMMAND test-metadata)