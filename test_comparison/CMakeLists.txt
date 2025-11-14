# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 The ModelVirtualizer Authors
cmake_minimum_required(VERSION 3.16)
project(VirtualizedComparison)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable verbose build output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find TensorFlow Lite library based on BUILD file analysis
# The library should be built using: cd ../tensorflow_src && bazel build //tensorflow/lite:tensorflowlite
find_library(TENSORFLOW_LITE_LIB 
    NAMES tensorflowlite libtensorflowlite.so
    PATHS 
        ${CMAKE_SOURCE_DIR}/../bazel-bin/tensorflow/lite
        ${CMAKE_SOURCE_DIR}/../tensorflow_src/bazel-bin/tensorflow/lite  
        ${CMAKE_SOURCE_DIR}/../bazel-out/k8-opt/bin/tensorflow/lite
        ${CMAKE_SOURCE_DIR}/../tensorflow_src/bazel-out/k8-opt/bin/tensorflow/lite
        # Additional possible paths
        ${CMAKE_SOURCE_DIR}/../tensorflow_src/bazel-out/k8-fastbuild/bin/tensorflow/lite
        ${CMAKE_SOURCE_DIR}/../tensorflow_src/bazel-out/k8-dbg/bin/tensorflow/lite
    NO_DEFAULT_PATH
)

if(NOT TENSORFLOW_LITE_LIB)
    message(FATAL_ERROR "TensorFlow Lite library not found! Please build it first using:
    cd ../tensorflow_src && bazel build //tensorflow/lite:tensorflowlite")
endif()

message(STATUS "Found TensorFlow Lite library: ${TENSORFLOW_LITE_LIB}")

# Common source files
set(COMMON_SOURCES
    comparison_utils.cc
    memory_monitor.cc
    peak_memory_tracker.cc
)

# Original combined test executable (kept for compatibility)
add_executable(test_virtualized_interpreter
    test_virtualized_interpreter.cc
    ${COMMON_SOURCES}
)

# New: Virtualized model only test
add_executable(test_virtualized_only
    test_virtualized_only.cc
    ${COMMON_SOURCES}
)

# New: Original model only test
add_executable(test_original_only
    test_original_only.cc
    ${COMMON_SOURCES}
)

# Include directories (shared by all targets)
set(COMMON_INCLUDES
    ${CMAKE_SOURCE_DIR}/../tensorflow_src
    ${CMAKE_SOURCE_DIR}/../tensorflow_src/tensorflow/lite
    ${CMAKE_SOURCE_DIR}/../tensorflow_src/bazel-tensorflow_src/external/flatbuffers/include
    ${CMAKE_SOURCE_DIR}/../tensorflow_src/bazel-tensorflow_src/external/com_google_absl
)

target_include_directories(test_virtualized_interpreter PRIVATE ${COMMON_INCLUDES})
target_include_directories(test_virtualized_only PRIVATE ${COMMON_INCLUDES})
target_include_directories(test_original_only PRIVATE ${COMMON_INCLUDES})

# Link libraries
target_link_libraries(test_virtualized_interpreter ${TENSORFLOW_LITE_LIB})
target_link_libraries(test_virtualized_only ${TENSORFLOW_LITE_LIB})
target_link_libraries(test_original_only ${TENSORFLOW_LITE_LIB})

# Platform-specific linking and compiler settings
if(WIN32)
    # Windows-specific settings
    target_link_libraries(test_virtualized_interpreter psapi)
    target_link_libraries(test_virtualized_only psapi)
    target_link_libraries(test_original_only psapi)

    target_compile_definitions(test_virtualized_interpreter PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_definitions(test_virtualized_only PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_definitions(test_original_only PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
elseif(ANDROID)
    # Android-specific settings (bionic provides pthread/dl; no extra linking needed)
    # Enable Android logging if needed:
    # target_link_libraries(test_virtualized_interpreter log)
    # target_link_libraries(test_virtualized_only log)
    # target_link_libraries(test_original_only log)
elseif(UNIX AND NOT ANDROID)
    # Linux/Unix (non-Android) specific settings
    target_link_libraries(test_virtualized_interpreter pthread dl)
    target_link_libraries(test_virtualized_only pthread dl)
    target_link_libraries(test_original_only pthread dl)

    set_target_properties(test_virtualized_interpreter PROPERTIES
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH "\$ORIGIN"
    )
    set_target_properties(test_virtualized_only PROPERTIES
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH "\$ORIGIN"
    )
    set_target_properties(test_original_only PROPERTIES
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH "\$ORIGIN"
    )
endif()

# Compiler-specific warnings and optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(COMMON_COMPILE_OPTIONS
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
        $<$<CONFIG:Debug>:-g -O0>
    )

    target_compile_options(test_virtualized_interpreter PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_compile_options(test_virtualized_only PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_compile_options(test_original_only PRIVATE ${COMMON_COMPILE_OPTIONS})
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

# Create a custom target to show build instructions
add_custom_target(instructions
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Build Instructions ==="
    COMMAND ${CMAKE_COMMAND} -E echo "1. First, build TensorFlow Lite library:"
    COMMAND ${CMAKE_COMMAND} -E echo "   cd ../tensorflow_src"
    COMMAND ${CMAKE_COMMAND} -E echo "   bazel build //tensorflow/lite:tensorflowlite"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "2. Then build this comparison test:"
    COMMAND ${CMAKE_COMMAND} -E echo "   mkdir build && cd build"
    COMMAND ${CMAKE_COMMAND} -E echo "   cmake .."
    COMMAND ${CMAKE_COMMAND} -E echo "   make"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "3. Run the test (from project root):"
    COMMAND ${CMAKE_COMMAND} -E echo "   cd .."
    COMMAND ${CMAKE_COMMAND} -E echo "   ./build/test_virtualized_interpreter"
    COMMAND ${CMAKE_COMMAND} -E echo ""
)

# Create a test target for easier running
enable_testing()
add_test(NAME virtualized_comparison_test 
        COMMAND test_virtualized_interpreter
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../)

# Print final message
message(STATUS "Configuration completed successfully!")
message(STATUS "Run 'make instructions' to see build instructions")
