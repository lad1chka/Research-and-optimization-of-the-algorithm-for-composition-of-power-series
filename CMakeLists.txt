cmake_minimum_required(VERSION 3.16)
project(pscomp
        VERSION 0.1.0
        DESCRIPTION "Power-series composition algorithms (naive, Brent-Kung, Kinoshita-Li)"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Default build type" FORCE)
endif()

option(PSCOMP_BUILD_TESTS      "Build pscomp test suite"      ON)
option(PSCOMP_BUILD_BENCHMARKS "Build pscomp benchmarks"      ON)
option(PSCOMP_NATIVE_ARCH      "Pass -march=native to compiler" ON)

add_library(pscomp INTERFACE)
target_include_directories(pscomp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(pscomp INTERFACE cxx_std_17)

# Build-only object library that turns the warning flags into a real consumer
# of the public headers. Helps catch issues during library development without
# imposing flags on downstream users.
add_library(pscomp_dev_check OBJECT src/dev_check.cpp)
target_link_libraries(pscomp_dev_check PRIVATE pscomp)
if(MSVC)
    target_compile_options(pscomp_dev_check PRIVATE /W4 /permissive-)
else()
    target_compile_options(pscomp_dev_check PRIVATE -Wall -Wextra -Wpedantic)
    if(PSCOMP_NATIVE_ARCH)
        target_compile_options(pscomp_dev_check PRIVATE -march=native)
    endif()
endif()

include(FetchContent)

if(PSCOMP_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_subdirectory(tests)
endif()

if(PSCOMP_BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS  OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)

    add_subdirectory(benchmarks)
endif()
