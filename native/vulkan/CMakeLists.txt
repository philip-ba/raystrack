cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(raystrack_vulkan_probe LANGUAGES CXX)
    include(CTest)
endif()

find_package(Vulkan 1.2 REQUIRED COMPONENTS glslc)

set(shader_header "${CMAKE_CURRENT_BINARY_DIR}/trace_spirv.inc")
add_custom_command(
    OUTPUT "${shader_header}"
    COMMAND Vulkan::glslc --target-env=vulkan1.2 -mfmt=c -O
            "${CMAKE_CURRENT_SOURCE_DIR}/trace.comp" -o "${shader_header}"
    DEPENDS trace.comp
    VERBATIM
)

add_library(raystrack_vulkan STATIC traversal.cpp "${shader_header}")
target_compile_features(raystrack_vulkan PUBLIC cxx_std_20)
target_include_directories(raystrack_vulkan
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(raystrack_vulkan PRIVATE Vulkan::Vulkan)
set_target_properties(raystrack_vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTING)
    add_executable(raystrack_vulkan_probe probe.cpp)
    target_link_libraries(raystrack_vulkan_probe PRIVATE raystrack_vulkan)
    add_test(NAME raystrack_vulkan_probe COMMAND raystrack_vulkan_probe)
    set_tests_properties(raystrack_vulkan_probe PROPERTIES FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")
endif()
