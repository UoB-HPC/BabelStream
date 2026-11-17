# Copyright (c) 2026 Stream HPC
#
# For full license terms please see the LICENSE file distributed with this
# source code

# Use
#
#   cmake -Bbuild -DMODEL=vulkan -DSLANGC_EXECUTABLE=foo/bar/bin/slangc
#
# Use -DSLANGC_EXECUTABLE to set the path to the slangc compiler binary
# Defaults to 'slangc' on the PATH, which should be available if the Vulkan SDK
# has been installed.
#
# The recommended way to install the VulkanSDK on Linux is by tarball, which
# also provides the VulkanHpp headers. This model has been tested building
# against VulkanSDK version 1.4.341.1 and later.
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection"
        "c++")

register_flag_optional(SLANGC_EXECUTABLE
        "Path to the slangc compiler executable"
        "slangc")

macro(setup)
    set(CMAKE_CXX_STANDARD 17)
    find_package(Vulkan REQUIRED)
    register_link_library(Vulkan::Vulkan)
endmacro()

macro(setup_target NAME)
    find_program(_slangc NAMES "${SLANGC_EXECUTABLE}" REQUIRED)

    set(_shader_src "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders.slang")
    set(_shader_dir "${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders")
    file(MAKE_DIRECTORY "${_shader_dir}/float")
    file(MAKE_DIRECTORY "${_shader_dir}/double")

    set(_outputs)

    foreach (_precision IN ITEMS float double)
        if (_precision STREQUAL "double")
            set(_use_double 1)
        else ()
            set(_use_double 0)
        endif ()

        # Embed SPIR-V shader into a C file compiled into the test binary, this is
        # less fragile for the user than reading from disk at runtime.
        set(_out "${_shader_dir}/${_precision}_shader.spv.h")
        add_custom_command(
            OUTPUT "${_out}"
            COMMAND "${_slangc}"
                    "${_shader_src}"
                    -target spirv
                    -profile spirv_1_5
                    -fvk-use-entrypoint-name
                    -entry "initShader"
                    -entry "copyShader"
                    -entry "mulShader"
                    -entry "addShader"
                    -entry "triadShader"
                    -entry "nstreamShader"
                    -entry "dotShader"
                    -DUSE_DOUBLE=${_use_double}
                    -source-embed-style u32
                    -source-embed-name "shader_${_precision}"
                    -o "${_out}"
            DEPENDS "${_shader_src}"
            VERBATIM
        )
        list(APPEND _outputs "${_out}")

    endforeach ()

    add_custom_target(${NAME}_slang_shaders DEPENDS ${_outputs})
    add_dependencies(${NAME} ${NAME}_slang_shaders)
    target_include_directories(${NAME} PRIVATE ${_shader_dir})
endmacro()
