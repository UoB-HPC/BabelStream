
# MUSA backend configuration
# Use:
#    cmake -Bbuild -H. -DMODEL=musa -DMUSA_COMPILER=/usr/local/musa/bin/mcc \
#          -DMUSA_ARCH=mp_31 -DCXX_EXTRA_FLAGS="-L/usr/local/musa/lib"
#    cmake --build build
# Run:
#     export MUSA_USERQ=1
#    ./build/musa-stream
register_flag_optional(MEM "Device memory mode:
        DEFAULT   - allocate host and device memory pointers.
        MANAGED   - use MUSA Managed Memory.
        PAGEFAULT - shared memory, only host pointers allocated."
        "DEFAULT")

register_flag_required(MUSA_COMPILER
        "Path to the MUSA mcc compiler")

register_flag_required(MUSA_ARCH
        "Mthreads architecture, will be passed in via `--offload-arch=` (e.g `mp_31`) for mcc")

register_flag_optional(MUSA_EXTRA_FLAGS
        "Additional MUSA flags passed to mcc, this is appended after `MUSA_ARCH`"
        "")


macro(setup)
	message(STATUS "Configuring MUSA backend: ${IMPL_SOURCES}")
	# load MUSA CMake module
	list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)
	find_package(MUSA REQUIRED)

	set(MUSA_VERBOSE_BUILD ON)
	set(MUSA_MCC_FLAGS
		"--offload-arch=${MUSA_ARCH} "
		${MUSA_EXTRA_FLAGS}
	)
	musa_include_directories(${CMAKE_SOURCE_DIR}/src)
	musa_compile(MUSAOBJ ${IMPL_SOURCES})

	# create the interface library for musa objects
    add_library(musa_objs INTERFACE)
    target_sources(musa_objs INTERFACE ${MUSAOBJ})
    target_link_libraries(musa_objs INTERFACE musart)

    register_link_library(musa_objs)

endmacro()

