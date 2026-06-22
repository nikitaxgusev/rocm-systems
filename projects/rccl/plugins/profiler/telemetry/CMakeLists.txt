# Telemetry profiler plugin: per-collective RCCL network telemetry.
set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/print_event.cc
)

# Create shared library
add_library(nccl-profiler-telemetry SHARED ${SRC_FILES})

# Include dirs: local nccl profiler headers only (plugin is self-contained).
target_include_directories(nccl-profiler-telemetry PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl
)

set_target_properties(nccl-profiler-telemetry PROPERTIES
    OUTPUT_NAME "nccl-profiler-telemetry"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

add_custom_command(TARGET nccl-profiler-telemetry POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/test/unit/plugins
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/lib/libnccl-profiler-telemetry.so ${CMAKE_BINARY_DIR}/test/unit/plugins
)

add_custom_target(clean-profiler-telemetry-lib
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/lib/libnccl-profiler-telemetry.so
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/test/unit/plugins/libnccl-profiler-telemetry.so
    COMMENT "Cleaning libnccl-profiler-telemetry.so"
)
