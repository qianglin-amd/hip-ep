##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ggml-hipep: out-of-tree ggml backend that runs llama.cpp graphs through the
# HIP EP compile & run path (docs/design/ggml-frontend.md). It must be built
# against the ggml headers and ggml-base library of the llama.cpp build that
# loads it (GGML_BACKEND_API_VERSION is checked at load time).

set(HIP_EP_GGML_SOURCE_DIR "" CACHE PATH
    "llama.cpp source tree providing ggml/include and ggml/src headers")
set(HIP_EP_GGML_BASE_LIBRARY "" CACHE FILEPATH
    "ggml-base import library (.lib) or shared object of that llama.cpp build")

if(NOT EXISTS "${HIP_EP_GGML_SOURCE_DIR}/ggml/src/ggml-backend-impl.h")
  message(FATAL_ERROR
    "HIP_EP_BUILD_GGML_BACKEND requires HIP_EP_GGML_SOURCE_DIR pointing at a "
    "llama.cpp checkout (missing ggml/src/ggml-backend-impl.h).")
endif()
if(NOT EXISTS "${HIP_EP_GGML_BASE_LIBRARY}")
  message(FATAL_ERROR
    "HIP_EP_BUILD_GGML_BACKEND requires HIP_EP_GGML_BASE_LIBRARY pointing at "
    "the ggml-base library of the llama.cpp build that loads the backend.")
endif()

find_package(hip CONFIG REQUIRED)

add_library(ggml-hipep SHARED
  src/ggml-hipep.cpp
  src/GgmlImporter.cpp
  src/HipEpEngine.cpp
)

add_dependencies(ggml-hipep
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_include_directories(ggml-hipep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/schemas
  ${CMAKE_SOURCE_DIR}/morphizen/morphizen-foundation/include
  ${CMAKE_SOURCE_DIR}/backend-mlir-compiler/custom-op-mlir/src
  ${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/include
)
# Appended last so ggml's generic header names cannot shadow project headers.
target_include_directories(ggml-hipep SYSTEM PRIVATE
  ${HIP_EP_GGML_SOURCE_DIR}/ggml/include
  ${HIP_EP_GGML_SOURCE_DIR}/ggml/src
)

target_compile_definitions(ggml-hipep PRIVATE GGML_SHARED)
if(WIN32)
  target_compile_definitions(ggml-hipep PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_link_libraries(ggml-hipep PRIVATE
  morphizen-custom-op-mlir
  morphizen::morphizen-core-static
  LibHipCompiler
  HipSupport
  hip::host
  "${HIP_EP_GGML_BASE_LIBRARY}"
)
hip_link_rocmlir_closure(ggml-hipep)

if(NOT WIN32)
  target_link_options(ggml-hipep PRIVATE "LINKER:--exclude-libs,ALL")
  set_target_properties(ggml-hipep PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# The JIT and the native side-effect ops load custom_kernels_<arch> from this
# module's directory.
foreach(_arch IN LISTS HIP_ARCHITECTURES)
  if(TARGET custom_kernels_${_arch})
    add_dependencies(ggml-hipep custom_kernels_${_arch})
    add_custom_command(TARGET ggml-hipep POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:custom_kernels_${_arch}>"
        "$<TARGET_FILE_DIR:ggml-hipep>"
      VERBATIM)
  endif()
endforeach()

install(TARGETS ggml-hipep
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
)

# GPU-free importer check: builds Llama-style graphs and writes each segment's
# MLIR (feed the files to `hip-mlir-opt --hipdnn-pipeline`).
add_executable(ggml-import-test
  tests/ggml-import-test.cpp
  src/GgmlImporter.cpp
)
target_include_directories(ggml-import-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(ggml-import-test SYSTEM PRIVATE
  ${HIP_EP_GGML_SOURCE_DIR}/ggml/include
  ${HIP_EP_GGML_SOURCE_DIR}/ggml/src
)
target_compile_definitions(ggml-import-test PRIVATE GGML_SHARED)
target_link_libraries(ggml-import-test PRIVATE "${HIP_EP_GGML_BASE_LIBRARY}")

# CPU-vs-HIPEP graph comparison; needs the ggml (registry) and ggml-cpu
# libraries that sit next to ggml-base in the llama.cpp build.
get_filename_component(_ggml_lib_dir "${HIP_EP_GGML_BASE_LIBRARY}" DIRECTORY)
find_library(HIP_EP_GGML_LIBRARY NAMES ggml PATHS "${_ggml_lib_dir}"
             NO_DEFAULT_PATH)
find_library(HIP_EP_GGML_CPU_LIBRARY NAMES ggml-cpu PATHS "${_ggml_lib_dir}"
             NO_DEFAULT_PATH)
if(HIP_EP_GGML_LIBRARY AND HIP_EP_GGML_CPU_LIBRARY)
  add_executable(ggml-graph-test tests/ggml-graph-test.cpp)
  target_include_directories(ggml-graph-test SYSTEM PRIVATE
    ${HIP_EP_GGML_SOURCE_DIR}/ggml/include
  )
  target_compile_definitions(ggml-graph-test PRIVATE
    GGML_SHARED GGML_BACKEND_SHARED)
  target_link_libraries(ggml-graph-test PRIVATE
    "${HIP_EP_GGML_BASE_LIBRARY}"
    "${HIP_EP_GGML_CPU_LIBRARY}"
    "${HIP_EP_GGML_LIBRARY}"
  )
endif()
