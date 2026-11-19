##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# oga-chat runs ONNX Runtime GenAI model packages on the HIP EP plugin. It is
# built when HIP_EP_OGA_DIR points at an ONNX Runtime GenAI distribution: a
# release zip (include/, lib/) or the extracted Microsoft.ML.OnnxRuntimeGenAI
# NuGet package (build/native/include/, runtimes/<rid>/native/).
set(HIP_EP_OGA_DIR "" CACHE PATH "ONNX Runtime GenAI distribution for oga-chat")
if(NOT HIP_EP_OGA_DIR)
  return()
endif()
if(NOT TARGET onnxruntime::onnxruntime)
  message(STATUS "oga-chat: onnxruntime not found, skipping")
  return()
endif()

if(WIN32)
  set(_oga_rid win-x64)
else()
  set(_oga_rid linux-x64)
endif()
find_path(OGA_INCLUDE_DIR ort_genai.h
  HINTS ${HIP_EP_OGA_DIR}
  PATH_SUFFIXES include build/native/include
  NO_DEFAULT_PATH)
find_library(OGA_LIBRARY onnxruntime-genai
  HINTS ${HIP_EP_OGA_DIR}
  PATH_SUFFIXES lib runtimes/${_oga_rid}/native
  NO_DEFAULT_PATH)
if(NOT OGA_INCLUDE_DIR OR NOT OGA_LIBRARY)
  message(FATAL_ERROR "HIP_EP_OGA_DIR=${HIP_EP_OGA_DIR} has no ort_genai.h or "
                      "onnxruntime-genai library")
endif()
get_filename_component(_oga_lib_dir ${OGA_LIBRARY} DIRECTORY)
if(WIN32)
  set(OGA_RUNTIME ${_oga_lib_dir}/onnxruntime-genai.dll)
else()
  set(OGA_RUNTIME ${OGA_LIBRARY})
endif()
message(STATUS "oga-chat: ONNX Runtime GenAI from ${HIP_EP_OGA_DIR}")

add_executable(oga-chat oga-chat.cpp)
# ort_genai.h uses std::span.
target_compile_features(oga-chat PRIVATE cxx_std_20)
target_include_directories(oga-chat PRIVATE ${OGA_INCLUDE_DIR})
target_link_libraries(oga-chat PRIVATE ${OGA_LIBRARY})

# The GenAI runtime goes next to the tool, as onnxruntime.dll does for
# hip-onnx-runner.
add_custom_command(TARGET oga-chat POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OGA_RUNTIME}
    $<TARGET_FILE_DIR:oga-chat>)

install(TARGETS oga-chat RUNTIME DESTINATION bin)
install(FILES ${OGA_RUNTIME} DESTINATION bin)
