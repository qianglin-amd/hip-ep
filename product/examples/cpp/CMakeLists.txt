##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.16)
project(hipep_cpp_examples CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_executable(chat_stream chat_stream.cpp)

# The only platform dependency is the sockets library; see http_stream.hpp for why the
# sample carries its own transport instead of requiring a package manager.
if(WIN32)
  target_link_libraries(chat_stream PRIVATE ws2_32)
endif()

if(MSVC)
  target_compile_options(chat_stream PRIVATE /W4)
  target_compile_definitions(chat_stream PRIVATE _CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN)
else()
  target_compile_options(chat_stream PRIVATE -Wall -Wextra)
endif()
