cmake_minimum_required(VERSION 3.14)
project(wespeaker VERSION 0.1)

option(ONNX "whether to build with ONNX" ON)
option(SPEAK_ID_NPU "whether to build with ONNX" ON)

set(CMAKE_VERBOSE_MAKEFILE OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")

 
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
 
add_subdirectory(utils)
add_subdirectory(frontend)



set(speaker_srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/speaker/speaker_engine.cc)

 
list(APPEND speaker_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/speaker/onnx_speaker_model.cc
    )

add_definitions(-DUSE_ONNX)

if(SPEAK_ID_NPU)
  add_definitions(-DUSE_NPU)
  list(APPEND speaker_srcs ${CMAKE_CURRENT_SOURCE_DIR}/speaker/rknn_speaker_model.cc)
endif()

if(TARGET_SOC STREQUAL "rk3588")
  add_library(speaker STATIC ${speaker_srcs})
else()
  add_library(speaker SHARED ${speaker_srcs})
endif()

target_link_libraries(speaker PUBLIC frontend onnxruntime)

 