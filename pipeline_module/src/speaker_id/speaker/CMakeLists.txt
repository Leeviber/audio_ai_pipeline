set(speaker_srcs
  speaker_engine.cc)

add_definitions(-DUSE_ONNX)

if(NOT ONNX AND NOT BPU)
  message(FATAL_ERROR "Please build with ONNX or BPU!")
endif()
list(APPEND speaker_srcs onnx_speaker_model.cc)
list(APPEND speaker_srcs rknn_speaker_model.cc)
 

add_library(speaker STATIC ${speaker_srcs})
target_link_libraries(speaker PUBLIC frontend)

target_link_libraries(speaker PUBLIC onnxruntime)
 

