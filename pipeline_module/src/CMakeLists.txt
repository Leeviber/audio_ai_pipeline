
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fft)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kws)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/alsa)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/litevad/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tts)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tts/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tts/include/espeak-ng)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tts/include/utf8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/speaker_diarization/clustering)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/speaker_diarization/onnxModel)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/speaker_diarization/frontend)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/speaker_diarization/segmentation)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/speaker_diarization/embedding)

set(LITEVAD_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/signal_processing/energy.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/signal_processing/division_operations.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/signal_processing/get_scaling_square.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/signal_processing/resample.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/signal_processing/resample_48khz.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/signal_processing/resample_by_2.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/signal_processing/resample_by_2_internal.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/signal_processing/resample_fractional.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/vad/vad_core.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/vad/vad_filterbank.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/vad/vad_gmm.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/vad/vad_sp.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/vad/webrtc_vad.c
  ${CMAKE_CURRENT_SOURCE_DIR}/litevad/src/litevad_api.c
)

add_library(litevad STATIC ${LITEVAD_SRC})
target_include_directories(litevad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/litevad/include)

if(TARGET_SOC STREQUAL "rk3588")
  # rknn api
  if(TARGET_ARCH STREQUAL "arm32")
    set(LIB_ARCH armhf)
    set(RKNN_API_PATH ${Cmin_num_samplesMAKE_CURRENT_SOURCE_DIR}/librknn_api)
    set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
    include_directories(${RKNN_API_PATH}/include)
  else()
    set(LIB_ARCH aarch64)
    set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
    set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
    include_directories(${RKNN_API_PATH}/include)
  endif()

  # kws depend on rknn and rk3588
  set(kws_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/kws/kws.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kws/kws_engine.cpp)
  add_library(kws-core ${kws_sources})

  target_link_libraries(kws-core kiss_fft)
endif()

set(kiss_fft
  ${CMAKE_CURRENT_SOURCE_DIR}/fft/kiss_fft.c
  ${CMAKE_CURRENT_SOURCE_DIR}/fft/kiss_fftr.c
)
add_library(kiss_fft ${kiss_fft})

set(alsa_cq_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/alsa/alsa_cq_buffer.cc
)
add_library(alsa-core ${alsa_cq_sources})
target_link_libraries(alsa-core -L$ENV{SHERPA_ONNX_ALSA_LIB_DIR} -lasound)

set(sherpa_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/cat.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/context-graph.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/features.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/file-utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/hypothesis.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/offline-lm-config.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/offline-lm.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/offline-model-config.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/offline-recognizer-impl.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/offline-recognizer.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/offline-rnn-lm.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/offline-stream.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/offline-transducer-greedy-search-decoder.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/offline-transducer-model-config.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/offline-transducer-model.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/offline-transducer-modified-beam-search-decoder.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/onnx-utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/session.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/packed-sequence.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/pad-sequence.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/parse-options.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/provider.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/resample.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/slice.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/stack.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/symbol-table.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/text-utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/transpose.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/unbind.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/utils.cc

)
add_library(sherpa-onnx-core ${sherpa_sources})

set(CMAKE_CXX_STANDARD 17) # stt require c++ 17

set(stt
  ${CMAKE_CURRENT_SOURCE_DIR}/tts/piper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tts/phoneme_ids.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tts/phonemize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tts/tts.cc
)
add_library(stt_core ${stt})

target_link_libraries(stt_core onnxruntime)

set(speake_diarization_core
  ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diarization/onnxModel/onnx_model.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diarization/clustering/clustering.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diarization/embedding/speaker_embedding.cc

)
add_library(speake_diarization ${speake_diarization_core})
target_link_libraries(speake_diarization PUBLIC onnxruntime)


target_include_directories(stt_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tts)
target_include_directories(stt_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tts/include)
target_include_directories(stt_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tts/include/espeak-ng)
target_include_directories(stt_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tts/include/utf8)

if(TARGET_SOC STREQUAL "rk3588")
  add_definitions(-DRK3588)

  if(TARGET_ARCH STREQUAL "arm32")
    link_directories(${CMAKE_CURRENT_LIST_DIR}/tts/lib_arm32/)
    target_link_directories(stt_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tts/lib_arm32/)

    set(ESPEAK_LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/tts/lib_arm32/libespeak-ng.so.1.1.51)
    install(
        FILES ${ESPEAK_LIB_PATH}
        DESTINATION lib
    )

  else()
    link_directories(${CMAKE_CURRENT_LIST_DIR}/tts/lib_aarch64/)
    target_link_directories(stt_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tts/lib_aarch64/)

    set(ESPEAK_LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/tts/lib_aarch64/libespeak-ng.so.1)
    install(
        FILES ${ESPEAK_LIB_PATH}
        DESTINATION lib
    )

  endif()

else()
  link_directories(${CMAKE_CURRENT_LIST_DIR}/tts/lib_x86/)
  target_link_directories(stt_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tts/lib_x86/)
endif()

# if(TARGET_SOC STREQUAL "rk3588")
# if(TARGET_ARCH STREQUAL "arm32")
# target_link_directories(stt_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tts/lib_arm32)
# else()
# target_link_directories(stt_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tts/lib_aarch64)
# endif()
# else()
# target_link_directories(stt_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tts/lib_x86)
# endif()
target_link_libraries(stt_core
  espeak-ng
)

add_subdirectory(speaker_id)
 
if(TARGET_SOC STREQUAL "rk3588")
  
  target_link_libraries(speaker PUBLIC ${RKNN_RT_LIB})
  target_link_libraries(sherpa-onnx-core kaldi-native-fbank-core sentencepiece-static) 
  target_link_libraries(sherpa-onnx-core onnxruntime)

  add_executable(vad_stt ${CMAKE_CURRENT_SOURCE_DIR}/vad_stt_online.cc)
  target_link_libraries(vad_stt sherpa-onnx-core litevad alsa-core -pthread)

  add_executable(kws_stt ${CMAKE_CURRENT_SOURCE_DIR}/kws_stt_online.cc)
  target_link_libraries(kws_stt sherpa-onnx-core kws-core alsa-core -pthread ${RKNN_RT_LIB})

  add_executable(kws_stt_tts ${CMAKE_CURRENT_SOURCE_DIR}/kws_stt_tts_online.cc)
  target_link_libraries(kws_stt_tts sherpa-onnx-core kws-core alsa-core -pthread ${RKNN_RT_LIB} stt_core)

  add_executable(vad_stt_tts ${CMAKE_CURRENT_SOURCE_DIR}/vad_stt_tts.cc)
  target_link_libraries(vad_stt_tts sherpa-onnx-core litevad alsa-core -pthread stt_core -lasound onnxruntime)
 
  # add_executable(id_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_id.cc)
  # target_link_libraries(id_demo PUBLIC alsa-core litevad -lasound -pthread speaker stt_core)
 
  # add_executable(diary_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diary.cc)
  # target_link_libraries(diary_demo PUBLIC alsa-core litevad -lasound -pthread speake_diarization)
 
  # add_executable(diary_demo_simple ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diary_simple.cc)
  # target_link_libraries(diary_demo_simple PUBLIC alsa-core litevad -lasound -pthread speake_diarization)

  if(SPEAK_ID_NPU)
    add_definitions(-DUSE_NPU)
  endif()

  add_executable(track_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_track.cc)
  target_link_libraries(track_demo PUBLIC alsa-core litevad -lasound -pthread speaker)

  install(TARGETS sherpa-onnx-core DESTINATION lib)
  install(TARGETS speaker DESTINATION lib)
 
  install(
    TARGETS
    kws_stt
    vad_stt
    vad_stt_tts
    # id_demo
    # diary_demo
    # diary_demo_simple
    track_demo
    DESTINATION
    bin
  )

else()
  target_link_libraries(sherpa-onnx-core kaldi-native-fbank-core sentencepiece-static)
  target_link_libraries(sherpa-onnx-core onnxruntime)

  add_executable(vad_stt_tts ${CMAKE_CURRENT_SOURCE_DIR}/vad_stt_tts.cc)
  target_link_libraries(vad_stt_tts sherpa-onnx-core litevad alsa-core -pthread stt_core -lespeak-ng -lasound onnxruntime)

  # add_executable(id_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_id.cc)
  # target_link_libraries(id_demo PUBLIC alsa-core litevad -lasound -pthread speaker stt_core)
 

  add_executable(track_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_track.cc)
  target_link_libraries(track_demo PUBLIC alsa-core litevad -lasound -pthread speaker)
 
  # add_executable(diary_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diary.cc)
  # target_link_libraries(diary_demo PUBLIC -pthread litevad speaker speake_diarization onnxruntime)

  # add_executable(diary_demo_simple ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diary_simple.cc)
  # target_link_libraries(diary_demo_simple PUBLIC alsa-core litevad speaker -pthread speake_diarization)
 
  install(TARGETS sherpa-onnx-core DESTINATION lib)
  install(TARGETS speaker DESTINATION lib)

  install(
    TARGETS
    vad_stt_tts
    # id_demo
    # diary_demo_simple
    track_demo
    DESTINATION
    bin
  )
endif()