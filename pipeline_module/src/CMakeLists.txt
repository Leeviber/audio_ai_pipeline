
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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/aivad)

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

file(GLOB sherpa_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/sherpa_stt/*.cc
)
add_library(sherpa-onnx-core ${sherpa_sources})
target_link_libraries(sherpa-onnx-core onnxruntime kaldi-native-fbank-core ) 
 
set(ai_engine_src
  ${CMAKE_CURRENT_SOURCE_DIR}/ai_engine/ai_engine.cc
) 
add_library(ai_engine ${ai_engine_src})
target_link_libraries(ai_engine sherpa-onnx-core)
# target_include_directories(ai_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine)

add_subdirectory(speaker_id)
set(speake_diarization_core
  ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diarization/onnxModel/onnx_model.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diarization/clustering/clustering.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diarization/embedding/speaker_embedding.cc

)
add_library(speake_diarization ${speake_diarization_core})
target_link_libraries(speake_diarization PUBLIC onnxruntime)

target_link_libraries(ai_engine sherpa-onnx-core speaker speake_diarization)




set(CMAKE_CXX_STANDARD 17) # tts require c++ 17
set(tts
  ${CMAKE_CURRENT_SOURCE_DIR}/tts/piper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tts/phoneme_ids.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tts/phonemize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tts/tts.cc
)
add_library(tts_core ${tts})

target_link_libraries(tts_core onnxruntime)




target_include_directories(tts_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tts)
target_include_directories(tts_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tts/include)
target_include_directories(tts_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tts/include/espeak-ng)
target_include_directories(tts_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tts/include/utf8)

if(TARGET_SOC STREQUAL "rk3588")
  add_definitions(-DRK3588)

  if(TARGET_ARCH STREQUAL "arm32")
    link_directories(${CMAKE_CURRENT_LIST_DIR}/tts/lib_arm32/)
    target_link_directories(tts_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tts/lib_arm32/)

    set(ESPEAK_LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/tts/lib_arm32/libespeak-ng.so.1.1.51)
    install(
        FILES ${ESPEAK_LIB_PATH}
        DESTINATION lib
    )

  else()
    link_directories(${CMAKE_CURRENT_LIST_DIR}/tts/lib_aarch64/)
    target_link_directories(tts_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tts/lib_aarch64/)

    set(ESPEAK_LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/tts/lib_aarch64/libespeak-ng.so.1)
    install(
        FILES ${ESPEAK_LIB_PATH}
        DESTINATION lib
    )

  endif()

else()
  link_directories(${CMAKE_CURRENT_LIST_DIR}/tts/lib_x86/)
  target_link_directories(tts_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tts/lib_x86/)
endif()


target_link_libraries(tts_core
  espeak-ng
)



if(TARGET_SOC STREQUAL "rk3588")
  if(SPEAK_ID_NPU)
    add_definitions(-DUSE_NPU)
  endif()

  target_link_libraries(speaker PUBLIC ${RKNN_RT_LIB})


  add_executable(vad_stt ${CMAKE_CURRENT_SOURCE_DIR}/vad_stt_online.cc)
  target_link_libraries(vad_stt ai_engine alsa-core -pthread)

  add_executable(vad_sd ${CMAKE_CURRENT_SOURCE_DIR}/vad_speaker_diarization_online.cc)
  target_link_libraries(vad_sd ai_engine alsa-core -pthread)

  # add_executable(kws_stt ${CMAKE_CURRENT_SOURCE_DIR}/kws_stt_online.cc)
  # target_link_libraries(kws_stt sherpa-onnx-core kws-core alsa-core -pthread ${RKNN_RT_LIB})

  # add_executable(kws_stt_tts ${CMAKE_CURRENT_SOURCE_DIR}/kws_stt_tts_online.cc)
  # target_link_libraries(kws_stt_tts sherpa-onnx-core kws-core alsa-core -pthread ${RKNN_RT_LIB} tts_core)

  # add_executable(vad_stt_tts ${CMAKE_CURRENT_SOURCE_DIR}/vad_stt_tts.cc)
  # target_link_libraries(vad_stt_tts sherpa-onnx-core litevad alsa-core -pthread tts_core -lasound onnxruntime)
 
  # add_executable(id_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_id.cc)
  # target_link_libraries(id_demo PUBLIC alsa-core litevad -lasound -pthread speaker tts_core)
 
  # add_executable(diary_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diary.cc)
  # target_link_libraries(diary_demo PUBLIC alsa-core litevad speaker -lasound -pthread speake_diarization onnxruntime)
 
  # add_executable(diary_demo_simple ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diary_simple.cc)
  # target_link_libraries(diary_demo_simple PUBLIC alsa-core sherpa-onnx-core litevad speaker -pthread speake_diarization)


  # add_executable(track_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_track.cc)
  # target_link_libraries(track_demo PUBLIC alsa-core litevad -lasound -pthread speaker)


  # add_executable(change_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_change.cc)
  # target_link_libraries(change_demo PUBLIC alsa-core -lasound -pthread speaker speake_diarization)

  install(TARGETS sherpa-onnx-core DESTINATION lib)
  install(TARGETS speaker DESTINATION lib)
  install(TARGETS ai_engine DESTINATION lib)

  install(
    TARGETS
    # kws_stt
    vad_stt
    vad_sd
    # vad_stt_tts
    # id_demo
    # diary_demo
    # diary_demo_simple
    # track_demo
    # change_demo
    DESTINATION
    bin
  )

else()
  # target_link_libraries(sherpa-onnx-core kaldi-native-fbank-core sentencepiece-static)
  # target_link_libraries(sherpa-onnx-core onnxruntime)

  add_executable(vad_stt ${CMAKE_CURRENT_SOURCE_DIR}/vad_stt_online.cc)
  target_link_libraries(vad_stt ai_engine alsa-core -pthread asound)

 
  add_executable(vad_sd ${CMAKE_CURRENT_SOURCE_DIR}/vad_speaker_diarization_online.cc)
  target_link_libraries(vad_sd ai_engine alsa-core -pthread asound)

  # add_executable(vad_stt_tts ${CMAKE_CURRENT_SOURCE_DIR}/vad_stt_tts.cc)
  # target_link_libraries(vad_stt_tts sherpa-onnx-core litevad alsa-core -pthread tts_core -lespeak-ng -lasound onnxruntime)

  # add_executable(id_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_id.cc)
  # target_link_libraries(id_demo PUBLIC alsa-core litevad -lasound -pthread speaker tts_core)
 

  # add_executable(track_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_track.cc)
  # target_link_libraries(track_demo PUBLIC alsa-core litevad -lasound -pthread speaker)
 
  # add_executable(change_id_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_change_id_base.cc)
  # target_link_libraries(change_id_demo PUBLIC alsa-core litevad -lasound -pthread speaker)
 
  # add_executable(diary_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diary.cc)
  # target_link_libraries(diary_demo PUBLIC -pthread litevad speaker speake_diarization onnxruntime)

  # add_executable(diary_demo_simple ${CMAKE_CURRENT_SOURCE_DIR}/speaker_diary_simple.cc)
  # target_link_libraries(diary_demo_simple PUBLIC alsa-core sherpa-onnx-core litevad speaker -pthread speake_diarization)
 
  # add_executable(change_demo ${CMAKE_CURRENT_SOURCE_DIR}/speaker_change.cc)
  # target_link_libraries(change_demo PUBLIC alsa-core -lasound -pthread speaker speake_diarization)

  install(TARGETS sherpa-onnx-core DESTINATION lib)
  install(TARGETS speaker DESTINATION lib)
  install(TARGETS ai_engine DESTINATION lib)

  install(
    TARGETS
    vad_stt
    vad_sd
    # id_demo
    # diary_demo
    # diary_demo_simple
    # change_demo
    # change_id_demo
    # track_demo
    DESTINATION
    bin
  )
endif()
