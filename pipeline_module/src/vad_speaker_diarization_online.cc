#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <math.h>

#include "alsa_cq_buffer.h"      // ALSA
#include "ai_engine/ai_engine.h" //STT & Speaker diarization

int main()
{

  //// Init ALSA and circular buffer////
  online_params params;

  int32_t ret = init_online_audio(&params);
  if (ret < 0)
  {
    fprintf(stderr, "Error init_kws \n");
    return -1;
  }
  snd_pcm_t *capture_handle;
  const char *device_name = "plughw:2,0"; // using arecord -l to checkout the alsa device name
  ret = audio_CQ_init(device_name, params.sample_rate, &params, capture_handle);
  //////////////////////////////////////

  //// Init Sherpa STT module //////////
  bool using_whisper = false;
  STTEngine stt_interface(using_whisper);
   //////////////////////////////////////

  /////////// Init chunk VAD //////////////////

  int vad_frame_ms = 96; // audio chunk length(ms) for VAD detect, (32,64,96), large is more accuray with more latency
  std::string vad_path = "./bin/silero_vad.onnx";
  float min_silence_duration = 0.01;
  float vad_threshold = 0.65;

  VADChunk vad_chunk_stt(vad_path, vad_frame_ms,vad_threshold, min_silence_duration);
 
  printf("start\n");

  std::vector<std::string> model_paths;
#ifdef USE_NPU
  std::string rknn_model_path = "./bin/Id1_resnet34_LM_main_part.rknn";
  std::string onnx_model_path = "./bin/Id2_resnet34_LM_post.onnx";
  model_paths.push_back(rknn_model_path);
  model_paths.push_back(onnx_model_path);

#else
  std::string onnx_model_path = "./bin/voxceleb_resnet34_LM.onnx";
  // std::string onnx_model_path = "./bin/voxceleb_CAM++_LM.onnx";
  model_paths.push_back(onnx_model_path);

#endif /
  int embedding_size = 256;

  // Init speaker id
  SpeakerID speaker_id(model_paths, embedding_size);
  
  //Init cluster
  Cluster cluster;


  printf("Init success\n");
  //// Main loop for audio real time process //////
  //////////////////////////////////////

  while (params.is_running)
  {

    while (true)
    {
      int len = audio_CQ_get(&params, vad_frame_ms, 0); // The audio windows for ai vad is 64ms
      if (len > 0)
      {
        vad_chunk_stt.PushAudioChunk(params.audio.pcmf32_new);
        break;
      }
    }

    bool isUpdate=vad_chunk_stt.SpeakerDiarization(&stt_interface, &speaker_id, &cluster);
    if(isUpdate)
    {
      vad_chunk_stt.printAllDiarizations(true);  // true mean the result will print in sequence
                                                // false will print in group
    }
  }

  ///////////////////////////////////////////
  return 0;
}
