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

#include "alsa_cq_buffer.h"        // ALSA
#include "ai_engine/ai_engine.h" //STT

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

  STTEngine stt_interface;
  bool using_whisper = true;
  stt_interface.init_stt(using_whisper);
  //////////////////////////////////////

  /////////// Init chunk VAD //////////////////

  int vad_frame_ms = 96; // audio chunk length(ms) for VAD detect, (32,64,96), large is more accuray with more latency
  std::string vad_path = "./bin/silero_vad.onnx";

  VADChunk vad_chunk_stt;
  vad_chunk_stt.InitVAD(vad_path, vad_frame_ms);

  printf("start\n");
  //// Main loop for audio real time process //////
  //////////////////////////////////////

    std::vector<std::string> model_paths;
#ifdef USE_NPU
    std::string rknn_model_path = "./bin/Id1_resnet34_LM_main_part.rknn";
    std::string onnx_model_path = "./bin/Id2_resnet34_LM_post.onnx";
    model_paths.push_back(rknn_model_path);
    model_paths.push_back(onnx_model_path); 

#else
    std::string onnx_model_path = "./bin/voxceleb_resnet34_LM.onnx";
    // std::string onnx_model_path = "./bin/voxceleb_CAM++_LM.onnx";
    printf("here\n");
    model_paths.push_back(onnx_model_path);

#endif  // 其他参数
  int feat_dim = 80;
  int sample_rate = 16000;
  int embedding_size = 256;
  int SamplesPerChunk = 32000;

  // 创建 SpeakerEngine 对象
  SpeakerID speaker_id(model_paths, feat_dim, sample_rate, embedding_size, SamplesPerChunk);
  Cluster cluster;
  printf("success\n");


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

    // vad_chunk_stt.STT(stt_interface);
    vad_chunk_stt.ExtractId(speaker_id,cluster);
    // vad_chunk_stt.process_embedding(speaker_id);

  }

  ///////////////////////////////////////////
  return 0;
}
