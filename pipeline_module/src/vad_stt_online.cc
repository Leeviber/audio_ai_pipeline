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
  bool using_Whisper = false;  // English only
  bool using_Chinese = true;   // Not support whipser
  STTEngine stt_interface(using_Whisper,using_Chinese);
  //////////////////////////////////////

  /////////// Init chunk VAD //////////////////

  int vad_frame_ms = 96; // audio chunk length(ms) for VAD detect, (32,64,96), large is more accuray with more latency
  std::string vad_path = "./bin/silero_vad.onnx";
  float min_silence_duration = 0.01;
  float vad_threshold = 0.65;
  VADChunk vad_chunk_stt(vad_path, vad_frame_ms,vad_threshold, min_silence_duration);
 
  //// Main loop for audio real time process //////
  //////////////////////////////////////


  printf("Init success\n");

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

    vad_chunk_stt.STT(&stt_interface);
  }

  ///////////////////////////////////////////
  return 0;
}
