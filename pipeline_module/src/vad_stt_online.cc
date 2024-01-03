#ifndef VAD_STT_ONLINE

#include <stdio.h>
#include <chrono>
#include <string>
#include <vector>

#include "vad_stt_online.h"

#include "sherpa_stt/sherpa_stt.h" //STT

#include "ai_vad.h" // ai based vad

#include "alsa_cq_buffer.h" // ALSA

#define SAMPLE_RATE 16000

int32_t init_online_audio(online_params *params)
{

  params->is_running = true;

  online_audio audio_buffer;
  audio_buffer.pcmf32_new = std::vector<float>(params->n_samples_30s, 0.0f);
  audio_buffer.CQ_buffer.resize(SAMPLE_RATE * 30);
  params->audio = audio_buffer;
  float value;

  return 0;
}

int main()
{

  ///////////// Init online audio ////////////////
  online_params params;
  int32_t ret = init_online_audio(&params);
  if (ret < 0)
  {
    fprintf(stderr, "Error init_kws \n");
    return -1;
  }
  bool using_vad = false;

  bool trigger_stt = false;
  //////////////////////////////////////

  //// Init ALSA and circular buffer////
  snd_pcm_t *capture_handle;
  const char *device_name = "plughw:2,0"; // using arecord -l to checkout the alsa device name
  ret = audio_CQ_init(device_name, SAMPLE_RATE, &params, capture_handle);
  //////////////////////////////////////

  //// Init Sherpa STT module //////////

  STTInterface stt_interface;

  bool using_whisper = false;
  stt_interface.init_stt(using_whisper);

  //////////////////////////////////////

  /////////// Init VAD //////////////////
  int vad_begin = 0;
  int vad_end = 0;

  int32_t vad_activate_sample = (SAMPLE_RATE * 200) / 1000;
  int32_t vad_silence_sample = (SAMPLE_RATE * 0) / 1000;
  int vad_frame_ms = 96;

  std::string path = "./bin/silero_vad.onnx";

  VadIterator ai_vad(path);

  //////////////////////////////////////

  //// Main loop for audio real time process //////
  while (params.is_running)
  {
    fprintf(stderr, "Started\n");

    while (true)
    {
      int len = audio_CQ_get(&params, vad_frame_ms, 0); // The audio windows for ai vad is 64ms
      if (len > 0)
      {

        int32_t vad_state = ai_vad.predict(params.audio.pcmf32_new);

        if (vad_state == 2)
        {
          printf("begin\n");
          printf("params->CQ_audio_exit%d\n", params.CQ_audio_exit);
          vad_begin = params.CQ_audio_exit - vad_activate_sample;
        }
        if (vad_state == 3)
        {
          printf("end\n");
          printf("params->CQ_audio_exit%d\n", params.CQ_audio_exit);
          vad_end = params.CQ_audio_exit - vad_silence_sample;

          len = audio_CQ_view(&params, vad_begin, vad_end);
          if (len > 0)
          {
            break;
          }
        }
      }
    }

    const auto begin = std::chrono::steady_clock::now();
    std::string text = stt_interface.perform_stt(params.audio.pcmf32_new);
    const auto end = std::chrono::steady_clock::now();
    fprintf(stderr, "TEXT: %s\n----\n", text.c_str());
    int len = audio_CQ_clear(&params); // clear the audio buffer when processing the last audio chunk

    float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count() /
        1000.;

    float duration = (float)params.audio.pcmf32_new.size() / (float)SAMPLE_RATE;

    float rtf = elapsed_seconds / duration;
    fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
            elapsed_seconds, duration, rtf);
  }

  ///////////////////////////////////////////
  return 0;
}
#endif
