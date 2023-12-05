#ifndef VAD_STT_TTS_ONLINE

#include <stdio.h>
#include <chrono>
#include <string>
#include <vector>
#include "sherpa_stt/offline-recognizer.h"
#include "sherpa_stt/offline-model-config.h"
#include "sherpa_stt/offline-transducer-model-config.h"

#include "litevad_api.h"    // vad
#include "alsa_cq_buffer.h" // ALSA
#include "vad_stt_tts.h"

#include "tts.h"    //tts
#include "ai_vad.h" // vad

#include <cctype> // 包含 tolower() 函数的头文件
#include <algorithm>

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

  std::string tokens = "./bin/tokens.txt";
  std::string encoder_filename = "./bin/encoder-epoch-30-avg-4.onnx";
  std::string decoder_filename = "./bin/decoder-epoch-30-avg-4.onnx";
  std::string joiner_filename = "./bin/joiner-epoch-30-avg-4.onnx";

  // std::string tokens= "./zh_model/tokens.txt";
  // std::string encoder_filename="./zh_model/encoder-epoch-12-avg-4.int8.onnx";
  // std::string decoder_filename="./zh_model/decoder-epoch-12-avg-4.int8.onnx";
  // std::string joiner_filename="./zh_model/joiner-epoch-12-avg-4.int8.onnx";

  // std::string tokens= "./bin/icefall-asr-zipformer-wenetspeech-20230615/data/lang_char/tokens.txt";
  // std::string encoder_filename="./bin/icefall-asr-zipformer-wenetspeech-20230615/exp/encoder-epoch-12-avg-4.int8.onnx";
  // std::string decoder_filename="./bin/icefall-asr-zipformer-wenetspeech-20230615/exp/decoder-epoch-12-avg-4.int8.onnx";
  // std::string joiner_filename="./bin/icefall-asr-zipformer-wenetspeech-20230615/exp/joiner-epoch-12-avg-4.int8.onnx";

  sherpa_onnx::OfflineTransducerModelConfig transducer;
  transducer.encoder_filename = encoder_filename;
  transducer.decoder_filename = decoder_filename;
  transducer.joiner_filename = joiner_filename;

  sherpa_onnx::OfflineModelConfig model_config;
  model_config.tokens = tokens;
  model_config.transducer = transducer;
  model_config.bpe_model = "./bin/bpe.model";
  sherpa_onnx::OfflineRecognizerConfig config;
  config.model_config = model_config;

  config.hotwords_file="hotwords.txt";  
  config.hotwords_score=2;

  if (!config.Validate())
  {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }


  fprintf(stdout, "Creating recognizer ...\n");
  sherpa_onnx::OfflineRecognizer recognizer(config);

  std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>> ss;
  std::vector<sherpa_onnx::OfflineStream *> ss_pointers;
  //////////////////////////////////////123

  /////////// Init VAD //////////////////

  int32_t segment_index = 0;
  litevad_handle_t vad_handle =
      litevad_create(SAMPLE_RATE, RECORD_CHANNEL_COUNT, RECORD_SAMPLE_BIT);
  if (vad_handle == NULL)
  {
    fprintf(stderr, "litevad_create failed\n");
  }
  int vad_begin = 0;
  int vad_end = 0;

  int32_t vad_activate_sample = (SAMPLE_RATE * 500) / 1000;
  int32_t vad_silence_sample = (SAMPLE_RATE * 0) / 1000;

  std::string path = "./bin/silero_vad.onnx";
  int test_sr = 16000;
  int test_frame_ms = 96;
  float test_threshold = 0.85f;
  int test_min_silence_duration_ms = 100;
  int test_speech_pad_ms = 0;
  int test_window_samples = test_frame_ms * (test_sr / 1000);

  VadIterator ai_vad(
      path, test_sr, test_frame_ms, test_threshold,
      test_min_silence_duration_ms, test_speech_pad_ms);

  //////////////////////////////////////


  //////////////// Init TTS //////////////////////

  tts_params ttsConfig;
  ttsConfig.modelPath = "./bin/en_US-joe-medium.onnx";
  ttsConfig.modelConfigPath = "./bin/en_US-joe-medium.onnx.json";
  ttsConfig.piperConfig.eSpeakDataPath = "./bin/espeak-ng-data/";
  ttsConfig.play_device = "plughw:1,0";  // using aplay -l to checkout the sound devices for play
  ttsConfig.silence_ms = 100;         // the silence ms between tecx chunk segment by punctuation

  // Initialize the tts
  init_tts(ttsConfig);

  // Perform the tts
  // std::string line = "Hello i'm vibe bot. I am an AI-powered assistant here to assist you. Whether you need information, help with answer question, or just someone to chat with, I'm here for you";
  // process_tts(line, ttsConfig);

  //The alsa play here is set to non-blocking mode. So, need wait it play finished.
  snd_pcm_state_t state = snd_pcm_state(ttsConfig.pcm_handle);
  while (state == SND_PCM_STATE_RUNNING)
  {
    state = snd_pcm_state(ttsConfig.pcm_handle);
  }

  ////////////////////////////////////////////////////////////////////////////////

  // params.is_running=false;

  //// Main loop for audio real time process //////
  while (params.is_running)
  {
    fprintf(stderr, "Started\n");

    while (true)
    {
      int len = audio_CQ_get(&params, 96, 0);
      if (len > 0)
      {
        const auto onnx_begin = std::chrono::steady_clock::now();

        int32_t vad_state = ai_vad.predict(params.audio.pcmf32_new);

        if (vad_state == 2)
        {
          printf("begin\n");
          // printf("params->CQ_audio_exit%d\n",params.CQ_audio_exit);
          vad_begin = params.CQ_audio_exit - vad_activate_sample;
        }
        if (vad_state == 3)
        {
          printf("end\n");
          // printf("params->CQ_audio_exit%d\n",params.CQ_audio_exit);
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
    auto s = recognizer.CreateStream();
    s->AcceptWaveform(SAMPLE_RATE, params.audio.pcmf32_new.data(), params.audio.pcmf32_new.size());

    ss.push_back(std::move(s));
    ss_pointers.push_back(ss.back().get());
    recognizer.DecodeStreams(ss_pointers.data(), 1);

    const auto end = std::chrono::steady_clock::now();
    std::string text = ss[0]->GetResult().text;

    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c)
                   { return std::tolower(c); });
    fprintf(stderr, "TEXT: %s\n----\n", text.c_str());
    float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count() /
        1000.;

    float duration = (float)params.audio.pcmf32_new.size() / (float)SAMPLE_RATE;

    float rtf = elapsed_seconds / duration;
    fprintf(stderr, "Real time factor for stt: %.3f / %.3f = %.3f\n",
            elapsed_seconds, duration, rtf);

    // tts the text generate by stt
    process_tts(text, ttsConfig);


    //The alsa play function is set to non-blocking mode. So, need wait it play finished.
    snd_pcm_state_t state = snd_pcm_state(ttsConfig.pcm_handle);
    while (state == SND_PCM_STATE_RUNNING)
    {
      state = snd_pcm_state(ttsConfig.pcm_handle);
    }

    int len = audio_CQ_clear(&params); // clear the audio buffer when processing the last audio chunk

    ss.clear();
    ss_pointers.clear();
  }

  /////////// deinit //////////////

  ss.clear();
  ss_pointers.clear();
  litevad_destroy(vad_handle);

  ///////////////////////////////////////////
  return 0;
}
#endif
