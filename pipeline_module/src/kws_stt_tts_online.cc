#ifndef KWS_STT_TTS_ONLINE

#include <stdio.h>
#include <chrono>  
#include <string>
#include <vector>
#include "sherpa_stt/offline-recognizer.h"
#include "sherpa_stt/offline-model-config.h"

#include "kws.h"              // kws
#include "alsa_cq_buffer.h"  // ALSA
#include "tts.h"        //tts



#include "kws_stt_tts_online.h"

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

int main() {
  
 
 ///////////// Init online audio ////////////////
  online_params params;
  int32_t ret = init_online_audio(&params);
  if(ret<0){
      fprintf(stderr, "Error init_kws \n");
      return -1;
  }

  bool trigger_stt=false;
//////////////////////////////////////


//// Init ALSA and circular buffer////
  snd_pcm_t *capture_handle;
  const char *device_name ="plughw:5,0";   // using arecord -l to checkout the alsa device name
  ret = audio_CQ_init(device_name, SAMPLE_RATE, &params, capture_handle);
//////////////////////////////////////


///////////// Init KWS ////////////////

  kws_params kws_params;
  kws_params.emb_path="./bin/embedding_864.bin";
  kws_params.fb_path="./bin/fb_t.bin";
  kws_params.eff_model_path="./bin/eff_word.rknn";
  ret = init_kws(&kws_params);
 

//// Init Sherpa STT module //////////

  std::string tokens= "./bin/tokens.txt";
  std::string encoder_filename="./bin/encoder-epoch-30-avg-4.int8.onnx";
  std::string decoder_filename="./bin/decoder-epoch-30-avg-4.int8.onnx";
  std::string joiner_filename="./bin/joiner-epoch-30-avg-4.int8.onnx";

  sherpa_onnx::OfflineTransducerModelConfig transducer;
  transducer.encoder_filename=encoder_filename;
  transducer.decoder_filename=decoder_filename;
  transducer.joiner_filename=joiner_filename;

  sherpa_onnx::OfflineModelConfig model_config;
  model_config.tokens=tokens;
  model_config.transducer=transducer;

  sherpa_onnx::OfflineRecognizerConfig config;
  config.model_config=model_config;
  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }
  fprintf(stdout, "Creating recognizer ...\n");
  sherpa_onnx::OfflineRecognizer recognizer(config);
  std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>> ss;
  std::vector<sherpa_onnx::OfflineStream *> ss_pointers;
////////////////////////////////////


//////////////// Init TTS //////////////////////
 
  tts_params ttsConfig;

  // Initialize the API
  init_tts(ttsConfig);

  // Perform the execution
  std::string line = "Hello i'm vibe bot. I am an AI-powered assistant here to assist you. Whether you need information, help with answer question, or just someone to chat with, I'm here for you";
  process_tts(line,ttsConfig);


////////////////////////////////////////////////////////////////////////////////


// Main loop for audio real time process //////
  while (params.is_running)
  {
    fprintf(stderr, "Started\n");
  
    while (true)
    {
        int len = audio_CQ_get(&params,1500,500);
        if (len >= (1e-3 * 1500) * SAMPLE_RATE)
        {
            break;
        }

    }
    bool isSpotting=kws_process(&params,&kws_params);

    if(isSpotting)
    {
        // relax_time=2;
        printf("!!!!!!!!! Hey vibe detected !!!!!!!!!!!");
        printf("\n .....Please Speaking..... \n");

        trigger_stt=true;
        int len=audio_CQ_clear(&params);

    }
    while(trigger_stt)
    {
        while (true)
        {
          int len = audio_CQ_get(&params,3000,2900);
          if (len >= (1e-3 * 3000) * SAMPLE_RATE)
          {
              break;
          }
        }

        const auto begin = std::chrono::steady_clock::now();
        auto s = recognizer.CreateStream();
        s->AcceptWaveform(SAMPLE_RATE, params.audio.pcmf32_new.data(), params.audio.pcmf32_new.size());
        
        ss.push_back(std::move(s));
        ss_pointers.push_back(ss.back().get());
        recognizer.DecodeStreams(ss_pointers.data(), 1);

        const auto end = std::chrono::steady_clock::now();
        const std::string text= ss[0]->GetResult().text;
        fprintf(stderr, "TEXT: %s\n----\n" ,text.c_str());
        process_tts(text,ttsConfig);

        float elapsed_seconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
                .count() /
            1000.;

        float duration =params.audio.pcmf32_new.size()/SAMPLE_RATE;

        float rtf = elapsed_seconds / duration;
        fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
                elapsed_seconds, duration, rtf);

        ss.clear();
        ss_pointers.clear();
        trigger_stt=false;
    }

  }

/////////// deinit //////////////
  ret = deinit_kws(&kws_params);
  if(ret<0){
      fprintf(stderr, "Error deinit_kws \n");
      return -1;
  }

  ss.clear();
  ss_pointers.clear();  
  

  // Terminate Piper
  piper::terminate(ttsConfig.piperConfig);

  snd_pcm_close(ttsConfig.pcm_handle);

 
 ///////////////////////////////////////////
  return 0;
}
#endif

