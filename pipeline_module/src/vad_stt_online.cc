#ifndef VAD_STT_ONLINE

#include <stdio.h>
#include <chrono>  
#include <string>
#include <vector>
#include "sherpa_stt/offline-recognizer.h"
#include "sherpa_stt/offline-model-config.h"
// #include "sherpa_stt/offline-transducer-model-config.h"
// #include "sherpa_stt/offline-whisper-model-config.h"

#include "ai_vad.h"       // ai based vad

#include "alsa_cq_buffer.h"  // ALSA
#include "vad_stt_online.h"

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
  bool using_vad=false;

  bool trigger_stt=false;
//////////////////////////////////////

//// Init ALSA and circular buffer////
  snd_pcm_t *capture_handle;
  const char *device_name ="plughw:2,0";   // using arecord -l to checkout the alsa device name
  ret = audio_CQ_init(device_name, SAMPLE_RATE, &params, capture_handle);
//////////////////////////////////////



//// Init Sherpa STT module //////////
  bool using_whisper=false;
  std::string tokens;
  
  sherpa_onnx::OfflineModelConfig model_config;

  if(using_whisper)
  {
    tokens= "./bin/distil-small.en-tokens.txt";
    std::string encoder_filename="./bin/distil-small.en-encoder.int8.onnx";
    std::string decoder_filename="./bin/distil-small.en-decoder.int8.onnx";
    sherpa_onnx::OfflineWhisperModelConfig whisper;
    whisper.encoder=encoder_filename;
    whisper.decoder=decoder_filename;
    whisper.language="en";
    whisper.tail_paddings=800;

    model_config.model_type="whisper";
    model_config.whisper=whisper;
  }
  else
  {
    // tokens= "./bin/encoder-epoch-30-avg-4-tokens.txt";
    // std::string encoder_filename="./bin/encoder-epoch-30-avg-4.int8.onnx";
    // std::string decoder_filename="./bin/decoder-epoch-30-avg-4.int8.onnx";
    // std::string joiner_filename="./bin/joiner-epoch-30-avg-4.int8.onnx";

    tokens= "./wenet_zh_model/tokens.txt";
    std::string encoder_filename="./wenet_zh_model/encoder-epoch-12-avg-4.int8.onnx";
    std::string decoder_filename="./wenet_zh_model/decoder-epoch-12-avg-4.int8.onnx";
    std::string joiner_filename="./wenet_zh_model/joiner-epoch-12-avg-4.int8.onnx";

    // tokens= "./zipformer_zh/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/tokens.txt";
    // std::string encoder_filename="./zipformer_zh/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/encoder-epoch-20-avg-1.int8.onnx";
    // std::string decoder_filename="./zipformer_zh/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/decoder-epoch-20-avg-1.int8.onnx";
    // std::string joiner_filename="./zipformer_zh/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/joiner-epoch-20-avg-1.int8.onnx";


    sherpa_onnx::OfflineTransducerModelConfig transducer;
    transducer.encoder_filename=encoder_filename;
    transducer.decoder_filename=decoder_filename;
    transducer.joiner_filename=joiner_filename;

    model_config.model_type="transducer";
    model_config.transducer=transducer;
  }

  model_config.tokens=tokens;

  // std::string tokens= "./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/data/lang_bpe_500/tokens.txt";
  // std::string encoder_filename="./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/encoder-epoch-60-avg-20.onnx"; 
  // std::string decoder_filename="./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/decoder-epoch-60-avg-20.onnx";
  // std::string joiner_filename="./icefall-asr-cv-corpus-13.0-2023-03-09-en-pruned-transducer-stateless7-2023-04-17/exp/joiner-epoch-60-avg-20.onnx";




  // sherpa_onnx::OfflineModelConfig model_config;
  // model_config.tokens=tokens;
  



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
//////////////////////////////////////


/////////// Init VAD //////////////////


  int vad_begin=0;
  int vad_end=0;
 
  int32_t vad_activate_sample = (SAMPLE_RATE * 200) / 1000;  
  int32_t vad_silence_sample = (SAMPLE_RATE * 0) / 1000;

  std::string path = "./bin/silero_vad.onnx";
  int test_sr = 16000;
  int test_frame_ms = 96;
  float test_threshold = 0.85f;
  int test_min_silence_duration_ms = 100;
  int test_speech_pad_ms = 0;
  int test_window_samples = test_frame_ms * (test_sr/1000);

  VadIterator ai_vad(
      path, test_sr, test_frame_ms, test_threshold,
      test_min_silence_duration_ms, test_speech_pad_ms);

//////////////////////////////////////


 
//// Main loop for audio real time process //////
while (params.is_running)
{
  fprintf(stderr, "Started\n");

  while (true)
  {
    int len = audio_CQ_get(&params,test_frame_ms,0);  // The audio windows for ai vad is 64ms
    if(len>0)
    {

      int32_t vad_state=ai_vad.predict(params.audio.pcmf32_new);


      if(vad_state==2)
      {
        printf("begin\n");
        printf("params->CQ_audio_exit%d\n",params.CQ_audio_exit);
        vad_begin=params.CQ_audio_exit-vad_activate_sample;

      }
      if(vad_state==3)
      {
        printf("end\n");
        printf("params->CQ_audio_exit%d\n",params.CQ_audio_exit);
        vad_end=params.CQ_audio_exit-vad_silence_sample;

        len=audio_CQ_view(&params,vad_begin,vad_end);
        if(len>0)
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
  const std::string text= ss[0]->GetResult().text;
  fprintf(stderr, "TEXT: %s\n----\n" ,text.c_str());
  int len=audio_CQ_clear(&params);        // clear the audio buffer when processing the last audio chunk 

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  float duration =(float)params.audio.pcmf32_new.size()/(float)SAMPLE_RATE;

  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  ss.clear();
  ss_pointers.clear();

}

/////////// deinit //////////////
 
  ss.clear();
  ss_pointers.clear();  
 
 ///////////////////////////////////////////
  return 0;
}
#endif

