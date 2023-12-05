#include <stdio.h>
#include <chrono>  
#include <string>
#include <vector>
#include "offline-recognizer.h"
#include "litevad_api.h"       // vad
#include "kws.h"              // kws
#include "alsa_cq_buffer.h"  // ALSA
// #include "BYTETracker.h"

#define SAMPLE_RATE 16000

int main() {
  
  // byte_track::BYTETracker tracker(3, 100);

  // for(int i=0;i<100;i++)
  // {

  //   std::vector<byte_track::Object> inputs;
  

  //   byte_track::Rect<float> rect1(0.0f, 24.0f, 199.0f, 10.0f);
  //   byte_track::Rect<float> rect2(0.0f, 53.0f, 5.0f, 54.0f);
  //   byte_track::Rect<float> rect3(0.3, 55.0f, 10.0f, 10.0f);
  //   byte_track::Rect<float> rect4(0.345, 5.0f, 3.43f, 4.0f);

  //   // 创建一个 Object 对象并为其成员变量赋值
  //   byte_track::Object obj1(rect1, 1, 0.8f);
  //   byte_track::Object obj2(rect2, 1, 0.6f);
  //   byte_track::Object obj3(rect3, 1, 0.8f);
  //   byte_track::Object obj4(rect4, 1, 0.7f);

  //   inputs.push_back(obj1);
  //   inputs.push_back(obj2);
  //   inputs.push_back(obj3);
  //   inputs.push_back(obj4);

  //   const auto outputs = tracker.update(inputs);
  //   printf("outputs size%d\n",outputs.size());

  //   printf("outputs 0 id%d \n",outputs[0]->getTrackId());
 
  //   byte_track::Rect<float> rect_res1=outputs[0]->getRect();
  //   printf("rect_res %f \n",rect_res1.width());
  //   inputs.clear();

  // }
 ///////////// Init KWS ////////////////
  kws_params params;
  int32_t ret = init_kws(&params);
  if(ret<0){
      fprintf(stderr, "Error init_kws \n");
      return -1;
  }
  bool using_vad=false;

  bool trigger_stt=false;
//////////////////////////////////////

//// Init ALSA and circular buffer////
  snd_pcm_t *capture_handle;
  const char *device_name ="plughw:5,0";   // using arecord -l to checkout the alsa device name
  ret = audio_CQ_init(device_name, SAMPLE_RATE, &params, capture_handle);
//////////////////////////////////////


//// Init Sherpa STT module //////////
  sherpa_onnx::OfflineRecognizerConfig config;
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
 
  int32_t segment_index = 0;
  litevad_handle_t vad_handle =
      litevad_create(SAMPLE_RATE, RECORD_CHANNEL_COUNT, RECORD_SAMPLE_BIT);
  if (vad_handle == NULL) {
      fprintf(stderr, "litevad_create failed\n");
  }
  bool using_kws=true;
  int vad_begin=0;
  int vad_end=0;
 
  int32_t vad_activate_sample = (SAMPLE_RATE * 600) / 1000;  
  int32_t vad_silence_sample = (SAMPLE_RATE * 200) / 1000;

//////////////////////////////////////


// std::ofstream  ("data.bin", std::ios::binary);  //save the input audio for debug


//// Main loop for audio real time process //////
  while (params.is_running)
  {
    fprintf(stderr, "Started\n");
  
    while (true)
    {
        if(using_vad)
        {
          int len = audio_CQ_get(&params,30,0);
          if(len>0)
          {
            short s16Data[480];  
            convertFloatToS16LE(params.audio.pcmf32_new.data(), s16Data, 480);
            litevad_result_t vad_state = litevad_process(vad_handle, s16Data, 960);
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
        else if(using_kws)
        {
          int len = audio_CQ_get(&params,params.step_ms,params.keep_ms);
          if (len >= params.n_samples_step)
          {
              break;
          }

        }
        else 
        {
          int len = audio_CQ_get(&params,3000,2900);
          if (len >= params.n_samples_step)
          {
              break;
          }
        }
              
    }
    while (using_kws)
    {
      bool isSpotting=kws_process(&params);

    }
    
    
    // file.write(reinterpret_cast<const char*>(params.audio.pcmf32_new.data()), params.audio.pcmf32_new.size() * sizeof(float));


    const auto begin = std::chrono::steady_clock::now();
    auto s = recognizer.CreateStream();
    s->AcceptWaveform(SAMPLE_RATE, params.audio.pcmf32_new.data(), params.audio.pcmf32_new.size());
    
    ss.push_back(std::move(s));
    ss_pointers.push_back(ss.back().get());
    recognizer.DecodeStreams(ss_pointers.data(), 1);

    const auto end = std::chrono::steady_clock::now();
    const std::string text= ss[0]->GetResult().text;
    fprintf(stderr, "TEXT: %s\n----\n" ,text.c_str());

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

  }

/////////// deinit //////////////
  ret = deinit_kws(&params);
  if(ret<0){
      fprintf(stderr, "Error deinit_kws \n");
      return -1;
  }

  ss.clear();
  ss_pointers.clear();  
  litevad_destroy(vad_handle);

 ///////////////////////////////////////////
  return 0;
}
