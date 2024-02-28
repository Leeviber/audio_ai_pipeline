#include "alsa_cq_buffer.h"      // ALSA
#include "ai_engine/ai_engine.h" // AI audio engine
#include "speaker_id/frontend/wav.h"
int main()
{

  //// Init ALSA and circular buffer////
  online_params params;

  // int32_t ret = init_online_audio(&params);
  // if (ret < 0)
  // {
  //   fprintf(stderr, "Error init_kws \n");
  //   return -1;
  // }
  // snd_pcm_t *capture_handle;
  // const char *device_name = "plughw:2,0"; // using arecord -l to checkout the alsa device name
  // ret = audio_CQ_init(device_name, params.sample_rate, &params, capture_handle);
  //////////////////////////////////////

  //// Init Sherpa STT module //////////
  bool using_Whisper = false;  // English only
  bool using_Chinese = true; // Not support whipser
  STTEngine stt_interface(using_Whisper, using_Chinese);
  //////////////////////////////////////

  /////////// Init chunk VAD //////////////////

  int vad_frame_ms = 32; // audio chunk length(ms) for VAD detect, (32,64,96), large is more accuray with more latency
  std::string vad_path = "./bin/silero_vad.onnx";
  float min_silence_duration = 0.2;
  float vad_threshold = 0.85;
  bool saveAnanotation = true;
  std::string filename = "diarization_output.txt";
  int test_window_samples = vad_frame_ms * (16000 / 1000);

  VADChunk vad_chunk_stt(vad_path, vad_frame_ms, vad_threshold, 
  min_silence_duration, saveAnanotation, filename);
  //////////////////////////////////////


  /////////// Init Segmentation model //////////////////

  const std::string segmentModel = "./bin/seg_model.onnx";
  SegmentModel mm(segmentModel);
  //////////////////////////////////////


  /////////// Init speaker id and cluster //////////////////

  std::vector<std::string> model_paths;
#ifdef USE_NPU
  std::string rknn_model_path = "./bin/Id1_resnet34_LM_main_part.rknn";
  std::string onnx_model_path = "./bin/Id2_resnet34_LM_post.onnx";
  model_paths.push_back(rknn_model_path);
  model_paths.push_back(onnx_model_path);

#else
  // std::string onnx_model_path = "./bin/3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx";
  std::string onnx_model_path = "./bin/voxceleb_resnet34_LM.onnx";
  // std::string onnx_model_path = "./bin/voxceleb_CAM++_LM.onnx";

  model_paths.push_back(onnx_model_path);

#endif 
  int embedding_size = 256;

  // Init speaker id
  SpeakerID speaker_id(model_paths, embedding_size);

  // Init cluster
  Cluster cluster;

  printf("Init success\n");
  //// Main loop for audio real time process //////
  //////////////////////////////////////
  std::string audio_path = "./bin/speaker_diarization_long_test/8speaker.wav";
  auto data_reader = wenet::ReadAudioFile(audio_path);
  int16_t *enroll_data_int16 = const_cast<int16_t *>(data_reader->data());
  int samples = data_reader->num_sample();
  std::vector<float> enroll_data_flt32(samples);
  for (int i = 0; i < samples; i++)
  {
      enroll_data_flt32[i] = static_cast<float>(enroll_data_int16[i]) / 32768;
  }

 

  // 记录开始时间
  auto start = std::chrono::high_resolution_clock::now();

  // 计算音频数据块的数量
  int audio_chunk = samples / test_window_samples;

  for (int j = 0; j < audio_chunk; j++)
  {
    while (true)
    {
      std::vector<float> window_chunk{&enroll_data_flt32[0] + j * test_window_samples, &enroll_data_flt32[0] + j * test_window_samples + test_window_samples};
      vad_chunk_stt.PushAudioChunk(window_chunk);
      break;  
    }
    vad_chunk_stt.SpeakerDiarization(&mm, &stt_interface, &speaker_id, &cluster);
    
  }

  // while (params.is_running)
  // {

  //   while (true)
  //   {
  //     int len = audio_CQ_get(&params, vad_frame_ms, 0); // The audio windows for ai vad is 64ms
     
  //     if (len >= vad_frame_ms/1000*params.sample_rate)
  //     {
  //       vad_chunk_stt.PushAudioChunk(params.audio.pcmf32_new);  
  //       break;
  //     }
  //   }

  //   vad_chunk_stt.SpeakerDiarization(&stt_interface, &speaker_id, &cluster);
  // }

  ///////////////////////////////////////////
  return 0;
}
