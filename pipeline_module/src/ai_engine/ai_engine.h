#ifndef STT_INTERFACE_H
#define STT_INTERFACE_H

#include <algorithm>
#include <numeric>
#include <cmath>
#include <math.h>

#include "sherpa_stt/offline-recognizer.h"
#include "sherpa_stt/offline-model-config.h"
#include "sherpa_stt/voice-activity-detector.h"
#include "speaker_id/frontend/feature_pipeline.h"

#include "speaker_id/speaker/speaker_model.h"
#include "speaker_id/speaker/rknn_speaker_model.h"
#include "speaker_id/speaker/onnx_speaker_model.h"

#include "speaker_diarization/speaker_diarization.h"

class STTEngine
{
private:
    std::unique_ptr<sherpa_onnx::OfflineRecognizer> recognizer;

    int sampleRate = 16000;

public:
    void init_stt(bool using_whisper);

    std::string perform_stt(const std::vector<float> &audioData);
};

class SpeakerID {
 public:
  explicit SpeakerID(const std::vector<std::string>& models_path,
                         const int feat_dim,
                         const int sample_rate,
                         const int embedding_size,
                         const int SamplesPerChunk);
                         
  // return embedding_size
  int EmbeddingSize();
  // extract fbank
  void ExtractFeature(const int16_t* data, int data_size,
    std::vector<std::vector<std::vector<float>>>* chunks_feat);
  // extract embedding
  void ExtractEmbedding(const int16_t* data, int data_size,
                        std::vector<float>* avg_emb);

  // void ExtractRknnResult(const int16_t* data, int data_size,
  //                       std::vector<float>* net_out);

  float CosineSimilarity(const std::vector<float>& emb1,
                        const std::vector<float>& emb2);

 private:
  void ApplyMean(std::vector<std::vector<float>>* feats,
                 unsigned int feat_dim);
  std::shared_ptr<wespeaker::SpeakerModel> rknn_model_ = nullptr;
  std::shared_ptr<wespeaker::SpeakerModel> onnx_model_ = nullptr;
  std::shared_ptr<wenet::FeaturePipelineConfig> feature_config_ = nullptr;
  std::shared_ptr<wenet::FeaturePipeline> feature_pipeline_ = nullptr;
  int embedding_size_ = 0;
  int per_chunk_samples_ = 32000;
  int sample_rate_ = 16000;
};

class VADChunk
{
public:
    void InitVAD(const std::string &model_path, const int window_size);
    void PushAudioChunk(const std::vector<float> &audio_chunk);
    void STT(STTEngine &stt_interface);
    void ExtractId(SpeakerID &speaker_id_engine, Cluster cluster);
    void process_embedding(SpeakerID &speaker_id_engine);


private:
    std::unique_ptr<sherpa_onnx::VoiceActivityDetector> vad_;
    std::vector<std::vector<double>> embeddings_; 

};


#endif // STT_INTERFACE_H
