#ifndef STT_INTERFACE_H
#define STT_INTERFACE_H

#include <algorithm>
#include <numeric>
#include <cmath>
#include <math.h>
#include <map>

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

class SpeakerID
{
public:
    struct EmbedSegment
    {

        std::string text;
        std::vector<float> embedding;

        EmbedSegment(const std::string &t, const std::vector<float> &emb) : text(t), embedding(emb) {}
    };

    void addSegment(const std::string &text, const std::vector<float> &embedding);
    void addEmbedSegmentsToMap(const std::vector<int> &ids);
    int  findMaxSimilarityKey(const std::vector<float>& inputVector);
    void mapToDoubleArray(std::vector<std::vector<double>>& outputArray);
    void updateAverageEmbedding(int key, const std::vector<float>& newEmbedding);
    void addNewKeyValue(const std::vector<float>& newValue);
 
    explicit SpeakerID(const std::vector<std::string> &models_path,
                       const int feat_dim,
                       const int sample_rate,
                       const int embedding_size,
                       const int SamplesPerChunk);

    // return embedding_size
    int EmbeddingSize();
    // extract fbank
    void ExtractFeature(const int16_t *data, int data_size,
                        std::vector<std::vector<std::vector<float>>> *chunks_feat);
    // extract embedding
    void ExtractEmbedding(const int16_t *data, int data_size,
                          std::vector<float> *avg_emb);

    // void ExtractRknnResult(const int16_t* data, int data_size,
    //                       std::vector<float>* net_out);

    float CosineSimilarity(const std::vector<float> &emb1,
                           const std::vector<float> &emb2);
    bool reduce = false;
    std::map<int, std::vector<float>> averageEmbeddings;

private:
    void ApplyMean(std::vector<std::vector<float>> *feats,
                   unsigned int feat_dim);
    std::shared_ptr<wespeaker::SpeakerModel> rknn_model_ = nullptr;
    std::shared_ptr<wespeaker::SpeakerModel> onnx_model_ = nullptr;
    std::shared_ptr<wenet::FeaturePipelineConfig> feature_config_ = nullptr;
    std::shared_ptr<wenet::FeaturePipeline> feature_pipeline_ = nullptr;
    int embedding_size_ = 0;
    int per_chunk_samples_ = 32000;
    int sample_rate_ = 16000;
    std::vector<EmbedSegment> segments;
    std::map<int, std::vector<EmbedSegment>> idMap;

};

class VADChunk
{
public:
    struct Diarization
    {
        int id;
        std::vector<std::string> texts;

        // 构造函数，允许提供初始值
        Diarization(int newId, const std::vector<std::string>& newTexts) : id(newId), texts(newTexts) {}

        // 成员函数，用于向 texts 向量中添加新的元素
        void addText(const std::string& newText) {
            texts.push_back(newText);
        }
    };

    std::vector<Diarization> diarization_annote;

    void InitVAD(const std::string &model_path, const int window_size);
    void PushAudioChunk(const std::vector<float> &audio_chunk);
    void STT(STTEngine &stt_interface);
    void ExtractId(STTEngine &stt_interface, SpeakerID &speaker_id_engine, Cluster cluster);
    void process_embedding(SpeakerID &speaker_id_engine);

private:
    std::unique_ptr<sherpa_onnx::VoiceActivityDetector> vad_;
    std::vector<std::vector<double>> embeddings_;
    std::vector<std::string> texts_;
    std::map<int, std::vector<std::string>> textIdMap;

};

#endif // STT_INTERFACE_H
