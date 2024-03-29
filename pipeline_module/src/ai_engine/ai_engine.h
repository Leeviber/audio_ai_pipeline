#ifndef STT_ENGINE_H
#define STT_ENGINE_H

#include <algorithm>
#include <numeric>
#include <cmath>
#include <math.h>
#include <map>

#include "wav/wavfile.hpp"

#include "sherpa_stt/offline-recognizer.h"
#include "sherpa_stt/offline-model-config.h"
#include "sherpa_stt/voice-activity-detector.h"

#include "speaker_id/frontend/feature_pipeline.h"
#include "speaker_id/speaker/speaker_model.h"
#include "speaker_id/speaker/rknn_speaker_model.h"
#include "speaker_id/speaker/onnx_speaker_model.h"

#include "speaker_diarization/speaker_diarization.h"
struct Diarization
{
    int id;
    std::vector<float> start;
    std::vector<float> end;
    std::vector<std::string> texts;

    Diarization(int newId, const std::vector<float> &newStart, const std::vector<float> &newEnd, const std::vector<std::string> &newTexts) : id(newId), start(newStart), end(newEnd), texts(newTexts) {}

    void addDiarization(const float newStart, const float newEnd, const std::string &newText)
    {
        start.push_back(newStart);
        end.push_back(newEnd);
        texts.push_back(newText);
    }
};

struct DiarizationSequence
{
    int x;
    int y;
    DiarizationSequence(int newX, int newY) : x(newX), y(newY) {}
};

class STTEngine
{
public:
    STTEngine(bool using_whisper, bool using_chinese);

    std::string perform_stt(const std::vector<float> *audioData);

private:
    std::unique_ptr<sherpa_onnx::OfflineRecognizer> recognizer;

    int sampleRate = 16000;
};

class SpeakerID
{
public:
    SpeakerID(const std::vector<std::string> &models_path,
              const int embedding_size);

    // return embedding_size
    int EmbeddingSize();
    // extract fbank
    void ExtractFeature(const int16_t *data, int data_size,
                        std::vector<std::vector<std::vector<float>>> *chunks_feat);
    // extract embedding
    void ExtractEmbedding(const int16_t *data, int data_size,
                          std::vector<float> *avg_emb);

    void ApplyMean(std::vector<std::vector<float>> *feats,
                   unsigned int feat_dim);

    int FindMaxSimilarityKey(const std::vector<float> &inputVector);

    void MapToDoubleArray(std::vector<std::vector<double>> &outputArray);

    void UpdateAverageEmbedding(int key, const std::vector<float> &newEmbedding);

    void AddNewKeyValue(const std::vector<float> &newValue);

    float CosineSimilarity(const std::vector<float> &emb1,
                           const std::vector<float> &emb2);

private:
    std::shared_ptr<wespeaker::SpeakerModel> rknn_model_ = nullptr;

    std::shared_ptr<wespeaker::SpeakerModel> onnx_model_ = nullptr;

    std::shared_ptr<wenet::FeaturePipelineConfig> feature_config_ = nullptr;

    std::shared_ptr<wenet::FeaturePipeline> feature_pipeline_ = nullptr;

    std::map<int, std::vector<float>> averageEmbeddings;

    int embedding_size_ = 0;

    int per_chunk_samples_ = 32000;

    int sample_rate_ = 16000;
};

class VADChunk
{
public:
    VADChunk(
        const std::string &model_path,
        const int window_size,
        const float vad_threshold,
        const float min_silence_duration,
        bool dumpOutput,
        const std::string textFileName);

    void PushAudioChunk(const std::vector<float> &audio_chunk);

    void STT(STTEngine *stt_interface);

    void SpeakerDiarization(STTEngine *stt_interface, SpeakerID *speaker_id_engine, Cluster *cst);

    void printSTTAnnotation(double start, double end, const std::string &text);

    void printAllDiarizations(bool sequence);

    void saveSTTAnnotation(std::string fileName, double start, double end, const std::string &text, bool appendToFile = false);

    void saveDiarizationsAnnotation(std::string fileName, bool sequence, bool appendToFile);

private:
    int sampleRate = 16000;

    float min_segment_length = 1.5;

    std::unique_ptr<sherpa_onnx::VoiceActivityDetector> vad_;

    std::vector<Diarization> diarization_annote;

    std::vector<DiarizationSequence> diarization_sequence;

    bool dumpOutput;

    std::string fileName;
};

#endif // STT_ENGINE_H
