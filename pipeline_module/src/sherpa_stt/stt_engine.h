#ifndef STT_INTERFACE_H
#define STT_INTERFACE_H

#include <algorithm>
#include <numeric>
#include <cmath>
#include <math.h>

#include "offline-recognizer.h"
#include "offline-model-config.h"
#include "voice-activity-detector.h"

class STTEngine
{
private:
    std::unique_ptr<sherpa_onnx::OfflineRecognizer> recognizer;
    std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>> ss;
    std::vector<sherpa_onnx::OfflineStream *> ss_pointers;
    int sampleRate = 16000;

public:
    void init_stt(bool using_whisper);

    std::string perform_stt(const std::vector<float> &audioData);
};

class VADChunkSTT {
public:
    void InitVAD(const std::string& model_path,const int window_size);
    void PushAudioChunk(const std::vector<float>& audio_chunk);
    void STT(STTEngine& stt_interface);

private:
    std::unique_ptr<sherpa_onnx::VoiceActivityDetector> vad_;
};

#endif // STT_INTERFACE_H
