#ifndef STT_INTERFACE_H
#define STT_INTERFACE_H

#include <algorithm>
#include <numeric>
#include <cmath>
#include <math.h>

#include "sherpa_stt/offline-recognizer.h"
#include "sherpa_stt/offline-model-config.h"

class STTInterface
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

#endif // STT_INTERFACE_H
