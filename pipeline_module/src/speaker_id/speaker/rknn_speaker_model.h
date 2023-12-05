#ifndef SPEAKER_RKNN_SPEAKER_MODEL_H_
#define SPEAKER_RKNN_SPEAKER_MODEL_H_

#ifdef USE_NPU

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstring> 
#include "rknn_api.h"

// #include "onnxruntime_cxx_api.h"  // NOLINT
#include "speaker_model.h"

namespace wespeaker {

class RknnSpeakerModel : public SpeakerModel {
public:
    static void InitEngineThreads(int num_threads = 1);

public:
    explicit RknnSpeakerModel(const std::string& model_path);

    void ExtractResnet(const std::vector<std::vector<float>>& feats,
                            std::vector<float>& resnet_out) override;

private:
    // session
    char* eff_model_path;
    rknn_context rk_ctx;
    int rknn_ret;
        // node names
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    int embedding_size_ = 0;
};

}  // namespace wespeaker
#endif

#endif  // SPEAKER_RKNN_SPEAKER_MODEL_H_
