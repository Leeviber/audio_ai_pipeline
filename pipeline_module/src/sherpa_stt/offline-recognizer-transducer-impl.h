// sherpa-onnx/csrc/offline-recognizer-transducer-impl.h
//

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <regex>  
#include <iostream>

#include "sentencepiece_processor.h"  // NOLINT

#include "context-graph.h"
#include "macros.h"
#include "offline-recognizer-impl.h"
#include "offline-recognizer.h"
#include "offline-transducer-decoder.h"
#include "offline-transducer-greedy-search-decoder.h"
#include "offline-transducer-model.h"
#include "offline-transducer-modified-beam-search-decoder.h"
#include "pad-sequence.h"
#include "symbol-table.h"
#include "utils.h"

namespace sherpa_onnx {

static OfflineRecognitionResult Convert(
    const OfflineTransducerDecoderResult &src, const SymbolTable &sym_table,
    int32_t frame_shift_ms, int32_t subsampling_factor) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];
    text.append(sym);

    r.tokens.push_back(std::move(sym));
  }
  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    r.timestamps.push_back(time);
  }

  return r;
}

class OfflineRecognizerTransducerImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerTransducerImpl(
      const OfflineRecognizerConfig &config)
      : config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineTransducerModel>(config_.model_config)) {
      if (!config_.model_config.bpe_model.empty()) {
        auto status = bpe_processor_.Load(config_.model_config.bpe_model);
        if (!status.ok()) {
          SHERPA_ONNX_LOGE("Load bpe model error, status : %s.",
                          status.ToString().c_str());
          exit(-1);
      }
    }
    if (!config_.hotwords_file.empty()) {
      printf("using hotword");
      InitHotwords();
    }
    if (config_.decoding_method == "greedy_search") {
      printf("using greedy_search");

      decoder_ =
          std::make_unique<OfflineTransducerGreedySearchDecoder>(model_.get());
    } else if (config_.decoding_method == "modified_beam_search") {
            printf("using modified_beam_search");

      if (!config_.lm_config.model.empty()) {
        lm_ = OfflineLM::Create(config.lm_config);
      }

      decoder_ = std::make_unique<OfflineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      exit(-1);
    }
  }



  std::unique_ptr<OfflineStream> CreateStream(
      const std::string &hotwords) const override {
    auto hws = std::regex_replace(hotwords, std::regex("/"), "\n");
    std::istringstream is(hws);
    std::vector<std::vector<int32_t>> current;
    if (!EncodeHotwords(is, symbol_table_, &current)) {
      SHERPA_ONNX_LOGE("Encode hotwords failed, skipping, hotwords are : %s",
                       hotwords.c_str());
    }
    current.insert(current.end(), hotwords_.begin(), hotwords_.end());

    auto context_graph =
        std::make_shared<ContextGraph>(current, config_.hotwords_score);
    return std::make_unique<OfflineStream>(config_.feat_config, context_graph);
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config,
                                           hotwords_graph_);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = ss[0]->FeatureDim();

    std::vector<Ort::Value> features;
  
    features.reserve(n);

    std::vector<std::vector<float>> features_vec(n);
    std::vector<int64_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      auto f = ss[i]->GetFrames();
      int32_t num_frames = f.size() / feat_dim;

      features_length_vec[i] = num_frames;
      features_vec[i] = std::move(f);

      std::array<int64_t, 2> shape = {num_frames, feat_dim};

      Ort::Value x = Ort::Value::CreateTensor(
          memory_info, features_vec[i].data(), features_vec[i].size(),
          shape.data(), shape.size());
      features.push_back(std::move(x));
    }

    std::vector<const Ort::Value *> features_pointer(n);
    for (int32_t i = 0; i != n; ++i) {
      features_pointer[i] = &features[i];
    }

    std::array<int64_t, 1> features_length_shape = {n};
    Ort::Value x_length = Ort::Value::CreateTensor(
        memory_info, features_length_vec.data(), n,
        features_length_shape.data(), features_length_shape.size());

    Ort::Value x = PadSequence(model_->Allocator(), features_pointer,
                               -23.025850929940457f);

    auto t = model_->RunEncoder(std::move(x), std::move(x_length));
    auto results =
        decoder_->Decode(std::move(t.first), std::move(t.second), ss, n);

    int32_t frame_shift_ms = 10;
    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_, frame_shift_ms,
                       model_->SubsamplingFactor());

      ss[i]->SetResult(r);
    }
  }

  void InitHotwords() {
    // each line in hotwords_file contains space-separated words

    std::ifstream is(config_.hotwords_file);
    if (!is) {
      SHERPA_ONNX_LOGE("Open hotwords file failed: %s",
                       config_.hotwords_file.c_str());
      exit(-1);
    }

    InitHotwords(is);  //for text

    // if (!EncodeHotwords(is, symbol_table_, &hotwords_)) {   //for token
    //   SHERPA_ONNX_LOGE("Encode hotwords failed.");
    //   exit(-1);
    // }
    hotwords_graph_ =
        std::make_shared<ContextGraph>(hotwords_, config_.hotwords_score);
  }

  void EncodeWithBpe(const std::string word, std::vector<std::string> *syms) {
    syms->clear();
    std::vector<std::string> bpes;
    if (bpe_processor_.status().ok()) {
      if (bpe_processor_.Encode(word, &bpes).ok()) {
        for (auto bpe : bpes) {
          if (bpe.size() >= 3) {
            // For BPE-based models, we replace ▁ with a space
            // Unicode 9601, hex 0x2581, utf8 0xe29681
            const uint8_t *p = reinterpret_cast<const uint8_t *>(bpe.c_str());
            if (p[0] == 0xe2 && p[1] == 0x96 && p[2] == 0x81) {
              bpe = bpe.replace(0, 3, " ");
            }
          }
          syms->push_back(bpe);
        }
      } else {
        SHERPA_ONNX_LOGE("SentencePiece encode error for hotword %s. ",
                         word.c_str());
        exit(-1);
      }
    } else {
      SHERPA_ONNX_LOGE("SentencePiece processor error : %s.",
                       bpe_processor_.status().ToString().c_str());
      exit(-1);
    }
  }

  void InitHotwords(std::istream &is) {
    std::vector<int32_t> tmp;
    std::string line;
    std::string word;

    while (std::getline(is, line)) {
      std::istringstream iss(line);
      std::vector<std::string> syms;
      while (iss >> word) {
        if (config_.model_config.tokens_type == "cjkchar") {
          syms.push_back(word);
        } else if (config_.model_config.tokens_type == "bpe") {
          std::vector<std::string> bpes;
          EncodeWithBpe(word, &bpes);
          syms.insert(syms.end(), bpes.begin(), bpes.end());
        }  
      }
      for (auto sym : syms) {
        if (symbol_table_.contains(sym)) {
          int32_t number = symbol_table_[sym];
          tmp.push_back(number);
        } else {
          SHERPA_ONNX_LOGE(
              "Cannot find ID for hotword %s at line: %s. (Hint: words on "
              "the "
              "same line are separated by spaces)",
              sym.c_str(), line.c_str());
          exit(-1);
        }
      }
      hotwords_.push_back(std::move(tmp));
    }
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::vector<std::vector<int32_t>> hotwords_;
  ContextGraphPtr hotwords_graph_;

  sentencepiece::SentencePieceProcessor bpe_processor_;

  std::unique_ptr<OfflineTransducerModel> model_;
  std::unique_ptr<OfflineTransducerDecoder> decoder_;
  std::unique_ptr<OfflineLM> lm_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
