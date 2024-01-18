// sherpa-onnx/csrc/voice-activity-detector.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "voice-activity-detector.h"

#include <algorithm>
#include <queue>
#include <utility>

#include "circular-buffer.h"
#include "vad-model.h"

namespace sherpa_onnx
{

  class VoiceActivityDetector::Impl
  {
  public:
    explicit Impl(const VadModelConfig &config, float buffer_size_in_seconds = 90)
        : model_(VadModel::Create(config)),
          config_(config),
          buffer_(buffer_size_in_seconds * config.sample_rate) {}

    void AcceptWaveform(const float *samples, int32_t n)
    {
      int32_t window_size = model_->WindowSize();
      // note n is usually window_size and there is no need to use
      // an extra buffer here
      last_.insert(last_.end(), samples, samples + n);

      int32_t k = static_cast<int32_t>(last_.size()) / window_size;

      const float *p = last_.data();
      bool is_speech = false;

      for (int32_t i = 0; i != k; ++i, p += window_size)
      {
        buffer_.Push(p, window_size);
        is_speech = is_speech || model_->IsSpeech(p, window_size);
      }

      last_ = std::vector<float>(
          p, static_cast<const float *>(last_.data()) + last_.size());

      if (is_speech)
      {
 
        if (start_ == -1)
        {
          // beginning of speech
          start_ = std::max(buffer_.Tail() - 2 * model_->WindowSize() -
                                model_->MinSpeechDurationSamples(),
                            buffer_.Head());
        }
      }
      else
      {
        // non-speech
        if (start_ != -1 && buffer_.Size())
        {

          // end of speech, save the speech segment
          int32_t end = buffer_.Tail() - model_->MinSilenceDurationSamples();

          std::vector<float> s = buffer_.Get(start_, end - start_);

          int32_t s_length = s.size();

          int chunkSize = whisper_30s_samples; // 30秒的长度限制

          if (s_length < whisper_30s_samples)
          {
            SpeechSegment segment;
            segment.start = start_;
            segment.samples = std::move(s);
            segments_.push(std::move(segment));
          }
          else
          {
            for (int i = 0; i < s_length; i += chunkSize)
            {
              SpeechSegment segment;
              segment.start = start_ + i; // update segment start time

              // Calculate the end index of the current chunk
              int endIdx = i + chunkSize;
              if (endIdx > s_length)
              {
                endIdx = s_length;
              }

              // Create a new vector with the chunk of samples
              segment.samples = std::vector<float>(s.begin() + i, s.begin() + endIdx);

              segments_.push(std::move(segment));
            }
          }

          buffer_.Pop(end - buffer_.Head());
        }

        if (start_ == -1)
        {
          int32_t end = buffer_.Tail() - 2 * model_->WindowSize() -
                        model_->MinSpeechDurationSamples();
          int32_t n = std::max(0, end - buffer_.Head());
          if (n > 0)
          {
            buffer_.Pop(n);
          }
        }

        start_ = -1;
      }
    }

    bool Empty() const { return segments_.empty(); }

    void Pop() { segments_.pop(); }

    void Clear() { std::queue<SpeechSegment>().swap(segments_); }

    const SpeechSegment &Front() const { return segments_.front(); }

    void Reset()
    {
      std::queue<SpeechSegment>().swap(segments_);

      model_->Reset();
      buffer_.Reset();

      start_ = -1;
    }

    bool IsSpeechDetected() const { return start_ != -1; }

  private:
    std::queue<SpeechSegment> segments_;

    std::unique_ptr<VadModel> model_;
    VadModelConfig config_;
    CircularBuffer buffer_;
    std::vector<float> last_;

    int32_t start_ = -1;
    int32_t whisper_30s_samples = 29 * config_.sample_rate;
  };

  VoiceActivityDetector::VoiceActivityDetector(
      const VadModelConfig &config, float buffer_size_in_seconds /*= 60*/)
      : impl_(std::make_unique<Impl>(config, buffer_size_in_seconds)) {}

  VoiceActivityDetector::~VoiceActivityDetector() = default;

  void VoiceActivityDetector::AcceptWaveform(const float *samples, int32_t n)
  {
    impl_->AcceptWaveform(samples, n);
  }

  bool VoiceActivityDetector::Empty() const { return impl_->Empty(); }

  void VoiceActivityDetector::Pop() { impl_->Pop(); }

  void VoiceActivityDetector::Clear() { impl_->Clear(); }

  const SpeechSegment &VoiceActivityDetector::Front() const
  {
    return impl_->Front();
  }

  void VoiceActivityDetector::Reset() { impl_->Reset(); }

  bool VoiceActivityDetector::IsSpeechDetected() const
  {
    return impl_->IsSpeechDetected();
  }

} // namespace sherpa_onnx
