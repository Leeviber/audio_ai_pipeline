#include "ai_engine.h"

void STTEngine::init_stt(bool using_whisper)
{
    std::string tokens;
    sherpa_onnx::OfflineModelConfig model_config;

    if (using_whisper)
    {
        // tokens = "./bin/distil-small.en-tokens.txt";
        // std::string encoder_filename = "./bin/distil-small.en-encoder.int8.onnx";
        // std::string decoder_filename = "./bin/distil-small.en-decoder.int8.onnx";

        tokens = "./bin/distil-medium.en-tokens.txt";
        std::string encoder_filename = "./bin/distil-medium.en-encoder.int8.onnx";
        std::string decoder_filename = "./bin/distil-medium.en-decoder.int8.onnx";

        sherpa_onnx::OfflineWhisperModelConfig whisper;
        whisper.encoder = encoder_filename;
        whisper.decoder = decoder_filename;
        whisper.language = "en";
        whisper.tail_paddings = 800;

        model_config.model_type = "whisper";
        model_config.whisper = whisper;
    }
    else
    {
        tokens = "./bin/encoder-epoch-30-avg-4-tokens.txt";
        std::string encoder_filename = "./bin/encoder-epoch-30-avg-4.int8.onnx";
        std::string decoder_filename = "./bin/decoder-epoch-30-avg-4.int8.onnx";
        std::string joiner_filename = "./bin/joiner-epoch-30-avg-4.int8.onnx";

        sherpa_onnx::OfflineTransducerModelConfig transducer;
        transducer.encoder_filename = encoder_filename;
        transducer.decoder_filename = decoder_filename;
        transducer.joiner_filename = joiner_filename;

        model_config.model_type = "transducer";
        model_config.transducer = transducer;
    }

    model_config.tokens = tokens;

    sherpa_onnx::OfflineRecognizerConfig config;
    config.model_config = model_config;

    if (!config.Validate())
    {
        fprintf(stderr, "Errors in config!\n");
        return;
    }

    fprintf(stdout, "Creating recognizer ...\n");
    recognizer = std::make_unique<sherpa_onnx::OfflineRecognizer>(config);
}

std::string STTEngine::perform_stt(const std::vector<float> &audioData)
{

    auto s = recognizer->CreateStream();
    s->AcceptWaveform(sampleRate, audioData.data(), audioData.size());
    recognizer->DecodeStream(s.get());
    const std::string text = s->GetResult().text;

    return text;
}

void VADChunk::InitVAD(const std::string &model_path, const int window_size)
{
    sherpa_onnx::VadModelConfig vad_config;
    sherpa_onnx::SileroVadModelConfig silero_vad;
    silero_vad.model = model_path;
    silero_vad.window_size = (window_size / 1000.0f) * vad_config.sample_rate;
    vad_config.silero_vad = silero_vad;
    vad_ = std::make_unique<sherpa_onnx::VoiceActivityDetector>(vad_config);
}

void VADChunk::PushAudioChunk(const std::vector<float> &audio_chunk)
{
    vad_->AcceptWaveform(audio_chunk.data(), audio_chunk.size());
}

void VADChunk::STT(STTEngine &stt_interface)
{
    while (!vad_->Empty())
    {
        auto &segment = vad_->Front();
        printf("stt samples length%d",segment.samples.size());
        std::string text = stt_interface.perform_stt(segment.samples);
        printf("TEXT: %s\n----\n", text.c_str());
        vad_->Pop();
    }
}

void VADChunk::ExtractId(SpeakerID &speaker_id_engine, Cluster cst)
{
    bool is_update = false;
    while (!vad_->Empty())
    {

        auto &segment = vad_->Front();
        std::vector<int16_t> enroll_data_int16(segment.samples.size());

        for (int i = 0; i < segment.samples.size(); i++)
        {
             enroll_data_int16[i] = static_cast<int16_t>(segment.samples[i] * 32767.0f);

        }
        std::vector<float> chunk_emb(speaker_id_engine.EmbeddingSize(), 0);
        std::vector<double> chunk_emb_double(speaker_id_engine.EmbeddingSize(), 0);

        speaker_id_engine.ExtractEmbedding(enroll_data_int16.data(),enroll_data_int16.size(),&chunk_emb);
        for (int i = 0; i < chunk_emb.size(); i++)
        {
            chunk_emb_double[i] = static_cast<double>(chunk_emb[i]);
        }
        printf("extract once\n");
        embeddings_.push_back(chunk_emb_double);
        vad_->Pop();
        is_update = true;
    }
    if(is_update)
    {
        std::vector<int> clustersRes; // 存储聚类结果
        cst.custom_clustering(embeddings_, clustersRes);
        std::vector<int> merged_renumbered_numbers;
        merged_renumbered_numbers = cst.mergeAndRenumber(clustersRes);
    }

}
void VADChunk::process_embedding(SpeakerID &speaker_id_engine)
{
    printf("length of embedding %d\n",embeddings_.size());
}


SpeakerID::SpeakerID(const std::vector<std::string>& models_path,
                             const int feat_dim,
                             const int sample_rate,
                             const int embedding_size,
                             const int SamplesPerChunk) {
  // NOTE(cdliang): default num_threads = 1
  const int kNumGemmThreads = 4;
  // LOG(INFO) << "Reading model " << model_path;
  embedding_size_ = embedding_size;
  // LOG(INFO) << "Embedding size: " << embedding_size_;
  per_chunk_samples_ = SamplesPerChunk;
  // LOG(INFO) << "per_chunk_samples: " << per_chunk_samples_;
  sample_rate_ = sample_rate;
  // LOG(INFO) << "Sample rate: " << sample_rate_;
  feature_config_ = std::make_shared<wenet::FeaturePipelineConfig>(
    feat_dim, sample_rate);
  feature_pipeline_ = \
    std::make_shared<wenet::FeaturePipeline>(*feature_config_);
  feature_pipeline_->Reset();
  wespeaker::OnnxSpeakerModel::InitEngineThreads(kNumGemmThreads);
 
#ifdef USE_NPU
  std::string rknn_model_path = models_path[0];
  std::string onnx_model_path = models_path[1];

  rknn_model_ = std::make_shared<RknnSpeakerModel>(rknn_model_path);
  onnx_model_ = std::make_shared<OnnxSpeakerModel>(onnx_model_path);
#else
  std::string onnx_model_path = models_path[0];
  onnx_model_ = std::make_shared<wespeaker::OnnxSpeakerModel>(onnx_model_path);
#endif
}

int SpeakerID::EmbeddingSize() {
  return embedding_size_;
}

void SpeakerID::ApplyMean(std::vector<std::vector<float>>* feat,
                              unsigned int feat_dim) {
  std::vector<float> mean(feat_dim, 0);
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), mean.begin(),
                   std::plus<>{});
  }
  std::transform(mean.begin(), mean.end(), mean.begin(),
                 [&](const float d) {return d / feat->size();});
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), i.begin(), std::minus<>{});
  }
}

// 1. full mode
// When per_chunk_samples_ <= 0, extract the features of the full audio.
// 2. chunk by chunk
// Extract audio features chunk by chunk, with 198 frames for each chunk.
// If the last chunk is less than 198 frames,
// concatenate the head frame to the tail.
void SpeakerID::ExtractFeature(const int16_t* data, int data_size,
    std::vector<std::vector<std::vector<float>>>* chunks_feat) {
  if (data != nullptr) {
    std::vector<std::vector<float>> chunk_feat;
    feature_pipeline_->AcceptWaveform(std::vector<int16_t>(
        data, data + data_size));
    if (per_chunk_samples_ <= 0) {
      // full mode
      feature_pipeline_->Read(feature_pipeline_->num_frames(), &chunk_feat);
      feature_pipeline_->Reset();
      chunks_feat->emplace_back(chunk_feat);
      chunk_feat.clear();
    } else {
      // NOTE(cdliang): extract feature with chunk by chunk
      int num_chunk_frames_ = 1 + ((
        per_chunk_samples_ - sample_rate_ / 1000 * 25) /
        (sample_rate_ / 1000 * 10));

      int chunk_num = std::ceil( 
        feature_pipeline_->num_frames() / num_chunk_frames_);


      chunks_feat->reserve(chunk_num);
      chunk_feat.reserve(num_chunk_frames_);
      while (feature_pipeline_->NumQueuedFrames() >= num_chunk_frames_) {
        feature_pipeline_->Read(num_chunk_frames_, &chunk_feat);
        chunks_feat->emplace_back(chunk_feat);
        chunk_feat.clear();
      }
      // last_chunk
      int last_frames = feature_pipeline_->NumQueuedFrames();

      if (last_frames > 0) {
        feature_pipeline_->Read(last_frames, &chunk_feat);
        if (chunks_feat->empty()) {
          // wav_len < chunk_len
          int num_pad = static_cast<int>(num_chunk_frames_ / last_frames);
          for (int i = 1; i < num_pad; i++) {
            chunk_feat.insert(chunk_feat.end(), chunk_feat.begin(),
                              chunk_feat.begin() + last_frames);
          }
          chunk_feat.insert(chunk_feat.end(), chunk_feat.begin(),
            chunk_feat.begin() + num_chunk_frames_ - chunk_feat.size());
        } else {
          chunk_feat.insert(chunk_feat.end(),
            (*chunks_feat)[0].begin(),
            (*chunks_feat)[0].begin() + num_chunk_frames_ - chunk_feat.size());
        }
        // CHECK_EQ(chunk_feat.size(), num_chunk_frames_);
        chunks_feat->emplace_back(chunk_feat);
        chunk_feat.clear();
      }
      feature_pipeline_->Reset();
    }
 
  } else {
    // LOG(ERROR) << "Input is nullptr!";
    printf("input is null");
  }
}

void SpeakerID::ExtractEmbedding(const int16_t* data, int data_size,
                                     std::vector<float>* avg_emb) {
  // chunks_feat: [nchunk, T, D]

  std::vector<std::vector<std::vector<float>>> chunks_feat;
  this->ExtractFeature(data, data_size, &chunks_feat);
  int chunk_num = chunks_feat.size();
  avg_emb->resize(embedding_size_, 0);
 
  for (int i = 0; i < chunk_num; i++) {
    std::vector<float> tmp_emb;
    this->ApplyMean(&chunks_feat[i], chunks_feat[i][0].size());

#ifdef USE_NPU
    std::vector<float> resnet_out;
    resnet_out.resize(256*10*25);
    rknn_model_->ExtractResnet(chunks_feat[i], resnet_out);
    onnx_model_->ResnetPostprocess(&resnet_out,&tmp_emb);
#else
    onnx_model_->ExtractEmbedding(chunks_feat[i], &tmp_emb);
#endif

    for (int j = 0; j < tmp_emb.size(); j++) {
      (*avg_emb)[j] += tmp_emb[j];
    }
  }
  for (int i = 0; i < avg_emb->size(); i++) {
    (*avg_emb)[i] /= chunk_num;
  }

  

}
 
float SpeakerID::CosineSimilarity(const std::vector<float>& emb1,
                                      const std::vector<float>& emb2) {
  // CHECK_EQ(emb1.size(), emb2.size());
  float dot = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0);
  float emb1_sum = std::inner_product(emb1.begin(), emb1.end(),
                                      emb1.begin(), 0.0);
  float emb2_sum = std::inner_product(emb2.begin(), emb2.end(),
                                      emb2.begin(), 0.0);
  dot /= std::max(std::sqrt(emb1_sum) * std::sqrt(emb2_sum),
                  std::numeric_limits<float>::epsilon());
  return dot;
}
