#include "ai_engine.h"

STTEngine::STTEngine(bool using_whisper, bool using_chinese)
{
  std::string tokens;
  sherpa_onnx::OfflineModelConfig model_config;

  if (using_chinese)
  {
    tokens = "./bin/sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt";
    std::string model = "./bin/sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx";
    sherpa_onnx::OfflineParaformerModelConfig paraformer;
    paraformer.model = model;
    model_config.model_type = "paraformer";
    model_config.paraformer = paraformer;
  }
  else
  {

    if (using_whisper)
    {
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

std::string STTEngine::perform_stt(const std::vector<float> *audioData)
{

  auto s = recognizer->CreateStream();
  s->AcceptWaveform(sampleRate, audioData->data(), audioData->size());
  recognizer->DecodeStream(s.get());
  const std::string text = s->GetResult().text;

  return text;
}

VADChunk::VADChunk(
    const std::string &model_path,
    const int window_size,
    const float vad_threshold,
    const float min_silence_duration,
    bool dumpOutput,
    std::string textFileName)
{
  sherpa_onnx::VadModelConfig vad_config;
  sherpa_onnx::SileroVadModelConfig silero_vad;
  silero_vad.model = model_path;
  silero_vad.min_silence_duration = min_silence_duration;
  silero_vad.threshold = vad_threshold;
  silero_vad.window_size = (window_size / 1000.0f) * vad_config.sample_rate;
  vad_config.silero_vad = silero_vad;
  vad_ = std::make_unique<sherpa_onnx::VoiceActivityDetector>(vad_config);
  dumpOutput = dumpOutput;
  fileName = textFileName;
}

void VADChunk::PushAudioChunk(const std::vector<float> &audio_chunk)
{
  vad_->AcceptWaveform(audio_chunk.data(), audio_chunk.size());
}

void VADChunk::STT(STTEngine *stt_interface)
{
  while (!vad_->Empty())
  {

    auto &segment = vad_->Front();
    std::string text = stt_interface->perform_stt(&segment.samples);
    float start = (float)segment.start / sampleRate;
    float end = (float)(segment.start + segment.samples.size()) / sampleRate;
    if (dumpOutput)
    {
      saveSTTAnnotation(fileName, start, end, text, true);
    }

    printSTTAnnotation(start, end, text);

    vad_->Pop();
  }
}

void VADChunk::SpeakerDiarization(STTEngine *stt_interface, SpeakerID *speaker_id_engine, Cluster *cst)
{

  if (!vad_->Empty())
  {

    auto &segment = vad_->Front();
    int segment_length = segment.samples.size();

    if (segment_length < min_segment_length * sampleRate)
    {
      vad_->Pop();
      return;
    }

    float start = (float)segment.start / sampleRate;
    float end = (float)(segment.start + segment_length) / sampleRate;

    std::string text = stt_interface->perform_stt(&segment.samples);
    std::vector<int16_t> enroll_data_int16(segment_length);

    for (int i = 0; i < segment_length; i++)
    {
      enroll_data_int16[i] = static_cast<int16_t>(segment.samples[i] * 32767.0f);
    }
    std::vector<float> chunk_emb(speaker_id_engine->EmbeddingSize(), 0);

    speaker_id_engine->ExtractEmbedding(enroll_data_int16.data(), enroll_data_int16.size(), &chunk_emb);

    int match_idx = speaker_id_engine->FindMaxSimilarityKey(chunk_emb);

    if (match_idx != -1)
    {
      printf("matched\n");

      speaker_id_engine->UpdateAverageEmbedding(match_idx, chunk_emb);
      int text_size = diarization_annote[match_idx].texts.size();

      diarization_sequence.push_back(DiarizationSequence(match_idx, text_size));

      diarization_annote[match_idx].addDiarization(start, end, text);
    }
    else
    {
      printf("unmatched\n");

      speaker_id_engine->AddNewKeyValue(chunk_emb);

      std::vector<std::vector<double>> idArray;
      speaker_id_engine->MapToDoubleArray(idArray);
      int min_clu_size = idArray.size();
      std::vector<int> clustersRes; // 存储聚类结果
      cst->custom_clustering(idArray, clustersRes);
      std::vector<int> id_list;

      id_list = cst->mergeAndRenumber(clustersRes);

      if (id_list.size() < 1)
      {
        diarization_annote.push_back(Diarization(0, {start}, {end}, {text}));
        diarization_sequence.push_back(DiarizationSequence(0, 0));
      }
      else
      {
        diarization_sequence.push_back(DiarizationSequence(diarization_annote.size(), 0));
        diarization_annote.push_back(Diarization(id_list[-1], {start}, {end}, {text}));

        for (int i = 0; i < diarization_annote.size(); i++)
        {
          if (diarization_annote[i].id != id_list[i])
          {
            diarization_annote[i].id = id_list[i];
          }
        }
      }
    }

    printAllDiarizations(true); // true mean the result will print in sequence
                                // false will print in group
    if (dumpOutput)
    {
      saveDiarizationsAnnotation(fileName, true, false);
    }
    vad_->Pop();
  }

  return;
}

void VADChunk::printAllDiarizations(bool sequence)
{
  printf("\n-------------------speaker diaization ------------------\n");

  if (sequence)
  {
    for (const auto &diarization_seq : diarization_sequence)
    {
      Diarization diarization = diarization_annote[diarization_seq.x];
      int idx = diarization_seq.y;
      printf("Speaker: %d, Time:[%.2f - %.2f]s , Text: \"%s\"\n", diarization.id, diarization.start[idx], diarization.end[idx], diarization.texts[idx].c_str());
    }
  }
  else
  {
    for (const auto &diarization : diarization_annote)
    {
      for (size_t i = 0; i < diarization.start.size(); ++i)
      {
        printf("Speaker: %d, Time:[%.2f - %.2f]s , Text: \"%s\"\n", diarization.id, diarization.start[i], diarization.end[i], diarization.texts[i].c_str());
      }
    }
  }
}

void VADChunk::saveSTTAnnotation(std::string fileName, double start, double end, const std::string &text, bool appendToFile)
{
  // 打开文本文件，根据参数决定是覆盖还是追加
  std::ofstream outputFile(fileName, appendToFile ? std::ios_base::app : std::ios_base::out);
  printf("file name%s\n", fileName.c_str());
  if (!outputFile.is_open())
  {
    std::cerr << "Error opening file for writing." << std::endl;
    return;
  }

  // 将信息写入文件
  outputFile << "Time: [" << start << "s ~ " << end << "s], Text: " << text << "\"\n";

  // 关闭文件
  outputFile.close();
}
void VADChunk::printSTTAnnotation(double start, double end, const std::string &text)
{
  printf("Time: [%.2fs~%.2fs], Text: %s\n", start, end, text.c_str());
}

void VADChunk::saveDiarizationsAnnotation(std::string fileName, bool sequence, bool appendToFile = false)

{
  // 打开文本文件，根据参数决定是覆盖还是追加
  std::ofstream outputFile(fileName, appendToFile ? std::ios_base::app : std::ios_base::out);

  if (!outputFile.is_open())
  {
    std::cerr << "Error opening file for writing." << std::endl;
    return;
  }

  if (sequence)
  {
    for (const auto &diarization_seq : diarization_sequence)
    {
      Diarization diarization = diarization_annote[diarization_seq.x];
      int idx = diarization_seq.y;
      // 将信息写入文件
      outputFile << "Speaker: " << diarization.id << ", Time: [" << diarization.start[idx] << " - " << diarization.end[idx] << "]s, Text: \"" << diarization.texts[idx] << "\"\n";
    }
  }
  else
  {
    for (const auto &diarization : diarization_annote)
    {
      for (size_t i = 0; i < diarization.start.size(); ++i)
      {
        // 将信息写入文件
        outputFile << "Speaker: " << diarization.id << ", Time: [" << diarization.start[i] << " - " << diarization.end[i] << "]s, Text: \"" << diarization.texts[i] << "\"\n";
      }
    }
  }

  // 关闭文件
  outputFile.close();
}

SpeakerID::SpeakerID(const std::vector<std::string> &models_path,
                     const int embedding_size)
{
  const int kNumGemmThreads = 4;
  embedding_size_ = embedding_size;

  feature_config_ = std::make_shared<wenet::FeaturePipelineConfig>(
      80, sample_rate_);
  feature_pipeline_ =
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

int SpeakerID::EmbeddingSize()
{
  return embedding_size_;
}

void SpeakerID::ApplyMean(std::vector<std::vector<float>> *feat,
                          unsigned int feat_dim)
{
  std::vector<float> mean(feat_dim, 0);
  for (auto &i : *feat)
  {
    std::transform(i.begin(), i.end(), mean.begin(), mean.begin(),
                   std::plus<>{});
  }
  std::transform(mean.begin(), mean.end(), mean.begin(),
                 [&](const float d)
                 { return d / feat->size(); });
  for (auto &i : *feat)
  {
    std::transform(i.begin(), i.end(), mean.begin(), i.begin(), std::minus<>{});
  }
}

void SpeakerID::ExtractFeature(const int16_t *data, int data_size,
                               std::vector<std::vector<std::vector<float>>> *chunks_feat)
{
  if (data != nullptr)
  {
    std::vector<std::vector<float>> chunk_feat;
    feature_pipeline_->AcceptWaveform(std::vector<int16_t>(
        data, data + data_size));
    if (per_chunk_samples_ <= 0)
    {
      feature_pipeline_->Read(feature_pipeline_->num_frames(), &chunk_feat);
      feature_pipeline_->Reset();
      chunks_feat->emplace_back(chunk_feat);
      chunk_feat.clear();
    }
    else
    {
      int num_chunk_frames_ = 1 + ((
                                       per_chunk_samples_ - sample_rate_ / 1000 * 25) /
                                   (sample_rate_ / 1000 * 10));

      int chunk_num = std::ceil(
          feature_pipeline_->num_frames() / num_chunk_frames_);

      chunks_feat->reserve(chunk_num);
      chunk_feat.reserve(num_chunk_frames_);
      while (feature_pipeline_->NumQueuedFrames() >= num_chunk_frames_)
      {
        feature_pipeline_->Read(num_chunk_frames_, &chunk_feat);
        chunks_feat->emplace_back(chunk_feat);
        chunk_feat.clear();
      }
      int last_frames = feature_pipeline_->NumQueuedFrames();

      if (last_frames > 0)
      {
        feature_pipeline_->Read(last_frames, &chunk_feat);
        if (chunks_feat->empty())
        {
          int num_pad = static_cast<int>(num_chunk_frames_ / last_frames);
          for (int i = 1; i < num_pad; i++)
          {
            chunk_feat.insert(chunk_feat.end(), chunk_feat.begin(),
                              chunk_feat.begin() + last_frames);
          }
          chunk_feat.insert(chunk_feat.end(), chunk_feat.begin(),
                            chunk_feat.begin() + num_chunk_frames_ - chunk_feat.size());
        }
        else
        {
          chunk_feat.insert(chunk_feat.end(),
                            (*chunks_feat)[0].begin(),
                            (*chunks_feat)[0].begin() + num_chunk_frames_ - chunk_feat.size());
        }
        chunks_feat->emplace_back(chunk_feat);
        chunk_feat.clear();
      }
      feature_pipeline_->Reset();
    }
  }
  else
  {
    printf("input is null");
  }
}

void SpeakerID::ExtractEmbedding(const int16_t *data, int data_size,
                                 std::vector<float> *avg_emb)
{

  std::vector<std::vector<std::vector<float>>> chunks_feat;
  this->ExtractFeature(data, data_size, &chunks_feat);
  int chunk_num = chunks_feat.size();
  avg_emb->resize(embedding_size_, 0);

  for (int i = 0; i < chunk_num; i++)
  {
    std::vector<float> tmp_emb;
    this->ApplyMean(&chunks_feat[i], chunks_feat[i][0].size());

#ifdef USE_NPU
    std::vector<float> resnet_out;
    resnet_out.resize(256 * 10 * 25);
    rknn_model_->ExtractResnet(chunks_feat[i], resnet_out);
    onnx_model_->ResnetPostprocess(&resnet_out, &tmp_emb);
#else
    onnx_model_->ExtractEmbedding(chunks_feat[i], &tmp_emb);
#endif

    for (int j = 0; j < tmp_emb.size(); j++)
    {
      (*avg_emb)[j] += tmp_emb[j];
    }
  }
  for (int i = 0; i < avg_emb->size(); i++)
  {
    (*avg_emb)[i] /= chunk_num;
  }
}

float SpeakerID::CosineSimilarity(const std::vector<float> &emb1,
                                  const std::vector<float> &emb2)
{
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

int SpeakerID::FindMaxSimilarityKey(const std::vector<float> &inputVector)
{
  float maxSimilarity = 0.4;
  int maxKey = -1;

  for (const auto &pair : averageEmbeddings)
  {
    int key = pair.first;
    const std::vector<float> &embedding = pair.second;

    // 计算余弦相似度
    float similarity = CosineSimilarity(inputVector, embedding);

    // 更新最大相似度和对应的键
    if (similarity > maxSimilarity)
    {
      maxSimilarity = similarity;
      maxKey = key;
    }
  }

  return maxKey;
}

void SpeakerID::MapToDoubleArray(std::vector<std::vector<double>> &outputArray)
{
  // 清空输出数组
  outputArray.clear();

  // 遍历输入 map
  for (const auto &pair : averageEmbeddings)
  {
    const std::vector<float> &values = pair.second;
    std::vector<double> chunk_emb_double(EmbeddingSize(), 0);

    for (int i = 0; i < EmbeddingSize(); i++)
    {
      chunk_emb_double[i] = static_cast<double>(values[i]);
    }

    outputArray.push_back(chunk_emb_double);
  }
}

void SpeakerID::UpdateAverageEmbedding(int key, const std::vector<float> &newEmbedding)
{
  auto it = averageEmbeddings.find(key);

  if (it != averageEmbeddings.end())
  {

    // 如果 key 已存在，更新平均值
    std::vector<float> &existingValue = it->second;

    if (existingValue.size() == newEmbedding.size())
    {

      // 合并平均值
      std::transform(existingValue.begin(), existingValue.end(), newEmbedding.begin(),
                     existingValue.begin(), [](float a, float b)
                     { return (a + b) / 2.0; });
    }
    else
    {
      std::cerr << "Error: Vector sizes mismatch." << std::endl;
    }
  }
  else
  {
    printf("Error\n");

    // 如果 key 不存在，直接插入新值
    averageEmbeddings[key] = newEmbedding;
  }
}
void SpeakerID::AddNewKeyValue(const std::vector<float> &newValue)
{
  int newKey = static_cast<int>(averageEmbeddings.size());

  averageEmbeddings[newKey] = newValue;
}
