#ifndef AI_ENGINE_H

#include "ai_engine.h"

// WAV 文件头结构
struct WavHeader
{
  char chunkId[4];
  uint32_t chunkSize;
  char format[4];
  char subchunk1Id[4];
  uint32_t subchunk1Size;
  uint16_t audioFormat;
  uint16_t numChannels;
  uint32_t sampleRate;
  uint32_t byteRate;
  uint16_t blockAlign;
  uint16_t bitsPerSample;
  char subchunk2Id[4];
  uint32_t subchunk2Size;
};

// 保存 WAV 文件
void saveWavFile(const std::string &filename, const std::vector<int16_t> &data, uint16_t numChannels, uint32_t sampleRate, uint16_t bitsPerSample)
{
  std::ofstream file(filename, std::ios::binary);

  // 创建 WAV 文件头
  WavHeader header;
  strncpy(header.chunkId, "RIFF", 4);
  header.chunkSize = data.size() * sizeof(int16_t) + sizeof(WavHeader) - 8;
  strncpy(header.format, "WAVE", 4);
  strncpy(header.subchunk1Id, "fmt ", 4);
  header.subchunk1Size = 16;
  header.audioFormat = 1;
  header.numChannels = numChannels;
  header.sampleRate = sampleRate;
  header.bitsPerSample = bitsPerSample;
  header.byteRate = sampleRate * numChannels * bitsPerSample / 8;
  header.blockAlign = numChannels * bitsPerSample / 8;
  strncpy(header.subchunk2Id, "data", 4);
  header.subchunk2Size = data.size() * sizeof(int16_t);

  // 写入文件头
  file.write(reinterpret_cast<const char *>(&header), sizeof(WavHeader));

  // 写入音频数据
  file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(int16_t));

  // 关闭文件
  file.close();

  std::cout << "WAV 文件保存成功：" << filename << std::endl;
}

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

void VADChunk::SpeakerDiarization(SegmentModel *mm, STTEngine *stt_interface, SpeakerID *speaker_id_engine, Cluster *cst)
{

  if (!vad_->Empty())
  {

    auto &segment = vad_->Front();
    int segment_length = segment.samples.size();
    std::string basePath = "test_audio/new_sd_audio_output/";
    std::string prefix = "audio";
    std::string fileExtension = ".wav";
    // std::string filename = basePath + prefix + std::to_string(file_count) + fileExtension;
    // file_count += 1;
    // saveWavFile(filename, enroll_data_int16, 1, 16000, 16);

    if (segment_length < min_segment_length * sampleRate)
    {
      printf("\n vad pop \n");
      vad_->Pop();
      return;
    }

    float start = (float)segment.start / sampleRate;
    float end = (float)(segment.start + segment_length) / sampleRate;

    
    std::vector<int16_t> enroll_data_int16(segment_length);

    for (int i = 0; i < segment_length; i++)
    {
      enroll_data_int16[i] = static_cast<int16_t>(segment.samples[i] * 32767.0f);
    }

    double start_time = static_cast<double>(segment.start) / sampleRate;
    double end_time = static_cast<double>(segment.start + segment.samples.size()) / sampleRate;
    start_time = floor(start_time * 100) / 100; // Keep only 2 decimal places
    end_time = floor(end_time * 100) / 100;     // Keep only 2 decimal places

    std::vector<std::pair<double, double>> segments;
    std::vector<std::vector<std::vector<float>>> segmentations;
    SlidingWindow res_frames;

    auto binarized = runSegmentationModel(mm, segment.samples, segments, segmentations, res_frames);
    int num_chunk = binarized.size();
    int num_frame = binarized[0].size();
    int frame_total = num_chunk * num_frame;
    std::vector<float> speaker_prob = countSpeakerProbabilities(binarized);
    for (int i = 0; i < speaker_prob.size(); ++i)
    {
      std::cout << "Speaker " << i << ": " << speaker_prob[i] << std::endl;
    }

    int num_chunk_thres = 1;
    float thres = (num_chunk <= 1) ? 0.3 : 0.2;
    int embedding_size = speaker_id_engine->EmbeddingSize();
    std::vector<std::vector<double>> tmp_embedding;
    std::vector<Annotation::Result> tmp_annote;
    if (shouldProcess(speaker_prob, thres))
    {

      std::vector<Annotation::Result> allLabel;
      std::map<int, std::vector<Annotation::Result>> mergedResults;
      auto audio_chunk = speaker_id_engine->generateDiarization(mm, segment.samples, binarized, segmentations, segments, res_frames, 32, speaker_id_engine, mergedResults, allLabel);
      printf("audio_chunk size%d \n", audio_chunk.size());
      printf("allLabel size%d \n", allLabel.size());

      for (int i = 0; i < audio_chunk.size(); i++)
      {
        std::vector<int16_t> shortVector = floatToShort(audio_chunk[i]);

        std::vector<float> single_emb(embedding_size, 0.0);
        speaker_id_engine->ExtractEmbedding(shortVector.data(),
                                            shortVector.size(),
                                            &single_emb);
        std::vector<double> double_embedding(single_emb.begin(), single_emb.end());

        tmp_embedding.push_back(double_embedding);
        Annotation::Result corret_label(allLabel[i].start + start_time, allLabel[i].end + end_time, allLabel[i].label);
        tmp_annote.push_back(corret_label);

        std::string text = stt_interface->perform_stt(&audio_chunk[i]);
        texts.push_back(text);

        std::string filename = basePath + prefix + std::to_string(file_count) + fileExtension;
        file_count += 1;
        // saveWavFile(filename, floatToShort(audio_chunk[i]), 1, 16000, 16);


      }

      std::cout << "Processing segment...\n";
      // 在这里添加你想要执行的处理逻辑
    }
    else
    {

      std::vector<float> single_emb(embedding_size, 0.0);
      speaker_id_engine->ExtractEmbedding(enroll_data_int16.data(),
                                          enroll_data_int16.size(),
                                          &single_emb);

      std::vector<double> double_embedding(single_emb.begin(), single_emb.end());

      tmp_embedding.push_back(double_embedding);
      Annotation::Result corret_label(start_time, end_time, 0);
      tmp_annote.push_back(corret_label);
      std::string text = stt_interface->perform_stt(&segment.samples);
      texts.push_back(text);

      std::string filename = basePath + prefix + std::to_string(file_count) + fileExtension;
      file_count += 1;
      // saveWavFile(filename, enroll_data_int16, 1, 16000, 16);
      std::cout << "Skipping segment...\n";
    }

    for (int i = 0; i < tmp_embedding.size(); ++i)
    {

      if (!std::isnan(tmp_embedding[i][0]))
      { // Assuming all elements in the innermost array are NaN or not NaN
        filter_global_embedding.push_back(tmp_embedding[i]);
        filter_global_annote.push_back(tmp_annote[i]);
      }
    }

    // Perform clustering on embeddings
    Cluster cst;
    std::vector<int> clustersRes;
    cst.custom_clustering(filter_global_embedding, clustersRes);
    // 合并并重新编号聚类结果的数字
    std::vector<int> merged_renumbered_numbers;
    merged_renumbered_numbers = mergeAndRenumberNumbers(clustersRes);

    // Output clustering results
    for (size_t i = 0; i < merged_renumbered_numbers.size(); ++i)
    {
      std::cout << "Audio start: " << secondsToMinutesAndSeconds(filter_global_annote[i].start) << ", end: " << secondsToMinutesAndSeconds(filter_global_annote[i].end) << " belongs to cluster " << merged_renumbered_numbers[i] << 
      "  text:"<<texts[i]<<std::endl;
    }

    // for(int i=0;i<global_embedding.size();i++)
    // {

    //   std::vector<float> chunk_emb =global_embedding[i];

    //   // speaker_id_engine->ExtractEmbedding(enroll_data_int16.data(), enroll_data_int16.size(), &chunk_emb);

    //   int match_idx = speaker_id_engine->FindMaxSimilarityKey(chunk_emb);

    //   if (match_idx != -1)
    //   {
    //     printf("matched\n");

    //     speaker_id_engine->UpdateAverageEmbedding(match_idx, chunk_emb);
    //     int text_size = diarization_annote[match_idx].texts.size();

    //     diarization_sequence.push_back(DiarizationSequence(match_idx, text_size));

    //     diarization_annote[match_idx].addDiarization(start, end, text);
    //   }
    //   else
    //   {
    //     printf("unmatched\n");

    //     speaker_id_engine->AddNewKeyValue(chunk_emb);

    //     std::vector<std::vector<double>> idArray;
    //     speaker_id_engine->MapToDoubleArray(idArray);
    //     int min_clu_size = idArray.size();

    //     std::vector<int> clustersRes; // 存储聚类结果
    //     cst->custom_clustering(idArray, clustersRes);
    //     std::vector<int> id_list;
    //     id_list = cst->mergeAndRenumber(clustersRes);

    //     if (id_list.size() < 1)
    //     {
    //       diarization_annote.push_back(Diarization(0, {start}, {end}, {text}));
    //       diarization_sequence.push_back(DiarizationSequence(0, 0));
    //     }
    //     else
    //     {
    //       diarization_sequence.push_back(DiarizationSequence(diarization_annote.size(), 0));
    //       diarization_annote.push_back(Diarization(id_list[-1], {start}, {end}, {text}));

    //       for (int i = 0; i < diarization_annote.size(); i++)
    //       {
    //         if (diarization_annote[i].id != id_list[i])
    //         {
    //           diarization_annote[i].id = id_list[i];
    //         }
    //       }
    //     }
    //   }

    //   printAllDiarizations(true); // true mean the result will print in sequence
    //                               // false will print in group
    //   if (dumpOutput)
    //   {
    //     saveDiarizationsAnnotation(fileName, true, false);
    //   }

    // }

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

int embed_model_size = 256;

int16_t floatToInt16(float value)
{
  // return static_cast<int16_t>(std::round(value));
  return static_cast<int16_t>(std::round(value * 32767.0f));
}

std::vector<std::vector<double>> SpeakerID::getEmbedding(SpeakerID *speaker_id_engine, const std::vector<std::vector<float>> &dataChunks,
                                                         const std::vector<std::vector<float>> &masks)
{

  // Debug
  static int number = 0;

  size_t batch_size = dataChunks.size();
  size_t num_samples = dataChunks[0].size();

  // python: imasks = F.interpolate(... ) and imasks = imasks > 0.5
  auto imasks = Helper::interpolate(masks, num_samples, 0.5);

  // masks is [32x293] imask is [32x80000], dataChunks is [32x80000] as welll

  // python: signals = pad_sequence(...)
  auto signals = Helper::padSequence(dataChunks, imasks);

  // python: wav_lens = imasks.sum(dim=1)
  std::vector<float> wav_lens(batch_size, 0.0);
  float max_len = 0;
  int index = 0;
  for (const auto &a : imasks)
  {
    float tmp = std::accumulate(a.begin(), a.end(), 0.0);
    wav_lens[index++] = tmp;
    if (tmp > max_len)
      max_len = tmp;
  }

  // python: if max_len < self.min_num_samples: return np.NAN * np.zeros(...
  if (max_len < 640)
  {
    // TODO: don't call embedding process, direct return
    // batch_size x 192, where 192 is size of length embedding result for each waveform
    // python: return np.NAN * np.zeros((batch_size, self.dimension))
    std::vector<std::vector<double>> embeddings(batch_size, std::vector<double>(embed_model_size, NAN));
    return embeddings;
  }

  // python:
  //      too_short = wav_lens < self.min_num_samples
  //      wav_lens = wav_lens / max_len
  //      wav_lens[too_short] = 1.0
  std::vector<bool> too_short(wav_lens.size(), false);
  for (size_t i = 0; i < wav_lens.size(); ++i)
  {
    if (wav_lens[i] < 640)
    {
      wav_lens[i] = 1.0;
      too_short[i] = true;
    }
    else
    {
      wav_lens[i] /= max_len;
    }
  }

  // signals is [32x80000], wav_lens is of length 32 of 1d array, an example for wav_lens
  // [1.0000, 1.0000, 1.0000, 0.0512, 1.0000, 1.0000, 0.1502, ...]
  // Now call embedding model to get embeddings of batches
  // speechbrain/pretrained/interfaces.py:903
  std::vector<std::vector<int16_t>> signals_int16(signals.size(), std::vector<int16_t>(num_samples));

  std::vector<std::vector<float>> embeddings_f(signals.size(),
                                               std::vector<float>(embed_model_size, 0.0));

  for (int i = 0; i < batch_size; i++)
  {
    std::transform(signals[i].begin(), signals[i].end(), signals_int16[i].begin(), floatToInt16);

    std::vector<float> single_emb(embed_model_size, 0.0);
    speaker_id_engine->ExtractEmbedding(signals_int16[i].data(),
                                        num_samples,
                                        &single_emb);
    embeddings_f[i] = single_emb;
  }

  // auto embeddings_f = em.infer( signals, wav_lens );

  // Convert float to double
  size_t col = embeddings_f[0].size();
  std::vector<std::vector<double>> embeddings(embeddings_f.size(),
                                              std::vector<double>(col));

  // python: embeddings[too_short.cpu().numpy()] = np.NAN
  for (size_t i = 0; i < too_short.size(); ++i)
  {
    if (too_short[i])
    {
      for (size_t j = 0; j < col; ++j)
      {
        embeddings[i][j] = NAN;
      }
    }
    else
    {
      for (size_t j = 0; j < col; ++j)
      {
        embeddings[i][j] = static_cast<double>(embeddings_f[i][j]);
      }
    }
  }
  // std::cout<<"embeddings "<<embeddings.size()<<embeddings[0].size()<<std::endl;

  return embeddings;
}

std::vector<std::vector<float>> SpeakerID::generateDiarization(SegmentModel *mm,
                                                               const std::vector<float> &input_wav,
                                                               const std::vector<std::vector<std::vector<double>>> &binarized,
                                                               const std::vector<std::vector<std::vector<float>>> &segmentations,
                                                               const std::vector<std::pair<double, double>> &segments,
                                                               SlidingWindow &res_frames,
                                                               size_t embedding_batch_size,
                                                               SpeakerID *speaker_id_engine,
                                                               std::map<int, std::vector<Annotation::Result>> &mergedResults,
                                                               std::vector<Annotation::Result> &allSegment)
{
  std::vector<std::vector<float>> batchData;
  std::vector<std::vector<float>> batchMasks;
  std::vector<std::vector<double>> embeddings;

  double duration = 5.0;
  int num_samples = input_wav.size();
  SlidingWindow count_frames(num_samples);
  double self_frame_step = 0.016875;
  double self_frame_duration = 0.016875;
  double self_frame_start = 0.0;
  size_t min_num_samples = 640;
  SlidingWindow pre_frame(self_frame_start, self_frame_step, self_frame_duration);
  auto count_data = mm->speaker_count(segmentations, binarized,
                                      pre_frame, count_frames, num_samples);
  // 计算最小帧数
  size_t min_num_frames = ceil(binarized[0].size() * min_num_samples / (duration * 16000));

  // 清理分割结果
  auto clean_segmentations = Helper::cleanSegmentations(binarized);
  assert(binarized.size() == clean_segmentations.size());
  // 生成embedding
  for (size_t i = 0; i < binarized.size(); ++i)
  {
    auto chunkData = mm->crop(input_wav, segments[i]);
    auto &masks = binarized[i];
    auto &clean_masks = clean_segmentations[i];
    assert(masks[0].size() == 3);
    assert(clean_masks[0].size() == 3);
    for (size_t j = 0; j < clean_masks[0].size(); ++j)
    {
      std::vector<float> used_mask;
      float sum = 0.0;
      std::vector<float> reversed_clean_mask(clean_masks.size());
      std::vector<float> reversed_mask(masks.size());

      for (size_t k = 0; k < clean_masks.size(); ++k)
      {
        sum += clean_masks[k][j];
        reversed_clean_mask[k] = clean_masks[k][j];
        reversed_mask[k] = masks[k][j];
      }

      if (sum > min_num_frames)
      {
        used_mask = std::move(reversed_clean_mask);
      }
      else
      {
        used_mask = std::move(reversed_mask);
      }

      // 将数据加入batch
      batchData.push_back(chunkData);
      batchMasks.push_back(std::move(used_mask));

      // 达到batch大小时，进行embedding计算
      if (batchData.size() == embedding_batch_size)
      {
        auto embedding = getEmbedding(speaker_id_engine, batchData, batchMasks);
        batchData.clear();
        batchMasks.clear();

        for (auto &a : embedding)
        {
          embeddings.push_back(std::move(a));
        }
      }
    }
  }

  // 处理剩余的数据
  if (batchData.size() > 0)
  {
    auto embedding = getEmbedding(speaker_id_engine, batchData, batchMasks);
    for (auto &a : embedding)
    {
      embeddings.push_back(std::move(a));
    }
  }
  printf("finish embedding process, size%d\n", embeddings.size());
  auto embeddings1 = Helper::rearrange_up(embeddings, binarized.size());

  Cluster cst;
  std::vector<std::vector<int>> hard_clusters; // output 1 for clustering
  cst.clustering(embeddings1, binarized, hard_clusters);
  assert(hard_clusters.size() == binarized.size());
  assert(hard_clusters[0].size() == binarized[0][0].size());
  std::vector<std::vector<float>> inactive_speakers(binarized.size(),
                                                    std::vector<float>(binarized[0][0].size(), 0.0));
  for (size_t i = 0; i < binarized.size(); ++i)
  {
    for (size_t j = 0; j < binarized[0].size(); ++j)
    {
      for (size_t k = 0; k < binarized[0][0].size(); ++k)
      {
        inactive_speakers[i][k] += binarized[i][j][k];
      }
    }
  }
  for (size_t i = 0; i < inactive_speakers.size(); ++i)
  {
    for (size_t j = 0; j < inactive_speakers[0].size(); ++j)
    {
      if (abs(inactive_speakers[i][j]) < std::numeric_limits<double>::epsilon())
        hard_clusters[i][j] = -2;
    }
  }

  SlidingWindow activations_frames;
  auto discrete_diarization = reconstruct(segmentations, res_frames,
                                          hard_clusters, count_data, count_frames, activations_frames);

  float diarization_segmentation_min_duration_off = 0.5817029604921046; // see SegmentModel
  auto diarization = to_annotation(discrete_diarization,
                                   activations_frames, 0.5, 0.5, 0.0,
                                   diarization_segmentation_min_duration_off);

  std::cout << "----------------------------------------------------" << std::endl;
  auto diaRes = diarization.finalResult();
  for (const auto &dr : diaRes)
  {
    std::cout << "[" << dr.start << " -- " << dr.end << "]"
              << " --> Speaker_" << dr.label << std::endl;
  }
  std::cout << "----------------------------------------------------" << std::endl;
  // std::map<int, std::vector<Annotation::Result>> mergedResults;
  std::vector<std::vector<float>> audioSegments;
  mergeSegments(diaRes, input_wav, mergedResults, audioSegments, allSegment);
  // auto merged_audio= mergeAudio(input_wav,merged_result);
  // printf("size of merged audio %d\n", merged_audio.size());
  return audioSegments;
}

#endif // STT_ENGINE_H
