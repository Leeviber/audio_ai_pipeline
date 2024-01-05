#ifndef SPEAKER_DIARY_SIMPLE

#include <stdio.h>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include "speaker_id/speaker/speaker_engine.h"
#include "speaker_id/frontend/wav.h"
#include "speaker_diarization/speaker_diarization.h"
#include "sherpa_stt/offline-recognizer.h"
#include "sherpa_stt/offline-model-config.h"
#include "aivad/ai_vad.h" // ai based vad


#include "speaker_diary_simple.h"


void saveArrayToBinaryFile(const std::vector<std::vector<float>> &array, const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cout << "Failed to open file for writing." << std::endl;
        return;
    }

    // 获取数组的维度
    int dim1 = array.size();
    int dim2 = (dim1 > 0) ? array[0].size() : 0;
    // int dim3 = (dim2 > 0) ? array[0][0].size() : 0;

    // 写入数组数据
    for (int i = 0; i < dim1; ++i)
    {
        for (int j = 0; j < dim2; ++j)
        {
            // for (int k = 0; k < dim3; ++k) {
            float value = array[i][j];
            file.write(reinterpret_cast<const char *>(&value), sizeof(float));
            // }
        }
    }

    file.close();
    std::cout << "Array saved to binary file: " << filename << std::endl;
}
struct Embed_Segment
{
    int start;
    int end;
    std::string text;

    Embed_Segment(int s, int e, const std::string &t) : start(s), end(e), text(t) {}
};

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

std::vector<int> mergeAndRenumberNumbers(const std::vector<int> &numbers)
{
    std::unordered_map<int, int> index_map;
    std::vector<int> unique_numbers;

    // 获取唯一的数字并保留其首次出现的顺序
    for (int num : numbers)
    {
        if (index_map.find(num) == index_map.end())
        {
            index_map[num] = static_cast<int>(unique_numbers.size());
            unique_numbers.push_back(num);
        }
    }

    // 根据唯一的数字及其首次出现的顺序进行重新编号
    std::vector<int> result;
    for (int num : numbers)
    {
        result.push_back(index_map[num]);
    }

    return result;
}

void generateTimestamps(const std::vector<int> &renumbered_numbers, const std::vector<Embed_Segment> &embedd_segments)
{

    std::cout << "####################### Speaker diarization Result ###################### " << std::endl;

    for (int i = 0; i < embedd_segments.size(); i++)
    {
        std::cout << "Speaker ID: " << renumbered_numbers[i] << " start:" << embedd_segments[i].start / 16000.0 << ", end: " << embedd_segments[i].end / 16000.0;
        std::cout << " Text:" << embedd_segments[i].text << std::endl;
    }
}

int main()
{

    bool enable_stt = true;

    // online_params params;

    std::vector<std::string> model_paths;
#ifdef USE_NPU
    std::string rknn_model_path = "./bin/Id1_resnet34_LM_main_part.rknn";
    std::string onnx_model_path = "./bin/Id2_resnet34_LM_post.onnx";
    model_paths.push_back(rknn_model_path);
    model_paths.push_back(onnx_model_path); 

#else
    std::string onnx_model_path = "./bin/voxceleb_resnet152_LM.onnx";
    // std::string onnx_model_path = "./bin/voxceleb_CAM++_LM.onnx";

    model_paths.push_back(onnx_model_path);

#endif
    int embedding_size = 256;
    int feat_dim = 80;
    int SamplesPerChunk = 32000;
    auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
        model_paths, feat_dim, 16000,
        embedding_size, SamplesPerChunk);

    std::vector<float> last_embs(embedding_size, 0);
    std::vector<float> current_embs(embedding_size, 0);

    int sample_chunk_ms = 1000;

    std::vector<float> enroll_embs(embedding_size, 0);

    ///////////////// Init VAD /////////////////////////

    int vad_begin = 0;
    int vad_end = 0;

    int32_t vad_activate_sample = (16000 * 500) / 1000;
    int32_t vad_silence_sample = (16000 * 0) / 1000;

    std::string path = "./bin/silero_vad.onnx";
    int test_sr = 16000;
    int test_frame_ms = 96;
    float test_threshold = 0.85f;
    int test_min_silence_duration_ms = 200;
    int test_speech_pad_ms = 0;
    int test_window_samples = test_frame_ms * (test_sr / 1000);

    VadIterator ai_vad(
        path, test_sr, test_frame_ms, test_threshold,
        test_min_silence_duration_ms, test_speech_pad_ms);

    ///////////////////////////////////////////////////////

    //// Init Sherpa STT module //////////

    std::string tokens = "./bin/tokens.txt";
    std::string encoder_filename = "./bin/encoder-epoch-30-avg-4.int8.onnx";
    std::string decoder_filename = "./bin/decoder-epoch-30-avg-4.int8.onnx";
    std::string joiner_filename = "./bin/joiner-epoch-30-avg-4.int8.onnx";

    sherpa_onnx::OfflineTransducerModelConfig transducer;
    transducer.encoder_filename = encoder_filename;
    transducer.decoder_filename = decoder_filename;
    transducer.joiner_filename = joiner_filename;

    sherpa_onnx::OfflineModelConfig model_config;
    model_config.tokens = tokens;
    model_config.transducer = transducer;

    sherpa_onnx::OfflineRecognizerConfig config;
    config.model_config = model_config;
    if (!config.Validate())
    {
        fprintf(stderr, "Errors in config!\n");
        return -1;
    }
    fprintf(stdout, "Creating recognizer ...\n");
    sherpa_onnx::OfflineRecognizer recognizer(config);
    std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>> ss;
    std::vector<sherpa_onnx::OfflineStream *> ss_pointers;
    ////////////////////////////////////////////////////

    ///////////////// READ and Wrirte WAV /////////////////////////
    std::string audio_path = "./bin/team_talk_chinese.wav";
    auto data_reader = wenet::ReadAudioFile(audio_path);
    int16_t *enroll_data_int16 = const_cast<int16_t *>(data_reader->data());
    int samples = data_reader->num_sample();
    std::vector<float> enroll_data_flt32(samples);
    for (int i = 0; i < samples; i++)
    {
        enroll_data_flt32[i] = static_cast<float>(enroll_data_int16[i]) / 32768;
    }

    int seg_start = -1;
    int seg_end = -1;
    std::vector<std::vector<float>> chunk_enroll_embs;
    std::vector<std::vector<double>> double_chunk_enroll_embs;

    std::string basePath = "test_audio/audio_output/";
    std::string prefix = "audio";

    std::string fileExtension = ".wav";
    int file_count = 0;
    ////////////////////////////////////////////////////

    ///////////////// Process the audio for diarization /////////////////////////

    // 创建一个存储嵌入式段落的向量
    std::vector<Embed_Segment> embedd_segment;

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 计算音频数据块的数量
    int audio_chunk = samples / test_window_samples;

    // 对每个音频数据块进行处理
    for (int j = 0; j < audio_chunk; j++)
    {
        // 提取当前窗口的音频数据
        std::vector<float> window_chunk{&enroll_data_flt32[0] + j * test_window_samples, &enroll_data_flt32[0] + j * test_window_samples + test_window_samples};

        // 使用VAD（Voice Activity Detection）模块进行语音状态预测
        int32_t vad_state = ai_vad.predict(window_chunk);

        // 当检测到语音开始
        if (vad_state == 2)
        {
            seg_start = (j-1) * test_window_samples; // 记录语音起始位置
        }

        // 当检测到语音结束
        if (vad_state == 3)
        {
            seg_end = j * test_window_samples; // 记录语音结束位置

            // 如果已经有起始位置，进行处理
            if (seg_start != -1)
            {
                // if(seg_end-seg_start<=8000)
                // {
                //     continue;
                // }
                // 提取VAD检测到的语音片段
                std::vector<int16_t> vad_chunk_int16{&enroll_data_int16[seg_start], &enroll_data_int16[seg_end]};
                std::vector<float> vad_chunk_fp32{&enroll_data_flt32[seg_start], &enroll_data_flt32[seg_end]};

                // 创建一个STT（语音到文本）流并进行音频输入
                auto s = recognizer.CreateStream();
                s->AcceptWaveform(16000, vad_chunk_fp32.data(), vad_chunk_fp32.size());
                ss.push_back(std::move(s));
                ss_pointers.push_back(ss.back().get());
                recognizer.DecodeStreams(ss_pointers.data(), 1);

                // 获取STT识别结果并将嵌入式段落存储到向量中
                const std::string text = ss[0]->GetResult().text;
                embedd_segment.push_back(Embed_Segment(seg_start, seg_end, text));

                // 清空STT流相关的数据结构
                ss.clear();
                ss_pointers.clear();

                // 保存语音片段到WAV文件
                std::vector<float> chunk_emb(embedding_size, 0);
                std::vector<double> chunk_emb_double(embedding_size, 0);
                std::string filename = basePath + prefix + std::to_string(file_count) + fileExtension;
                file_count += 1;
                saveWavFile(filename, vad_chunk_int16, 1, 16000, 16);

                // 提取语音片段的嵌入式特征
                speaker_engine->ExtractEmbedding(vad_chunk_int16.data(), vad_chunk_int16.size(), &chunk_emb);

                // 将嵌入式特征转换为双精度类型并存储到向量中
                for (int i = 0; i < chunk_emb.size(); i++)
                {
                    chunk_emb_double[i] = static_cast<double>(chunk_emb[i]);
                }

                double_chunk_enroll_embs.push_back(chunk_emb_double);
            }
        }
    }

    // 创建一个用于聚类的Cluster对象
    Cluster cst;
    std::vector<int> clustersRes; // 存储聚类结果
    cst.custom_clustering(double_chunk_enroll_embs, clustersRes);

    // 合并并重新编号聚类结果的数字
    std::vector<int> merged_renumbered_numbers;
    merged_renumbered_numbers = mergeAndRenumberNumbers(clustersRes);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "diarization 运行时间: " << duration.count() << " 毫秒\n";

    // 生成转录时间戳
    generateTimestamps(merged_renumbered_numbers, embedd_segment);
}
#endif


