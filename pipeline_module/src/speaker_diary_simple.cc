#include <stdio.h>
#include <chrono>
#include <string>
#include <vector>

#include "ai_vad.h" // ai based vad

// #include "alsa_cq_buffer.h" // ALSAstt_core

#include "speaker_diary_simple.h"

// #include "tts.h" //tts

// std::string model_path ="./bin/voxceleb_resnet34_LM.onnx";
// int embedding_size=256;

// int32_t init_online_audio(online_params *params)
// {

//     params->is_running = true;

//     online_audio audio_buffer;
//     audio_buffer.pcmf32_new = std::vector<float>(params->n_samples_30s, 0.0f);
//     audio_buffer.CQ_buffer.resize(sample_rate * 30);
//     params->audio = audio_buffer;
//     float value;

//     return 0;
// }

int16_t float32ToInt16(float value)
{
    return static_cast<int16_t>(std::round(value * 32767.0f));
}

void convertFloat32ToInt16(const float *floatData, int16_t *intData, size_t numSamples)
{
    for (size_t i = 0; i < numSamples; ++i)
    {
        intData[i] = float32ToInt16(floatData[i]);
    }
}

void saveArrayToBinaryFile(const std::vector<std::vector<float>>& array, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "Failed to open file for writing." << std::endl;
        return;
    }

    // 获取数组的维度
    int dim1 = array.size();
    int dim2 = (dim1 > 0) ? array[0].size() : 0;
    // int dim3 = (dim2 > 0) ? array[0][0].size() : 0;

    // 写入数组数据
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            // for (int k = 0; k < dim3; ++k) {
                float value = array[i][j];
                file.write(reinterpret_cast<const char*>(&value), sizeof(float));
            // }
        }
    }

    file.close();
    std::cout << "Array saved to binary file: " << filename << std::endl;
}


// WAV 文件头结构
struct WavHeader {
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
void saveWavFile(const std::string& filename, const std::vector<int16_t>& data, uint16_t numChannels, uint32_t sampleRate, uint16_t bitsPerSample) {
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
    file.write(reinterpret_cast<const char*>(&header), sizeof(WavHeader));

    // 写入音频数据
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(int16_t));

    // 关闭文件
    file.close();

    std::cout << "WAV 文件保存成功：" << filename << std::endl;
}
int main()
{

    // online_params params;
 
    std::vector<std::string> model_paths;
    std::string onnx_model_path ="./bin/voxceleb_resnet34_LM.onnx";
    model_paths.push_back(onnx_model_path);

    int embedding_size=256;
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


    ///////////////// READ WAV /////////////////////////
    std::string audio_path="./bin/output.wav";
    auto data_reader = wenet::ReadAudioFile(audio_path);
    int16_t *enroll_data_int16 = const_cast<int16_t *>(data_reader->data());
    int samples = data_reader->num_sample();
    printf("samples%d\n",samples);
    std::vector<float> enroll_data_flt32(samples);
    for (int i = 0; i < samples; i++)
    {
        enroll_data_flt32[i] = static_cast<float>(enroll_data_int16[i]) / 32768;
    }
    int seg_start = -1;
    int seg_end = -1;
    std::vector<std::vector<float>> chunk_enroll_embs;
    std::string baseFilename = "audio_output/audio";
    std::string fileExtension = ".wav";
    int file_count=0;
    for (int j = 0; j < samples; j += test_window_samples)
    {

        std::vector<float> window_chunk{&enroll_data_flt32[0] + j, &enroll_data_flt32[0] + j + test_window_samples};

        int32_t vad_state = ai_vad.predict(window_chunk);
        if (vad_state == 2)
        {
            seg_start = j - test_window_samples;
        }
        if (vad_state == 3)
        {
            seg_end = j;
            if (seg_start != -1)
            {
                std::vector<int16_t> vad_chunk_int16{&enroll_data_int16[seg_start], &enroll_data_int16[seg_end]};
                std::vector<float> chunk_emb(embedding_size, 0);
                std::vector<double> chunk_emb_double(embedding_size, 0);
                std::string filename = baseFilename + std::to_string(file_count) + fileExtension;
                file_count+=1;
                saveWavFile(filename, vad_chunk_int16, 1, 16000, 16);

                // speaker_engine->ExtractEmbedding(vad_chunk_int16.data(),
                //                                  vad_chunk_int16.size(),
                //                                  &chunk_emb);
                
                // for(int i=0;i<chunk_emb.size();i++)
                // {
                //     chunk_emb_double[i]=static_cast<double>(chunk_emb[i]);
                // }

                // chunk_enroll_embs.push_back(chunk_emb);

          
            }
        }
    }
    // printf("chunk_enroll_embs size(%d,%d)",chunk_enroll_embs.size(),chunk_enroll_embs[0].size());
    // saveArrayToBinaryFile(chunk_enroll_embs,"speaker_embedding.bin");
    // auto embeddings1 = Helper::rearrange_up( chunk_enroll_embs, 17 );
    // std::cout << "embeddings1 result size" << embeddings1.size()<<","<<embeddings1[0].size()<<","<<embeddings1[0][0].size() << std::endl;

 
    // std::vector<std::vector<std::vector<double>>> segmentations;
    // /////////////////////////////////////////////////
    // Cluster cst;
    // std::vector<std::vector<int>> hard_clusters; // output 1 for clustering
    // cst.clustering(embeddings1, segmentations, hard_clusters);
    // printf("hard_clusters(%d,%d) ",hard_clusters.size(),hard_clusters[1].size());

    // for( size_t i = 0; i < hard_clusters.size(); ++i )
    // {
    //     for( size_t j = 0; j < hard_clusters[0].size(); ++j )
    //     {
    //        printf("hard_clusters[i][j]%d\n",hard_clusters[i][j]);
    //     }
    // }
}

 