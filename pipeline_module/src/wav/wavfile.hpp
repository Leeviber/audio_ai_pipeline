#ifndef WAVFILE_H_
#define WAVFILE_H_

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cstdint>
#include <sndfile.h>
#include <cstring>


struct WAVHeader {
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

class WavAudioSaver {
public:
    WavAudioSaver(int sampleRate, int channels) : sampleRate_(sampleRate), channels_(channels), audioCount_(0) {}

    void saveAudio(const std::vector<float>& floatData) {
        std::vector<int16_t> intData;
        convertFloat32ToInt16(floatData, intData);
        std::stringstream ss;
        ss << "vadchunk/audio" << audioCount_ << ".wav";
        writeWAV(ss.str(),intData,16000);
        audioCount_++;

 
    }

    void writeWAV(const std::string& filename, const std::vector<int16_t>& audioData, uint32_t sampleRate) {
        std::ofstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        // 填充WAV文件头
        WAVHeader header;
        std::memcpy(header.chunkId, "RIFF", 4);
        header.chunkSize = audioData.size() * sizeof(int16_t) + sizeof(WAVHeader) - 8;
        std::memcpy(header.format, "WAVE", 4);
        std::memcpy(header.subchunk1Id, "fmt ", 4);
        header.subchunk1Size = 16;
        header.audioFormat = 1; // PCM
        header.numChannels = 1; // 单声道
        header.sampleRate = sampleRate;
        header.byteRate = sampleRate * sizeof(int16_t);
        header.blockAlign = sizeof(int16_t);
        header.bitsPerSample = 16;
        std::memcpy(header.subchunk2Id, "data", 4);
        header.subchunk2Size = audioData.size() * sizeof(int16_t);

        // 写入文件头
        file.write(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

        // 写入音频数据
        file.write(reinterpret_cast<const char*>(audioData.data()), audioData.size() * sizeof(int16_t));

        file.close();
    }


private:
    int sampleRate_;
    int channels_;
    int audioCount_;

    void convertFloat32ToInt16(const std::vector<float>& floatData, std::vector<int16_t>& intData) {
        intData.resize(floatData.size());
        for (size_t i = 0; i < floatData.size(); ++i) {
            // 将float32数据转换为int16数据，可以根据需要进行适当的缩放
            intData[i] = static_cast<int16_t>(floatData[i] * 32767.0f);
        }
    }

};

#endif // WAVFILE_H_



// int main() {
//     // 示例音频数据
//     const std::vector<float> floatData = {0.5, -0.2, 0.8, -0.1, 0.3};

//     // 创建WavAudioSaver对象
//     WavAudioSaver audioSaver(16000, 2);

//     // 保存音频数据
//     audioSaver.saveAudio(floatData);

//     return 0;
// }