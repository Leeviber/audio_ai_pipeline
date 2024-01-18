#ifndef WAVFILE_H_
#define WAVFILE_H_

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cstdint>
#include <sndfile.h>

struct WavHeader {
  uint8_t RIFF[4] = {'R', 'I', 'F', 'F'};
  uint32_t chunkSize;
  uint8_t WAVE[4] = {'W', 'A', 'V', 'E'};

  // fmt
  uint8_t fmt[4] = {'f', 'm', 't', ' '};
  uint32_t fmtSize = 16;    // bytes
  uint16_t audioFormat = 1; // PCM
  uint16_t numChannels;     // mono
  uint32_t sampleRate;      // Hertz
  uint32_t bytesPerSec;     // sampleRate * sampleWidth
  uint16_t blockAlign = 2;  // 16-bit mono
  uint16_t bitsPerSample = 16;

  // data
  uint8_t data[4] = {'d', 'a', 't', 'a'};
  uint32_t dataSize;
};

// Write WAV file header only
void writeWavHeader(int sampleRate, int sampleWidth, int channels,
                    uint32_t numSamples, std::ostream &audioFile) {
  WavHeader header;
  header.dataSize = numSamples * sampleWidth * channels;
  header.chunkSize = header.dataSize + sizeof(WavHeader) - 8;
  header.sampleRate = sampleRate;
  header.numChannels = channels;
  header.bytesPerSec = sampleRate * sampleWidth * channels;
  header.blockAlign = sampleWidth * channels;
  audioFile.write(reinterpret_cast<const char *>(&header), sizeof(header));

} /* writeWavHeader */


class WavAudioSaver {
public:
    WavAudioSaver(int sampleRate, int channels) : sampleRate_(sampleRate), channels_(channels), audioCount_(0) {}

    void saveAudio(const std::vector<float>& floatData) {
        std::vector<short> intData;
        convertFloat32ToInt16(floatData, intData);

        std::stringstream ss;
        ss << "vadchunk/audio" << audioCount_ << ".wav";
        std::ofstream audioFile(ss.str(), std::ios::binary);

        writeWavHeader(sampleRate_, channels_, 1, static_cast<int32_t>(intData.size()), audioFile);
        audioFile.write(reinterpret_cast<const char*>(intData.data()), sizeof(int16_t) * intData.size());

        std::cout << "Audio data saved to " << ss.str() << std::endl;

        audioCount_++;
    }

private:
    int sampleRate_;
    int channels_;
    int audioCount_;

    void convertFloat32ToInt16(const std::vector<float>& floatData, std::vector<short>& intData) {
        intData.resize(floatData.size());
        for (size_t i = 0; i < floatData.size(); ++i) {
            // 将float32数据转换为int16数据，可以根据需要进行适当的缩放
            intData[i] = static_cast<short>(floatData[i] * 32767.0f);
        }
    }

    void writeWavHeader(int sampleRate, int numChannels, int bitsPerSample, int dataSize, std::ofstream& file) {
        // 编写.wav文件头
        file.write("RIFF", 4);
        int32_t fileSize = 36 + dataSize;
        file.write(reinterpret_cast<char*>(&fileSize), 4);
        file.write("WAVE", 4);
        file.write("fmt ", 4);
        int32_t fmtSize = 16;
        file.write(reinterpret_cast<char*>(&fmtSize), 4);
        int16_t audioFormat = 1; // PCM
        file.write(reinterpret_cast<char*>(&audioFormat), 2);
        file.write(reinterpret_cast<char*>(&numChannels), 2);
        file.write(reinterpret_cast<char*>(&sampleRate), 4);
        int32_t byteRate = sampleRate * numChannels * bitsPerSample / 8;
        file.write(reinterpret_cast<char*>(&byteRate), 4);
        int16_t blockAlign = numChannels * bitsPerSample / 8;
        file.write(reinterpret_cast<char*>(&blockAlign), 2);
        int16_t bitsPerSampleShort = bitsPerSample;
        file.write(reinterpret_cast<char*>(&bitsPerSampleShort), 2);
        file.write("data", 4);
        file.write(reinterpret_cast<char*>(&dataSize), 4);
    }
};

#endif // WAVFILE_H_



int main() {
    // 示例音频数据
    const std::vector<float> floatData = {0.5, -0.2, 0.8, -0.1, 0.3};

    // 创建WavAudioSaver对象
    WavAudioSaver audioSaver(16000, 2);

    // 保存音频数据
    audioSaver.saveAudio(floatData);

    return 0;
}