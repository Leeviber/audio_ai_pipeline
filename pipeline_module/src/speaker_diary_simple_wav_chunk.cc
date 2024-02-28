#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem> // For directory iteration
#include "speaker_id/speaker/speaker_engine.h"
#include "speaker_id/frontend/wav.h"
#include "speaker_diarization/speaker_diarization.h"
#include <boost/filesystem.hpp>


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

void processAudioFile(const std::string& filepath, wespeaker::SpeakerEngine& speaker_engine, std::vector<std::vector<double>>& embeddings, std::vector<std::string>& filenames) {
    // Read audio file
    printf("Reading file: %s \n",filepath.c_str());
    auto data_reader = wenet::ReadAudioFile(filepath);
    int samples = data_reader->num_sample();
    if (samples < 16000) {
        std::cout << "Skipping file: " << filepath << " (duration less than 1 second)" << std::endl;
        return;
    }

    std::vector<int16_t> audio_data(samples);
    std::copy(data_reader->data(), data_reader->data() + samples, audio_data.begin());

    // Extract embedding
    std::vector<float> embedding;
    speaker_engine.ExtractEmbedding(audio_data.data(), samples, &embedding);

    // Convert embedding to double precision
    std::vector<double> double_embedding(embedding.begin(), embedding.end());

    // Store embedding
    embeddings.push_back(double_embedding);

    // Store filename
    filenames.push_back(filepath);

}
bool numericStringCompare(const std::string& s1, const std::string& s2) {
    // Extract all numeric substrings from the filenames
    std::vector<int> nums1, nums2;
    auto extractNumbers = [](const std::string& s, std::vector<int>& nums) {
        size_t pos = 0;
        while (pos < s.length()) {
            size_t start = s.find_first_of("0123456789", pos);
            if (start == std::string::npos) {
                break; // No more numbers found
            }
            size_t end = s.find_first_not_of("0123456789", start);
            std::string numStr = s.substr(start, end - start);
            nums.push_back(std::stoi(numStr));
            pos = end;
        }
    };
    extractNumbers(s1, nums1);
    extractNumbers(s2, nums2);

    // Compare the extracted numbers
    auto it1 = nums1.begin(), it2 = nums2.begin();
    while (it1 != nums1.end() && it2 != nums2.end()) {
        if (*it1 != *it2) {
            return *it1 < *it2;
        }
        ++it1;
        ++it2;
    }

    // If one filename has more numbers than the other, the one with fewer numbers comes first
    return nums1.size() < nums2.size();
}
int main() {
    // Initialize Speaker Engine
    // std::vector<std::string> model_paths = {"./bin/3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx"};
    std::vector<std::string> model_paths = {"./bin/voxceleb_resnet34_LM.onnx"};

    int embedding_size = 256;
    int feat_dim = 80;
    int SamplesPerChunk = 32000;
    auto speaker_engine = wespeaker::SpeakerEngine(model_paths, feat_dim, 16000, embedding_size, SamplesPerChunk);


    // Directory containing audio files
    std::string audio_directory = "./test_audio/8speaker_segment/";

    // Initialize embeddings vector
    std::vector<std::vector<double>> embeddings;

    // Initialize filenames vector
    std::vector<std::string> filenames;

    // // Iterate over files in directory
    // boost::filesystem::path dir_path(audio_directory);
    // if (!boost::filesystem::is_directory(dir_path)) {
    //     std::cerr << "Error: " << audio_directory << " is not a directory." << std::endl;
    //     return 1;
    // }


    std::vector<std::string> file_paths;
    boost::filesystem::path dir_path(audio_directory);
    if (boost::filesystem::is_directory(dir_path)) {
        boost::filesystem::directory_iterator end_itr;
        for (boost::filesystem::directory_iterator itr(dir_path); itr != end_itr; ++itr) {
            if (boost::filesystem::is_regular_file(itr->status())) {
                file_paths.push_back(itr->path().string());
            }
        }
    } else {
        std::cerr << "Error: " << audio_directory << " is not a directory." << std::endl;
        return 1;
    }

    std::sort(file_paths.begin(), file_paths.end(),numericStringCompare);
    for (const std::string& filepath : file_paths) {
        processAudioFile(filepath, speaker_engine, embeddings, filenames);
    }

    // Perform clustering on embeddings
    Cluster cst;
    std::vector<int> clustersRes;
    cst.custom_clustering(embeddings, clustersRes);
    // 合并并重新编号聚类结果的数字
    std::vector<int> merged_renumbered_numbers;
    merged_renumbered_numbers = mergeAndRenumberNumbers(clustersRes);


    // Output clustering results
    for (size_t i = 0; i < merged_renumbered_numbers.size(); ++i) {
        std::cout << "Audio file " << filenames[i] << " belongs to cluster " << merged_renumbered_numbers[i] << std::endl;
    }

    return 0;
}
