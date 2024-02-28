

#include <stdio.h>
#include <chrono>
#include <string>
#include <vector>
#include <regex>

#include "ai_vad.h"
#include "speaker_diarization/frontend/wav.h"
#include <boost/filesystem.hpp>
#include <filesystem> // For directory iteration
#include "speaker_id/speaker/speaker_engine.h"
#include "speaker_diarization/speaker_diarization.h"


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

void saveOneDimArrayToBinaryFile(const std::vector<int> &array, const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cout << "Failed to open file for writing." << std::endl;
        return;
    }

    // 获取数组的长度
    int size = array.size();

    // 将一维数组写入文件
    file.write(reinterpret_cast<const char *>(array.data()), size * sizeof(int));

    file.close();
    std::cout << "One-dimensional array saved to binary file: " << filename << std::endl;
}
void saveArrayToBinaryFile(const std::vector<std::vector<std::vector<double>>> &array, const std::string &filename)
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
    int dim3 = (dim2 > 0) ? array[0][0].size() : 0;

    // 写入数组数据
    for (int i = 0; i < dim1; ++i)
    {
        for (int j = 0; j < dim2; ++j)
        {
            for (int k = 0; k < dim3; ++k)
            {
                double value = array[i][j][k];
                file.write(reinterpret_cast<const char *>(&value), sizeof(double));
            }
        }
    }

    file.close();
    std::cout << "Array saved to binary file: " << filename << std::endl;
}
void saveFloatArrayToBinaryFile(const std::vector<std::vector<std::vector<float>>> &array, const std::string &filename)
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
    int dim3 = (dim2 > 0) ? array[0][0].size() : 0;

    // 写入数组数据
    for (int i = 0; i < dim1; ++i)
    {
        for (int j = 0; j < dim2; ++j)
        {
            for (int k = 0; k < dim3; ++k)
            {
                float value = array[i][j][k];
                file.write(reinterpret_cast<const char *>(&value), sizeof(float));
            }
        }
    }

    file.close();
    std::cout << "Array saved to binary file: " << filename << std::endl;
}

bool numericStringCompare(const std::string &s1, const std::string &s2)
{
    // Find the position of "audio_" in the strings
    size_t pos1 = s1.find("audio_");
    size_t pos2 = s2.find("audio_");

    // Extract numeric parts after "audio_"
    int num1 = std::stoi(s1.substr(pos1 + 6)); // 6 is the length of "audio_"
    int num2 = std::stoi(s2.substr(pos2 + 6)); // 6 is the length of "audio_"

    return num1 < num2;
}

// 初始化模型并返回已初始化的模型对象
SegmentModel initializeModel()
{
    // 模型文件路径
    const std::string segmentModel = "./bin/seg_model.onnx";
    // 初始化模型
    SegmentModel mm(segmentModel);
    return mm;
}

// 运行分割模型并返回模型输出
std::vector<std::vector<std::vector<double>>> runSegmentationModel(SegmentModel &mm, const std::string &filename,
                                                                   std::vector<std::pair<double, double>> &segments,
                                                                   std::vector<std::vector<std::vector<float>>> &segmentations,
                                                                   SlidingWindow &res_frames,
                                                                   std::vector<float> &input_wav)
{
    // 从音频文件中读取数据
    wav::WavReader wav_reader(filename);
    int num_channels = wav_reader.num_channels();
    int bits_per_sample = wav_reader.bits_per_sample();
    int sample_rate = wav_reader.sample_rate();
    const float *audio = wav_reader.data();
    int num_samples = wav_reader.num_samples();
    input_wav.resize(num_samples);
    for (int i = 0; i < num_samples; ++i)
    {
        input_wav[i] = audio[i] * 1.0f / 32768.0;
    }

    // 运行分割模型
    // SlidingWindow res_frames;
    segmentations = mm.slide(input_wav, res_frames);
    auto segment_data = res_frames.data();
    for (auto seg : segment_data)
    {
        segments.emplace_back(std::make_pair(seg.start, seg.end));
    }
    auto binarized = mm.binarize_swf(segmentations, false);

    return binarized;
}

std::vector<float> countSpeakerProbabilities(const std::vector<std::vector<std::vector<double>>> &binarized)
{
    int num_chunk = binarized.size();
    int num_frame = binarized[0].size();
    int num_speaker = binarized[0][0].size();

    std::vector<int> speaker_counts(num_speaker, 0);
    int frame_total = num_chunk * num_frame;
    printf("frame_total%d \n",frame_total);

    // 计算每个说话者在所有 chunk 中为 1 的次数
    for (int i = 0; i < num_chunk; ++i)
    {
        for (int j = 0; j < num_frame; ++j)
        {
            for (int k = 0; k < num_speaker; ++k)
            {
                speaker_counts[k] += binarized[i][j][k];
            }
        }
    }

    // 将次数转换为概率
    std::vector<float> speaker_probabilities(num_speaker, 0.0f);
    for (int k = 0; k < num_speaker; ++k)
    {
        speaker_probabilities[k] = static_cast<float>(speaker_counts[k]) / frame_total;
        printf("speaker_probabilities%f \n",speaker_probabilities[k]);
    }

    return speaker_probabilities;
}

bool shouldProcess(const std::vector<float> &speaker_counts, float threshold)
{
    int count_above_threshold = 0;

    for (int i = 0; i < speaker_counts.size(); ++i)
    {
        if (speaker_counts[i] > threshold)
        {
            count_above_threshold++;
            if (count_above_threshold >= 2)
            {
                return true; // 如果有两个以上的数字大于阈值，则处理数据
            }
        }
    }

    return false; // 如果不满足条件，则不处理数据
}

void mergeSegments(const std::vector<Annotation::Result>& results, const std::vector<float>& input_wav,
                                                            std::map<int, std::vector<Annotation::Result>>& mergedResults,
                                                                std::vector<std::vector<float>>& audioSegments) {
    // std::map<int, std::vector<Annotation::Result>> mergedResults;

    // Group segments by label
    for (const auto& result : results) {
        mergedResults[result.label].push_back(result);
    }
    // std::vector<std::vector<float>> audioSegments;

    // Merge segments within each label and extract audio segments
    for (auto& pair : mergedResults) {
        int label = pair.first;
        std::vector<Annotation::Result>& segmentList = pair.second;

        std::vector<Annotation::Result> mergedSegments;
        // mergedSegments.push_back(segmentList[0]);

        // Iterate over segments and merge
        std::vector<float> audioSegment;
        double mergedEnd = -1.0; // Initialize with an invalid value
        for (const auto& segment : segmentList) {
            // Calculate the start and end samples based on the sample rate
            int startSample = static_cast<int>(segment.start * 16000);
            int endSample = static_cast<int>(segment.end * 16000);

            // Extract the audio segment based on the start and end samples
            std::vector<float> segmentAudio(input_wav.begin() + startSample, input_wav.begin() + endSample);

            if (mergedEnd != -1.0 && segment.start - mergedEnd <= 1.5) {
                // Merge segments if the time difference is less than or equal to 1.5 seconds
                audioSegment.insert(audioSegment.end(), segmentAudio.begin(), segmentAudio.end());
                mergedSegments.back().end = segment.end;
            } else {
                // Push the previous audio segment if it exists
                if (!audioSegment.empty()) {
                    audioSegments.push_back(audioSegment);
                    audioSegment.clear();
                }
                // Start a new audio segment
                audioSegment = segmentAudio;
                mergedSegments.push_back(segment);

            }

            // Update the merged end time
            mergedEnd = segment.end;
        }

        // Push the last audio segment
        if (!audioSegment.empty()) {
            audioSegments.push_back(audioSegment);
        }

        // Store the audio segments in the result map
        pair.second = mergedSegments;
    }
}
// Function to print ID, start, and end for all elements in the map
void printMap(const std::map<int, std::vector<Annotation::Result>>& resultMap) {
    for (const auto& pair : resultMap) {
        int label = pair.first;
        const std::vector<Annotation::Result>& segmentList = pair.second;
        for (const auto& segment : segmentList) {
            std::cout << "ID: " << label << ", Start: " << segment.start << ", End: " << segment.end << std::endl;
        }
    }
}

std::vector<int16_t> floatToShort(const std::vector<float>& floatVector) {
    // Initialize the resulting int16_t vector
    std::vector<int16_t> shortVector;
    shortVector.reserve(floatVector.size());

    // Determine the scaling factor to convert floats to int16_t
    float scale = static_cast<float>(std::numeric_limits<int16_t>::max());

    // Convert each float value to int16_t
    for (float value : floatVector) {
        // Convert float to int16_t and ensure it is within the range [-32768, 32767]
        int16_t intValue = static_cast<int16_t>(std::round(value * scale));
        if (intValue > std::numeric_limits<int16_t>::max()) {
            intValue = std::numeric_limits<int16_t>::max();
        } else if (intValue < std::numeric_limits<int16_t>::min()) {
            intValue = std::numeric_limits<int16_t>::min();
        }

        // Store the converted value in the resulting vector
        shortVector.push_back(intValue);
    }

    return shortVector;
}
// Function to save audio segments to files
void saveAudioSegments(const std::vector<std::vector<float>>& audioSegments, const std::map<int, std::vector<Annotation::Result>>& mergedResults,std::string basePath ,const int index) {
    int segmentIndex = 0;
    for (const auto& pair : mergedResults) {
        int label = pair.first;
        const std::vector<Annotation::Result>& segmentList = pair.second;

        for (size_t i = 0; i < segmentList.size(); ++i) {
            const Annotation::Result& segment = segmentList[i];

            

            // Generate filename based on label and start/end times
            std::string filename = basePath + "audio" +std::to_string(index)+"_id_"+ std::to_string(label) + "_" + std::to_string(static_cast<int>(segment.start)) + "_" + std::to_string(static_cast<int>(segment.end)) + ".wav";
            std::vector<int16_t> shortVector = floatToShort(audioSegments[segmentIndex]);

            // Save audio segment to file (assuming you have a saveToFile function)
            saveWavFile(filename, shortVector, 1, 16000, 16);

            segmentIndex++;
        }
    }
}

std::vector<std::vector<float>> generateDiarization(SegmentModel &mm,
                                                    const std::vector<float> &input_wav,
                                                    const int index,
                                                    const std::vector<std::vector<std::vector<double>>> &binarized,
                                                    const std::vector<std::vector<std::vector<float>>> &segmentations,
                                                    const std::vector<std::pair<double, double>> &segments,
                                                    SlidingWindow &res_frames,
                                                    size_t embedding_batch_size,
                                                    const std::shared_ptr<wespeaker::SpeakerEngine> &speaker_engine,
                                                    std::map<int, std::vector<Annotation::Result>>& mergedResults)
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
    auto count_data = mm.speaker_count(segmentations, binarized,
                                       pre_frame, count_frames, num_samples);
    // 计算最小帧数
    size_t min_num_frames = ceil(binarized[0].size() * min_num_samples / (duration * 16000));

    // 清理分割结果
    auto clean_segmentations = Helper::cleanSegmentations(binarized);
    assert(binarized.size() == clean_segmentations.size());

    // 生成embedding
    for (size_t i = 0; i < binarized.size(); ++i)
    {
        auto chunkData = mm.crop(input_wav, segments[i]);
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
                auto embedding = getEmbedding(speaker_engine, batchData, batchMasks);
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
        auto embedding = getEmbedding(speaker_engine, batchData, batchMasks);
        for (auto &a : embedding)
        {
            embeddings.push_back(std::move(a));
        }
    }
    printf("finish embedding process, size%d\n",embeddings.size());
    auto embeddings1 = Helper::rearrange_up( embeddings, binarized.size());

    Cluster cst;
    std::vector<std::vector<int>> hard_clusters; // output 1 for clustering
    cst.clustering(embeddings1,binarized,hard_clusters);
    assert( hard_clusters.size() == binarized.size());
    assert( hard_clusters[0].size() == binarized[0][0].size());
    std::vector<std::vector<float>> inactive_speakers( binarized.size(),
            std::vector<float>( binarized[0][0].size(), 0.0));
    for( size_t i = 0; i < binarized.size(); ++i )
    {
        for( size_t j = 0; j < binarized[0].size(); ++j )
        {
            for( size_t k = 0; k < binarized[0][0].size(); ++k )
            {
                inactive_speakers[i][k] += binarized[i][j][k];
            }
        }
    }   
    for( size_t i = 0; i < inactive_speakers.size(); ++i )
    {
        for( size_t j = 0; j < inactive_speakers[0].size(); ++j )
        {
            if( abs( inactive_speakers[i][j] ) < std::numeric_limits<double>::epsilon())
                hard_clusters[i][j] = -2;
        }
    }

    SlidingWindow activations_frames;
    auto discrete_diarization = reconstruct( segmentations, res_frames, 
            hard_clusters, count_data, count_frames, activations_frames );


    float diarization_segmentation_min_duration_off = 0.5817029604921046; // see SegmentModel
    auto diarization = to_annotation(discrete_diarization, 
            activations_frames, 0.5, 0.5, 0.0, 
            diarization_segmentation_min_duration_off);

    std::cout<<"----------------------------------------------------"<<std::endl;
    auto diaRes = diarization.finalResult();
    for( const auto& dr : diaRes )
    {
        std::cout<<"["<<dr.start<<" -- "<<dr.end<<"]"<<" --> Speaker_"<<dr.label<<std::endl;
    }
    std::cout<<"----------------------------------------------------"<<std::endl;
    // std::map<int, std::vector<Annotation::Result>> mergedResults;
    std::vector<std::vector<float>> audioSegments;
    mergeSegments(diaRes,input_wav,mergedResults,audioSegments);
    // auto merged_audio= mergeAudio(input_wav,merged_result);
    printMap(mergedResults);
    // printf("size of merged audio %d\n", merged_audio.size());
    return audioSegments;
}

// Function to extract file index from filename
int extractFileIndex(const std::string& filename) {
    // Find the position of the last '/'
    size_t slashPos = filename.find_last_of('/');
    if (slashPos == std::string::npos) {
        return -1; // No '/' found
    }

    // Find the position of "audio"
    size_t audioPos = filename.find("audio", slashPos);
    if (audioPos == std::string::npos) {
        return -1; // "audio" not found after '/'
    }

    // Extract the substring after "audio"
    std::string indexStr = filename.substr(audioPos + 5); // "audio" has 5 characters
    // Find the position of ".wav"
    size_t dotPos = indexStr.find(".wav");
    if (dotPos == std::string::npos) {
        return -1; // ".wav" not found
    }

    // Extract the substring before ".wav"
    indexStr = indexStr.substr(0, dotPos);

    // Convert the substring to an integer
    try {
        int index = std::stoi(indexStr);
        return index;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid index: " << indexStr << std::endl;
        return -1; // Failed to convert to integer
    } catch (const std::out_of_range& e) {
        std::cerr << "Index out of range: " << indexStr << std::endl;
        return -1; // Index out of range
    }
}

 

bool copyFile(const std::string& sourcePath, const std::string& destinationPath) {
    std::ifstream sourceFile(sourcePath, std::ios::binary);
    if (!sourceFile.is_open()) {
        std::cerr << "Failed to open source file: " << sourcePath << std::endl;
        return false;
    }

    std::ofstream destinationFile(destinationPath, std::ios::binary);
    if (!destinationFile.is_open()) {
        std::cerr << "Failed to open destination file: " << destinationPath << std::endl;
        sourceFile.close();
        return false;
    }

    // Copy file contents
    destinationFile << sourceFile.rdbuf();

    // Close files
    sourceFile.close();
    destinationFile.close();

    std::cout << "File copied successfully." << std::endl;
    return true;
}


int main()
{

    SegmentModel mm = initializeModel();
    std::string audio_directory = "./test_audio/8speaker_vad/";
    std::string result_directory = "./test_audio/8speaker_segment/";   //diary_segment

    std::vector<std::string> filenames;

    // std::vector<std::string> model_paths = {"./bin/3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx"};
    std::vector<std::string> model_paths = {"./bin/voxceleb_resnet34_LM.onnx"};

    int embedding_size = 256;
    int feat_dim = 80;
    int SamplesPerChunk = 16000;

    auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
        model_paths, feat_dim, 16000,
        embedding_size, SamplesPerChunk);
    std::vector<float> last_embs(embedding_size, 0);
    std::vector<float> current_embs(embedding_size, 0);

    std::vector<std::string> file_paths;
    boost::filesystem::path dir_path(audio_directory);
    if (boost::filesystem::is_directory(dir_path))
    {
        boost::filesystem::directory_iterator end_itr;
        for (boost::filesystem::directory_iterator itr(dir_path); itr != end_itr; ++itr)
        {
            if (boost::filesystem::is_regular_file(itr->status()))
            {
                file_paths.push_back(itr->path().string());
            }
        }
    }
    else
    {
        std::cerr << "Error: " << audio_directory << " is not a directory." << std::endl;
        return 1;
    }

    std::sort(file_paths.begin(), file_paths.end());
    for (const std::string &filepath : file_paths)
    {

        int index = extractFileIndex(filepath);
        printf("file index %d",index);

        std::vector<std::pair<double, double>> segments;
        std::vector<std::vector<std::vector<float>>> segmentations;
        SlidingWindow res_frames;
        std::vector<float> input_wav; // 创建空的input_wav向量

        printf("filepath %s \n", filepath.c_str());
        auto binarized = runSegmentationModel(mm, filepath, segments, segmentations, res_frames, input_wav);
        int num_chunk = binarized.size();
        int num_frame = binarized[0].size();
        printf("num_chunk %d\n", num_chunk);
        int frame_total = num_chunk * num_frame;

        std::vector<float> speaker_prob = countSpeakerProbabilities(binarized);
        for (int i = 0; i < speaker_prob.size(); ++i)
        {
            std::cout << "Speaker " << i << ": " << speaker_prob[i] << std::endl;
        }
        size_t min_num_samples = 640;
        size_t embedding_batch_size = 32;


        if (num_chunk <= 1)
        {
            if (shouldProcess(speaker_prob, 0.3))
            {
                std::map<int, std::vector<Annotation::Result>> mergedResults;
                auto audio_chunk = generateDiarization(mm, input_wav,index, binarized, segmentations, segments,res_frames, embedding_batch_size, speaker_engine,mergedResults);
                saveAudioSegments(audio_chunk,mergedResults,result_directory,index);

                std::cout << "Processing...\n";
                // 在这里添加你想要执行的处理逻辑
            }
            else
            {
                std::string respath=result_directory+"audio"+std::to_string(index)+".wav";
                copyFile(filepath,respath);
                std::cout << "Skipping...\n";
            }
        }
        else
        {
            if (shouldProcess(speaker_prob, 0.2))
            {
                std::map<int, std::vector<Annotation::Result>> mergedResults;

                auto audio_chunk = generateDiarization(mm, input_wav, index, binarized, segmentations, segments,res_frames, embedding_batch_size, speaker_engine,mergedResults);
                saveAudioSegments(audio_chunk,mergedResults,result_directory,index);

                std::cout << "Processing...\n";
                // 在这里添加你想要执行的处理逻辑
            }
            else
            {
                std::string respath=result_directory+"audio"+std::to_string(index)+".wav";
                copyFile(filepath,respath);
                std::cout << "Skipping...\n";
            }
        }
    }

    return 0;
}
