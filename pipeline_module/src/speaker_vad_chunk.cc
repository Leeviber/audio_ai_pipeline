#include <iostream>
#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include "speaker_id/speaker/speaker_engine.h"
#include "speaker_id/frontend/wav.h"
#include "aivad/ai_vad.h" // ai based vad
#include "speaker_diarization/speaker_diarization.h"
#include "speaker_diarization/frontend/wav.h"

struct Embed_Segment
{
    int start;
    int end;
    std::string text;

    Embed_Segment(int s, int e, const std::string &t) : start(s), end(e), text(t) {}
};

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


// 运行分割模型并返回模型输出
std::vector<std::vector<std::vector<double>>> runSegmentationModel(SegmentModel &mm, std::vector<float> &input_wav,
                                                                   std::vector<std::pair<double, double>> &segments,
                                                                   std::vector<std::vector<std::vector<float>>> &segmentations,
                                                                   SlidingWindow &res_frames
                                                                   )
{
 
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
                                                                std::vector<std::vector<float>>& audioSegments,
                                                                std::vector<Annotation::Result>& allSegment) {
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

    for (auto& pair : mergedResults) {

        std::vector<Annotation::Result>& segmentList = pair.second;

        for (const auto& segment : segmentList) {
            allSegment.push_back(segment);
            printf("segment id%d\n",segment.label);
            printf("segment start:%f, end:%f\n",segment.start,segment.end);
        
        }
    
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

// to save audio segments to files
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
                                                    const std::vector<std::vector<std::vector<double>>> &binarized,
                                                    const std::vector<std::vector<std::vector<float>>> &segmentations,
                                                    const std::vector<std::pair<double, double>> &segments,
                                                    SlidingWindow &res_frames,
                                                    size_t embedding_batch_size,
                                                    const std::shared_ptr<wespeaker::SpeakerEngine> &speaker_engine,
                                                    std::map<int, std::vector<Annotation::Result>>& mergedResults,
                                                    std::vector<Annotation::Result>& allSegment)
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
    mergeSegments(diaRes,input_wav,mergedResults,audioSegments,allSegment);
    // auto merged_audio= mergeAudio(input_wav,merged_result);
    printMap(mergedResults);
    // printf("size of merged audio %d\n", merged_audio.size());
    return audioSegments;
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

std::string secondsToMinutesAndSeconds(double seconds) {
    int minutes = static_cast<int>(seconds) / 60;
    int remainingSeconds = static_cast<int>(seconds) % 60;
    double fractionalSeconds = seconds - static_cast<int>(seconds);
    int roundedFractionalSeconds = static_cast<int>(fractionalSeconds * 100);

    std::string result = std::to_string(minutes) + "分" + std::to_string(remainingSeconds) + "秒" + std::to_string(roundedFractionalSeconds) + "毫秒";
    return result;
}

int main() {
 

    // std::string audio_file = "./bin/speaker_diarization_long_test/meeting_2.wav";
    std::string audio_file = "./bin/record_bot_processed.wav";

    // Read audio file
    auto data_reader = wenet::ReadAudioFile(audio_file);
    int samples = data_reader->num_sample();
    int16_t *enroll_data_int16 = const_cast<int16_t *>(data_reader->data());
    printf("audio sample%d \n",samples);
    std::vector<float> enroll_data_flt32(samples);
    for (int i = 0; i < samples; i++)
    {
        enroll_data_flt32[i] = static_cast<float>(enroll_data_int16[i]) / 32768;
    }

   
    std::string vad_model_path = "./bin/silero_vad.onnx";
    int vad_sample_rate = 16000;
    int vad_frame_ms = 32;
    float vad_threshold = 0.85;
    int vad_min_silence_duration_ms = 200;
    int vad_speech_pad_ms = 0;
    int vad_window_samples = vad_frame_ms * (vad_sample_rate / 1000);
    VadIterator ai_vad(vad_model_path,vad_frame_ms,vad_threshold,vad_min_silence_duration_ms);

    const std::string segmentModel = "./bin/seg_model.onnx";
    SegmentModel mm(segmentModel);
    size_t min_num_samples = 640;
    size_t embedding_batch_size = 64;
    std::string result_directory = "./test_audio/8speaker_segment/";   //diary_segment
    // std::vector<std::string> model_paths = {"./bin/3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx"};
    std::vector<std::string> model_paths = {"./bin/voxceleb_resnet34_LM.onnx"};

    int embedding_size = 256;
    int feat_dim = 80;
    int SamplesPerChunk = 16000;

    auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
        model_paths, feat_dim, 16000,
        embedding_size, SamplesPerChunk);
     int seg_start = 0;
    int seg_end = 0;
    int file_count = 0;
    std::string basePath = "./test_audio/rand/";
    std::string prefix = "audio";
    std::string fileExtension = ".wav";


    std::vector<std::vector<double>> global_embedding;
    std::vector<Annotation::Result> global_annote;
    std::vector<std::vector<double>> filter_global_embedding;
    std::vector<Annotation::Result> filter_global_annote;

     // Process audio using VAD segmentation
    for (int j = 0; j < samples; j += vad_window_samples) {
        // Extract current window of audio data
        std::vector<float> window_chunk(&enroll_data_flt32[0] + j, &enroll_data_flt32[0] + j + vad_window_samples);

        // Perform VAD prediction
        int32_t vad_state = ai_vad.predict(window_chunk);

        // When speech is detected
        if (vad_state == 2) {
            seg_start = j; // Record speech start position
        }

        // When speech ends
        if (vad_state == 3) {
            seg_end = j; // Record speech end position

            // If there's a valid speech segment
            if (seg_start != 0 && seg_end != 0) {

                int sample_length = seg_end-seg_start;
                printf("sample_length %d\n",sample_length);

                if(sample_length<16000)
                {
                    std::cout << "Skipping (duration less than 1 second)" << std::endl;

                    continue;
                }
                // Extract speech segment
                std::vector<int16_t> vad_chunk_int16{&enroll_data_int16[seg_start], &enroll_data_int16[seg_end]};
                std::vector<float> vad_chunk_fp32{&enroll_data_flt32[seg_start], &enroll_data_flt32[seg_end]};

                        // Calculate start and end timestamps
                double start_time = static_cast<double>(seg_start) / vad_sample_rate;
                double end_time = static_cast<double>(seg_end) / vad_sample_rate;
                start_time = floor(start_time * 100) / 100; // Keep only 2 decimal places
                end_time = floor(end_time * 100) / 100; // Keep only 2 decimal places



                // std::vector<float> vad_chunk_flt32(vad_chunk_int16.size());
                // for (int i = 0; i < vad_chunk_int16.size(); i++)
                // {
                //     vad_chunk_flt32[i] = static_cast<float>(vad_chunk_int16[i]) / 32768;
                // }
 
                std::vector<std::pair<double, double>> segments;
                std::vector<std::vector<std::vector<float>>> segmentations;
                SlidingWindow res_frames;

                auto binarized = runSegmentationModel(mm,vad_chunk_fp32,segments,segmentations,res_frames);
                int num_chunk = binarized.size();
                int num_frame = binarized[0].size();
                int frame_total = num_chunk * num_frame;
                std::vector<float> speaker_prob = countSpeakerProbabilities(binarized);
                for (int i = 0; i < speaker_prob.size(); ++i)
                {
                    std::cout << "Speaker " << i << ": " << speaker_prob[i] << std::endl;
                }

                int num_chunk_thres=1;
                float thres = (num_chunk <= 1) ? 0.3 : 0.2;
 
                if (shouldProcess(speaker_prob, thres))
                {
                    
                    std::vector<Annotation::Result> allLabel;
                    std::map<int, std::vector<Annotation::Result>> mergedResults;
                    auto audio_chunk = generateDiarization(mm, vad_chunk_fp32, binarized, segmentations, segments,res_frames, embedding_batch_size, speaker_engine,mergedResults,allLabel);
                    printf("audio_chunk size%d \n",audio_chunk.size());
                    printf("allLabel size%d \n",allLabel.size());

                    for(int i=0;i<audio_chunk.size();i++)
                    {
                        std::vector<int16_t> shortVector = floatToShort(audio_chunk[i]);

                        std::vector<float> single_emb(embedding_size, 0.0);
                        speaker_engine->ExtractEmbedding(shortVector.data(),
                                                    shortVector.size(),
                                                    &single_emb);
                        std::vector<double> double_embedding(single_emb.begin(), single_emb.end());

                        global_embedding.push_back(double_embedding);
                        Annotation::Result corret_label(allLabel[i].start+start_time,allLabel[i].end+end_time,allLabel[i].label);
                        global_annote.push_back(corret_label);
                    }
              
 
                    std::cout << "Processing...\n";
                    // 在这里添加你想要执行的处理逻辑
                }
                else
                {
 
                    std::vector<float> single_emb(embedding_size, 0.0);
                    speaker_engine->ExtractEmbedding(vad_chunk_int16.data(),
                                                vad_chunk_int16.size(),
                                                &single_emb);

                    std::vector<double> double_embedding(single_emb.begin(), single_emb.end());

                    global_embedding.push_back(double_embedding);
                    Annotation::Result corret_label(start_time,end_time,0);
                    global_annote.push_back(corret_label);
                    printf("here7\n");

                    std::cout << "Skipping...\n";
                }
        
  

        
                // Convert timestamps to string format
                std::string start_time_str = std::to_string(start_time);
                std::string end_time_str = std::to_string(end_time);

                // Save speech segment as WAV file
                // std::string filename = basePath + prefix +  "_" + std::to_string(file_count)+"_" + start_time_str + "_" + end_time_str + fileExtension;
                std::string filename = basePath + prefix + std::to_string(file_count)+ fileExtension;

                printf("filename %s\n",filename.c_str());
                printf("vad_chunk_int16 %d\n",vad_chunk_int16.size());

                // saveWavFile(filename, vad_chunk_int16, 1, 16000, 16);
                file_count++;

                // Reset segment markers
                seg_start = 0;
                seg_end = 0;
            }
        }
    }

    printf(" size of global_embedding%d",global_embedding.size());
    printf("size of global label %d",global_annote.size());

    for (int i = 0; i < global_embedding.size(); ++i)
    {
       
        if (!std::isnan(global_embedding[i][0]))
        { // Assuming all elements in the innermost array are NaN or not NaN
            filter_global_embedding.push_back(global_embedding[i]);
            filter_global_annote.push_back(global_annote[i]);
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
    for (size_t i = 0; i < merged_renumbered_numbers.size(); ++i) {
        std::cout << "Audio start: " << secondsToMinutesAndSeconds(filter_global_annote[i].start) << ", end: "<<secondsToMinutesAndSeconds(filter_global_annote[i].end)<< " belongs to cluster " << merged_renumbered_numbers[i] << std::endl;
    }



    return 0;
}
