 

#include <stdio.h>
#include <chrono>  
#include <string>
#include <vector>

#include "ai_vad.h"        
#include "speaker_diary.h"
#include "speaker_diarization/frontend/wav.h"
#include <boost/filesystem.hpp>
#include <filesystem> // For directory iteration

  
int16_t float32ToInt16(float value) {
    // 将 float32 数据范围 [-1, 1] 映射到 int16 数据范围 [-32767, 32767]
    return static_cast<int16_t>(std::round(value * 32767.0f));
}

void convertFloat32ToInt16(const float* floatData, int16_t* intData, size_t numSamples) {
    for (size_t i = 0; i < numSamples; ++i) {
        intData[i] = float32ToInt16(floatData[i]);
    }
}

void saveOneDimArrayToBinaryFile(const std::vector<int>& array, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "Failed to open file for writing." << std::endl;
        return;
    }

    // 获取数组的长度
    int size = array.size();

    // 将一维数组写入文件
    file.write(reinterpret_cast<const char*>(array.data()), size * sizeof(int));

    file.close();
    std::cout << "One-dimensional array saved to binary file: " << filename << std::endl;
}
void saveArrayToBinaryFile(const std::vector<std::vector<std::vector<double>>>& array, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "Failed to open file for writing." << std::endl;
        return;
    }

    // 获取数组的维度
    int dim1 = array.size();
    int dim2 = (dim1 > 0) ? array[0].size() : 0;
    int dim3 = (dim2 > 0) ? array[0][0].size() : 0;

    // 写入数组数据
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                double value = array[i][j][k];
                file.write(reinterpret_cast<const char*>(&value), sizeof(double));
            }
        }
    }

    file.close();
    std::cout << "Array saved to binary file: " << filename << std::endl;
}
 void saveFloatArrayToBinaryFile(const std::vector<std::vector<std::vector<float>>>& array, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "Failed to open file for writing." << std::endl;
        return;
    }

    // 获取数组的维度
    int dim1 = array.size();
    int dim2 = (dim1 > 0) ? array[0].size() : 0;
    int dim3 = (dim2 > 0) ? array[0][0].size() : 0;

    // 写入数组数据
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                float value = array[i][j][k];
                file.write(reinterpret_cast<const char*>(&value), sizeof(float));
            }
        }
    }

    file.close();
    std::cout << "Array saved to binary file: " << filename << std::endl;
}
 
bool numericStringCompare(const std::string& s1, const std::string& s2) {
    // Find the position of "audio_" in the strings
    size_t pos1 = s1.find("audio_");
    size_t pos2 = s2.find("audio_");

    // Extract numeric parts after "audio_"
    int num1 = std::stoi(s1.substr(pos1 + 6)); // 6 is the length of "audio_"
    int num2 = std::stoi(s2.substr(pos2 + 6)); // 6 is the length of "audio_"

    return num1 < num2;
}


int main() {

    std::vector<std::string> model_paths = {"./bin/3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx"};
    // std::vector<std::string> model_paths = {"./bin/voxceleb_resnet34_LM.onnx"};

    int embedding_size=512;
    int feat_dim = 80;
    int SamplesPerChunk = 16000;
 
    auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
        model_paths, feat_dim, 16000,
        embedding_size, SamplesPerChunk);
    std::vector<float> last_embs(embedding_size, 0);
    std::vector<float> current_embs(embedding_size, 0);

    int sample_chunk_ms=1000;


    // std::vector<float> enroll_embs(embedding_size, 0);



    int embedding_batch_size = 32;
    double self_frame_step = 0.016875;
    double self_frame_duration = 0.016875;
    double self_frame_start = 0.0;
    size_t min_num_samples = 640;

    int vad_begin = 0;
    int vad_end = 0;

    int32_t vad_activate_sample = (16000 * 500) / 1000;
    int32_t vad_silence_sample = (16000 * 0) / 1000;

 
   
    // std::string vad_model_path = "./bin/silero_vad.onnx";
    // int vad_sample_rate = 16000;
    // int vad_frame_ms = 32;
    // float vad_threshold = 0.65f;
    // int vad_min_silence_duration_ms = 100;
    // int vad_speech_pad_ms = 0;
    // int vad_window_samples = vad_frame_ms * (vad_sample_rate / 1000);

    // VadIterator ai_vad(vad_model_path,vad_frame_ms,vad_threshold,vad_min_silence_duration_ms);
 
    ////////////////////////////////////////////////////////////////////////////////
    //// Init ALSA and circular buffer////
    // snd_pcm_t *capture_handle;
    // const char *device_name ="plughw:2,0";   // using arecord -l to checkout the alsa device name
    // ret = audio_CQ_init(device_name, sample_rate, &params, capture_handle);
    // printf("Started\n");
    // ai_vad.reset_states();
 
    // wav::WavReader wav_reader("./test_audio/meeting_one/audio5.wav");
    wav::WavReader wav_reader("./test_audio/meeting_one/audio93.wav");

    int num_channels = wav_reader.num_channels();
    int bits_per_sample = wav_reader.bits_per_sample();
    int sample_rate = wav_reader.sample_rate();
    const float* audio = wav_reader.data();
    int num_samples = wav_reader.num_samples();
    std::vector<float> input_wav{audio, audio + num_samples};
    for( int i = 0; i < num_samples; ++i )
    {
        input_wav[i] = input_wav[i]*1.0f/32768.0;
    }

    printf("input_wav size%d",input_wav.size());

    // printf("sample_rate %d",sample_rate);
    const std::string segmentModel="./bin/seg_model.onnx";
    // int embedding_size = 512;
    auto start_time = std::chrono::high_resolution_clock::now();

    SegmentModel mm(segmentModel);
    std::vector<std::pair<double, double>> segments;
    SlidingWindow res_frames;
    auto segmentations = mm.slide( input_wav, res_frames );
    auto segment_data = res_frames.data();
    for( auto seg : segment_data )
    {
        segments.emplace_back( std::make_pair( seg.start, seg.end ));
    }
    std::cout<<segmentations.size()<<"x"<<segmentations[0].size()<<"x"<<segmentations[0][0].size()<<std::endl;
    // saveFloatArrayToBinaryFile(segmentations, "segmentations.bin");
 
    auto binarized = mm.binarize_swf( segmentations, false );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "完整运行时间为: " << time.count() << " 毫秒" << std::endl;

    assert( binarized.size() == segments.size());

    int num_chunk = binarized.size();
    int num_frame = binarized[0].size();
    int num_speaker = binarized[0][0].size();
    std::vector<int> speaker_counts(num_speaker, 0);

    for (int i = 0; i < num_chunk; ++i) {
        for (int j = 0; j < num_frame; ++j) {
            for (int k = 0; k < num_speaker; ++k) {
                speaker_counts[k] += binarized[i][j][k];
            }
        }
    }
    int frame_total=num_chunk*num_frame;
    std::cout << "Speaker Counts:" << std::endl;
    for (int i = 0; i < speaker_counts.size(); ++i) {
        std::cout << "Speaker " << i << ": " << speaker_counts[i] << "prob: " << (float)speaker_counts[i]/frame_total << std::endl;
    }

 
    // estimate frame-level number of instantaneous speakers
    // In python code, binarized in speaker_count function is cacluated with 
    // same parameters as we did above, so we reuse it by passing it into speaker_count
    SlidingWindow count_frames( num_samples );
    SlidingWindow pre_frame( self_frame_start, self_frame_step, self_frame_duration );
    auto count_data = mm.speaker_count( segmentations, binarized, 
            pre_frame, count_frames, num_samples );
    // std::cout << "count_data: " << count_data.size() << std::endl;
    // std::cout << "count_data[0]: " << count_data[0]<< std::endl;
    saveOneDimArrayToBinaryFile(count_data,"count.bin");
    // python: duration = binary_segmentations.sliding_window.duration
    double duration = 5.0;
    size_t num_chunks = binarized.size();
    size_t num_frames = binarized[0].size(); 

    // python: num_samples = duration * self._embedding.sample_rate 
    size_t min_num_frames = ceil(num_frames * min_num_samples / ( duration * 16000 ));
    // std::cout << "min_num_frames: " << min_num_frames << std::endl;
    // std::cout << "num_chunks: " << num_chunks << std::endl;

    // python: clean_frames = 1.0 * ( np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2 )
    // python: clean_segmentations = SlidingWindowFeature( 
    //                               binary_segmentations.data * clean_frames, binary_segmentations.sliding_window )
    // saveArrayToBinaryFile(binarized, "binarized.bin");
    // printf("binarized (%d,%d,%d)",binarized.size(),binarized[0].size(),binarized[0][0].size());

    auto clean_segmentations = Helper::cleanSegmentations( binarized );
    saveArrayToBinaryFile(binarized, "binarized.bin");

    assert( binarized.size() == clean_segmentations.size());
    std::vector<std::vector<float>> batchData;
    std::vector<std::vector<float>> batchMasks;
    std::vector<std::vector<double>> embeddings;

    // printf("clean_segmentations (%d,%d,%d)",clean_segmentations.size(),clean_segmentations[0].size(),clean_segmentations[0][0].size());

    //  for (int i = 0; i < clean_segmentations.size(); ++i) {
    //     for (int j = 0; j < clean_segmentations[i].size(); ++j) {
    //         for (int k = 0; k < clean_segmentations[i][j].size(); ++k) {
    //             if (clean_segmentations[i][j][k] != 0) {
    //                 std::cout << "Non-zero element at index (" << i << "," << j << "," << k << "): " << clean_segmentations[i][j][k] << std::endl;
    //              }
    //         }
    //     }
    // }


    
    //////////////////////////////////////// Embedding //////////////////////////////////////////////////////
    for( size_t i = 0; i < binarized.size(); ++i )
    {
        auto chunkData = mm.crop( input_wav, segments[i] );
        auto& masks = binarized[i];
        auto& clean_masks = clean_segmentations[i];
        assert( masks[0].size() == 3 );
        assert( clean_masks[0].size() == 3 );
        // printf("chunkData size %d\n",chunkData.size());
        // printf("seg start %f\n",segments[i].first);
        // printf("seg end %f\n",segments[i].second);


        // python: for mask, clean_mask in zip(masks.T, clean_masks.T):
        for( size_t j = 0; j < clean_masks[0].size(); ++j )
        {
            std::vector<float> used_mask;
            float sum = 0.0;
            std::vector<float> reversed_clean_mask(clean_masks.size());
            std::vector<float> reversed_mask(masks.size());

            // python: np.sum(clean_mask)
            for( size_t k = 0; k < clean_masks.size(); ++k )
            {
 
                sum += clean_masks[k][j];
                reversed_clean_mask[k] = clean_masks[k][j];
                reversed_mask[k] = masks[k][j];
            }
            // printf("min_num_frames %d\n",min_num_frames);
 
            if( sum > min_num_frames )
            {
    
                used_mask = std::move( reversed_clean_mask );
            }
            else
            {
                used_mask = std::move( reversed_mask );
            }

            // batchify
            batchData.push_back( chunkData );
            batchMasks.push_back( std::move( used_mask ));
            if( batchData.size() == embedding_batch_size )
            {
                auto embedding = getEmbedding( speaker_engine, batchData, batchMasks );
                batchData.clear();
                batchMasks.clear();

                for( auto& a : embedding )
                {
                    embeddings.push_back( std::move( a ));
                }
            }
        }
    }
    // Process remaining
    if( batchData.size() > 0 )
    {
        auto embedding = getEmbedding( speaker_engine, batchData, batchMasks );
        for( auto& a : embedding )
        {
            embeddings.push_back( std::move( a ));
        }
    }
    printf("num_chunks %d",num_chunks);

    std::cout << "embeddings result size" << embeddings.size()<<","<<embeddings[0].size() << std::endl;
    auto embeddings1 = Helper::rearrange_up( embeddings, num_chunks );
    std::cout << "embeddings1 result size" << embeddings1.size()<<","<<embeddings1[0].size()<<","<<embeddings1[0][0].size() << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////



     /////////////////////////////////////////////////////////////////////////////////////////////////////////////

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


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

 

    return 0;
}

 
