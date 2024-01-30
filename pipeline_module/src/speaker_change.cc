#ifndef SPEAKER_CHANGE
#include "speaker_change.h"
#include "speaker_diarization/speaker_diarization.h"


class IDAnalyzer
{
private:
    std::vector<int> idWindow; // past speaker id embedding vector
    int window_frame_length=20; // the num of frame for past embedding buffer
public:
    void addID(int id)
    {
        idWindow.push_back(id);

        // If the window exceeds a time range of one minute, remove the oldest ID
        if (idWindow.size() > window_frame_length)
        {
            int oldestID = idWindow.front();
            idWindow.erase(idWindow.begin());
        }
    }

    double getIDRatio(int newID)
    {
        int count = 0;
        for (int id : idWindow)
        {
            if (id == newID)
            {
                count++;
            }
        }
        return static_cast<double>(count) / std::min(static_cast<int>(idWindow.size()), window_frame_length);
    }
};

class IDTrack
{
private:
    float speaker_id_threshold = 0.4;
    std::map<int, std::vector<std::vector<float>>> id_tracker;
    int new_id = 1;
    
public:
    struct MatchedInfo
    {
        bool is_first;
        int matched_id;
    };
    MatchedInfo matchedInfo;

    MatchedInfo checkMatchedTrack(std::shared_ptr<wespeaker::SpeakerEngine> speaker_engine, std::vector<float> current_embs)
    {

        float max_cosine_score = 0.0;
        bool matched = false;
        int matched_idx;

        for (auto &pair : id_tracker)
        {
            int id = pair.first;
            std::vector<std::vector<float>> embeddings = pair.second;
            for (int i = 0; i < embeddings.size(); i++)
            {
                float cosine_score = speaker_engine->CosineSimilarity(current_embs,
                                                                      embeddings[i]);
                // if (cosine_score > max_cosine_score && cosine_score > speaker_id_threshold)
                // {
                //     matched = true;
                //     max_cosine_score = cosine_score;
                //     matched_idx = id;
                // }
                if (cosine_score > speaker_id_threshold)
                {
                    matched = true;
                    // max_cosine_score = cosine_score;
                    matched_idx = id;
                    break;
                }
            }
        }
        if (matched)
        {
            matchedInfo.is_first = false;
            matchedInfo.matched_id = matched_idx;
            if (id_tracker[matched_idx].size() < 50)
            {
                id_tracker[matched_idx].push_back(current_embs);
            }
        }
        else
        {
            matchedInfo.is_first = true;
            matchedInfo.matched_id = new_id;

            std::vector<std::vector<float>> embeddings;
            embeddings.push_back(current_embs);
            id_tracker[new_id] = embeddings;
            new_id++;
        }
        return matchedInfo;
    }

    void filterTracker(IDAnalyzer idanalyzer)
    {
        for (auto it = id_tracker.begin(); it != id_tracker.end();)
        {
            // When the embedding size is less than three and the attention ratio in the past minute is below 0.2, filter it out
            if (it->second.size() < 3 && idanalyzer.getIDRatio(it->first) < 0.2)
            {
                it->second.clear();
                it = id_tracker.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }


};

// int32_t init_online_audio(online_params *params)
// {

//     params->is_running = true;

//     online_audio audio_buffer;
//     audio_buffer.pcmf32_new = std::vector<float>(params->n_samples_30s, 0.0f);
//     audio_buffer.CQ_buffer.resize(params->sample_rate * 30);
//     params->audio = audio_buffer;
//     float value;

//     return 0;
// }

void writeDataToFile(std::ofstream &file, bool overwrite, int tag, int first_time, double probability)
{
    if (file.is_open())
    {
        if (overwrite)
        {
            file.seekp(0); // 将写入位置移动到文件开头
        }
        std::string data = std::to_string(tag) + "/" + std::to_string(first_time) + "/" + std::to_string(probability);
        file.write(data.c_str(), data.size()); // 写入数据到文件

        file.flush(); // 刷新文件缓冲区
        std::cout << "success written to txt" << std::endl;
    }
    else
    {
        std::cerr << "unable create or open txt file" << std::endl;
    }
}


int main()
{

    //////////////// Init ALSA and circular buffer////////////////////////////////////////

    online_params params;
    int32_t ret = init_online_audio(&params);
    if (ret < 0)
    {
        fprintf(stderr, "Error init_alsa \n");
        return -1;
    }
    snd_pcm_t *capture_handle;
    const char *device_name = "plughw:2,0"; // using arecord -l to checkout the alsa device name
    ret = audio_CQ_init(device_name, params.sample_rate, &params, capture_handle);
    ////////////////////////////////////////////////////////////////////////////////

    /////////////////////// Init Speaker ID ///////////////////////////
    // std::string model_path = "./bin/voxceleb_CAM++_LM.onnx";
    // int embedding_size = 512;

    std::vector<std::string> model_paths;
#ifdef USE_NPU
    std::string rknn_model_path ="./bin/Id1_resnet34_LM_main_part.rknn";
    std::string onnx_model_path ="./bin/Id2_resnet34_LM_post.onnx";
    model_paths.push_back(rknn_model_path);
    model_paths.push_back(onnx_model_path);

#else
    std::string onnx_model_path ="./bin/voxceleb_resnet34_LM.onnx";
    model_paths.push_back(onnx_model_path);

#endif

    int embedding_size=256;
    int feat_dim = 80;
    int SamplesPerChunk = 32000;
    auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
        model_paths, feat_dim, params.sample_rate,
        embedding_size, SamplesPerChunk);

    std::vector<float> current_embs(embedding_size, 0);
    /////////////////////////////////////////////////////////////////

    /////////////////////// Init AI-VAD ///////////////////////////
    // std::string path = "./bin/silero_vad.onnx";
    // int vad_sr = 16000;
    // int vad_frame_ms = 96;
    // float vad_threshold = 0.85f;
    // int vad_min_silence_duration_ms = 100;
    // int vad_speech_pad_ms = 0;
    // int vad_window_samples = vad_frame_ms * (vad_sr / 1000);

    // VadIterator ai_vad(
    //     path, vad_sr, vad_frame_ms, vad_threshold,
    //     vad_min_silence_duration_ms, vad_speech_pad_ms);
    ////////////////////////////////////////////////////////////////////////////////

    //////////////// Init ID Track ////////////////////////////////////////////////

    // int speaker_frame_ms = 5000;
    // int speaking_threshold = 6;
    // int vad_detect_num = speaker_frame_ms / vad_frame_ms;
    // std::map<int, std::vector<std::vector<float>>> id_tracker;
    // int new_id = 1; /// id 0 for non-speech

    // IDTrack idTrack;
    // IDAnalyzer idanalyzer;
 
    // std::string filename = "/dev/shm/sid.txt";
    // std::ofstream file(filename); 
    // bool overwriteOutput = true;
    // int filter_count = 0;
    ////////////////////////////////////////////////////////////////////////////////
    const std::string segmentModel="./bin/seg_model.onnx";
    SlidingWindow res_frames;

    SegmentModel mm(segmentModel);




    printf("Started1\n");

    while (params.is_running)
    {
        int speaking_count = 0;

        while (true)
        {
            int len = audio_CQ_get(&params, 5000, 500);
            
        

            if (len >= 0)
            {
                int bufferFrames = params.audio.pcmf32_new.size();
                // printf("bufferFrames%d",bufferFrames);
 
                // for (int i = 0; i < vad_detect_num; i++)
                // {
                //     std::vector<float> chunk(vad_window_samples, 0.0);

                //     auto start = params.audio.pcmf32_new.begin() + (i * vad_window_samples);
                //     auto end = start + vad_window_samples;
                //     std::copy(start, end, chunk.begin());
                    // int32_t vad_state = ai_vad.predict(chunk);
                    // if (vad_state == 2 || vad_state == 4)
                    // {
                    //     speaking_count += 1;
                    // }
                // }
                break;  

            }

        }
        printf("process \n");
        auto segmentations = mm.slide( params.audio.pcmf32_new, res_frames );
        // std::cout<<segmentations.size()<<"x"<<segmentations[0].size()<<"x"<<segmentations[0][0].size()<<std::endl;

        //////////////////////////////// Embedding /////////////////////////////////////////////////

        // If the number of active speech segments within one second is greater than the speaking_threshold
        //  , ID tracking will be performed
        // if (speaking_count >= speaking_threshold)
        // {
        //     printf("speakeing\n");
        //     int bufferFrames = params.audio.pcmf32_new.size();

        //     // speaker id need int16 audio data
        //     std::vector<int16_t> int16Buffer(params.audio.pcmf32_new.size());
        //     for (int i = 0; i < bufferFrames; ++i)
        //     {
        //         float scaledValue = params.audio.pcmf32_new[i] * 32767.0f;
        //         int16Buffer[i] = static_cast<short>(std::round(scaledValue));
        //     }
        //     auto start = std::chrono::high_resolution_clock::now();

        //     speaker_engine->ExtractEmbedding(int16Buffer.data(),
        //                                      bufferFrames,
        //                                      &current_embs);
        //     auto end = std::chrono::high_resolution_clock::now();
        //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        //     std::cout << "once embedding time：" << duration << " ms" << std::endl;

        //     IDTrack::MatchedInfo matchinfo = idTrack.checkMatchedTrack(speaker_engine, current_embs);
        //     idanalyzer.addID(matchinfo.matched_id);

        //     // the id of this second is first appear over past 60s
        //     if (matchinfo.is_first)
        //     {
        //         writeDataToFile(file, overwriteOutput, matchinfo.matched_id, matchinfo.is_first, 0);
        //         printf("new id %d \n", matchinfo.matched_id);
        //     }
        //     // the id of this second already matched over past 60s
        //     else
        //     {
        //         double id_ratio = idanalyzer.getIDRatio(matchinfo.matched_id);
        //         printf("matched speaker id %d, attention ratio%f \n", matchinfo.matched_id, id_ratio);
        //         writeDataToFile(file, overwriteOutput, matchinfo.matched_id, matchinfo.is_first, id_ratio);
        //     }
        // }
        // else
        // {
        //     printf("silence\n");
        //     writeDataToFile(file, overwriteOutput, 0, 0, 0);
        // }

        // // filter out some short and inactivate id embedding every 60 second (120*0.5)
        // if (filter_count++ >= 120)
        // {
        //     idTrack.filterTracker(idanalyzer);
        //     filter_count = 0;
        // }

        // current_embs.clear();

        //////////////////////////////// Embedding /////////////////////////////////////////////////

    }
}

#endif