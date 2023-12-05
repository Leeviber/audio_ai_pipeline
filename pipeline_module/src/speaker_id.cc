#ifndef SPEAKER_ID
#include <stdio.h>
#include <chrono>
#include <string>
#include <vector>
// #include "sherpa_stt/offline-recognizer.h"
// #include "sherpa_stt/offline-model-config.h"
// #include "sherpa_stt/offline-transducer-model-config.h"
#include "ai_vad.h" // ai based vad

#include "alsa_cq_buffer.h" // ALSAstt_core

#include "speaker_id.h"

#include "tts.h" //tts

// std::string model_path ="./bin/voxceleb_resnet34_LM.onnx";
// int embedding_size=256;

std::string model_path = "./bin/voxceleb_CAM++_LM.onnx";
int embedding_size = 512;

int feat_dim = 80;
int sample_rate = 16000;
int SamplesPerChunk = 32000;
auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
    model_path, feat_dim, sample_rate,
    embedding_size, SamplesPerChunk);

int32_t init_online_audio(online_params *params)
{

    params->is_running = true;

    online_audio audio_buffer;
    audio_buffer.pcmf32_new = std::vector<float>(params->n_samples_30s, 0.0f);
    audio_buffer.CQ_buffer.resize(sample_rate * 30);
    params->audio = audio_buffer;
    float value;

    return 0;
}

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
std::vector<float> last_embs(embedding_size, 0);
std::vector<float> current_embs(embedding_size, 0);

int sample_chunk_ms = 1000;

std::vector<float> enroll_embs(embedding_size, 0);

std::vector<std::vector<float>> vad_sgement_embedding(std::string audio_path, VadIterator &vad, int vad_windoes_sample)
{
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

    std::vector<float> full_emb(embedding_size, 0);
    std::vector<float> chunk_mean_emb(embedding_size, 0);
    std::vector<std::vector<float>> chunk_enroll_embs;

    for (int j = 0; j < samples; j += vad_windoes_sample)
    {

        std::vector<float> window_chunk{&enroll_data_flt32[0] + j, &enroll_data_flt32[0] + j + vad_windoes_sample};

        int32_t vad_state = vad.predict(window_chunk);
        if (vad_state == 2)
        {
            seg_start = j - vad_windoes_sample;
        }
        if (vad_state == 3)
        {
            seg_end = j;
            if (seg_start != -1)
            {
                std::vector<int16_t> vad_chunk_int16{&enroll_data_int16[seg_start], &enroll_data_int16[seg_end]};
                std::vector<float> chunk_emb(embedding_size, 0);

                speaker_engine->ExtractEmbedding(vad_chunk_int16.data(),
                                                 vad_chunk_int16.size(),
                                                 &chunk_emb);

                chunk_enroll_embs.push_back(chunk_emb);

                for (size_t i = 0; i < embedding_size; i++)
                {
                    chunk_mean_emb[i] += chunk_emb[i];
                }
            }
        }
    }

    int chunk_size = chunk_enroll_embs.size();

    for (size_t i = 0; i < embedding_size; i++)
    {
        chunk_mean_emb[i] = (float)chunk_mean_emb[i] / (float)chunk_size;
    }
    chunk_enroll_embs.push_back(chunk_mean_emb);

    speaker_engine->ExtractEmbedding(enroll_data_int16,
                                     samples,
                                     &full_emb);
    chunk_enroll_embs.push_back(full_emb);

    return chunk_enroll_embs;
}

int main()
{

    online_params params;
    int32_t ret = init_online_audio(&params);
    if (ret < 0)
    {
        fprintf(stderr, "Error init_kws \n");
        return -1;
    }
    //////////////// Init TTS //////////////////////

    bool tts_response = true;
    tts_params ttsConfig;

    if (tts_response)
    {
        ttsConfig.modelPath = "./bin/en_US-joe-medium.onnx";
        ttsConfig.modelConfigPath = "./bin/en_US-joe-medium.onnx.json";
        ttsConfig.piperConfig.eSpeakDataPath = "./bin/espeak-ng-data/";
        ttsConfig.play_device = "plughw:1,0"; // using aplay -l to checkout the sound devices for play
        ttsConfig.silence_ms = 100;           // the silence ms between tecx chunk segment by punctuation

        // Initialize the tts
        init_tts(ttsConfig);

        // The alsa play here is set to non-blocking mode. So, need wait it play finished.
        snd_pcm_state_t state = snd_pcm_state(ttsConfig.pcm_handle);
        while (state == SND_PCM_STATE_RUNNING)
        {
            state = snd_pcm_state(ttsConfig.pcm_handle);
        }
    }

    int vad_begin = 0;
    int vad_end = 0;

    int32_t vad_activate_sample = (16000 * 500) / 1000;
    int32_t vad_silence_sample = (16000 * 0) / 1000;

    std::string path = "./bin/silero_vad.onnx";
    int test_sr = 16000;
    int test_frame_ms = 96;
    float test_threshold = 0.85f;
    int test_min_silence_duration_ms = 100;
    int test_speech_pad_ms = 0;
    int test_window_samples = test_frame_ms * (test_sr / 1000);

    VadIterator ai_vad(
        path, test_sr, test_frame_ms, test_threshold,
        test_min_silence_duration_ms, test_speech_pad_ms);

    // printf("here\n");
    json enroll_embeddings;
    std::ifstream jfile("enroll_embeddings.json");
    jfile >> enroll_embeddings;

    ///////////////////////////////////////// generate new user embedding json from wav audio ///////////////////////////////////////////////
    // json enroll_embeddings;

    // std::string audio_path ="./bin/shiyan_long.wav";
    // std::vector<std::vector<float>>shiyan_enroll = vad_sgement_embedding(audio_path,ai_vad,test_window_samples);
    // enroll_embeddings["new shiyan"]=shiyan_enroll;

    // std::ofstream outputFile("enroll_embeddings.json");
    // if (outputFile.is_open()) {
    //     outputFile << enroll_embeddings.dump(4); // 使用缩进格式化输出，4表示缩进的空格数
    //     outputFile.close();
    //     std::cout << "JSON saved to file." << std::endl;
    // } else {
    //     std::cerr << "Unable to open file for writing." << std::endl;
    // }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    float speaker_id_threshold = 0.4;

    ////////////////////////////////////////////////////////////////////////////////
    //// Init ALSA and circular buffer////
    snd_pcm_t *capture_handle;
    const char *device_name = "plughw:2,0"; // using arecord -l to checkout the alsa device name
    ret = audio_CQ_init(device_name, sample_rate, &params, capture_handle);
    printf("Started\n");
    ai_vad.reset_states();

    //////////////////////////////////////
    // while (false)
    while (params.is_running)
    {
        while (true)
        {
            int len = audio_CQ_get(&params, 96, 0);
            if (len > 0)
            {
                const auto onnx_begin = std::chrono::steady_clock::now();

                int32_t vad_state = ai_vad.predict(params.audio.pcmf32_new);

                if (vad_state == 2)
                {
                    printf("begin\n");
                    vad_begin = params.CQ_audio_exit - vad_activate_sample;
                }
                if (vad_state == 3)
                {
                    printf("end\n");
                    vad_end = params.CQ_audio_exit - vad_silence_sample;

                    len = audio_CQ_view(&params, vad_begin, vad_end);
                    if (len > 0)
                    {
                        break;
                    }
                }
            }
        }

        int bufferFrames = params.audio.pcmf32_new.size();
        const auto begin = std::chrono::steady_clock::now();

        std::vector<int16_t> int16Buffer(params.audio.pcmf32_new.size());
        for (int i = 0; i < bufferFrames; ++i)
        {
            float scaledValue = params.audio.pcmf32_new[i] * 32767.0f;
            int16Buffer[i] = static_cast<short>(std::round(scaledValue));
        }

        speaker_engine->ExtractEmbedding(int16Buffer.data(),
                                         bufferFrames,
                                         &current_embs);
        const auto end = std::chrono::steady_clock::now();
        float elapsed_seconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
                .count() /
            1000.;
        float duration = (float)params.audio.pcmf32_new.size() / (float)16000.0;
        float rtf = elapsed_seconds / duration;
        fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
                elapsed_seconds, duration, rtf);

        bool matched = false;
        float max_cosine_score = 0.0;
        std::string max_score_speaker = "Nobody";

        for (auto it = enroll_embeddings.begin(); it != enroll_embeddings.end(); ++it)
        {
            std::string name = it.key();
            int speaker_embed_num = enroll_embeddings[name].size();
 
            for (int i = 0; i < speaker_embed_num; i++)
            {
                float cosine_score = speaker_engine->CosineSimilarity(current_embs,
                                                                      enroll_embeddings[name][i]);
                if (cosine_score > max_cosine_score && cosine_score > speaker_id_threshold)
                {
                    matched = true;
                    max_cosine_score = cosine_score;
                    max_score_speaker = name;
                }
            }
        }

        if (tts_response && matched)
        {
            std::string prefix = "You are ";
            std::string line = prefix + max_score_speaker;
            std::cout << line << " and the probablity is " << max_cosine_score << std::endl;

            process_tts(line, ttsConfig);
            snd_pcm_state_t state = snd_pcm_state(ttsConfig.pcm_handle);
            while (state == SND_PCM_STATE_RUNNING)
            {
                state = snd_pcm_state(ttsConfig.pcm_handle);
            }
            enroll_embeddings[max_score_speaker].push_back(current_embs);
        }
        else if (!matched)
        {
            printf("Not detect any one\n");
        }

        current_embs.clear();
    }
}

#endif