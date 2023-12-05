#include "tts.h"

void init_tts(tts_params &ttsConfig)
{

    // Load the voice
    loadVoice(ttsConfig.piperConfig, ttsConfig.modelPath, ttsConfig.modelConfigPath, ttsConfig.voice);

    // Initialize tts model
    piper::initialize(ttsConfig.piperConfig);

    // The silence between text segment
    std::size_t segment_silence_Samples = (std::size_t)(ttsConfig.silence_ms *
                                                        ttsConfig.voice.synthesisConfig.sampleRate / 1000);
    std::vector<int16_t> segment_silence_chunk(segment_silence_Samples, 0);
    ttsConfig.silence_chunk = segment_silence_chunk;

    // Initialize the alsa pcm_handle for play sound
    snd_pcm_t *pcm_handle;
    int err = snd_pcm_open(&pcm_handle, ttsConfig.play_device, SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0)
    {
        printf("unable open play device: %s\n", snd_strerror(err));
    }

    snd_pcm_hw_params_t *hw_params;
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(pcm_handle, hw_params);
    snd_pcm_hw_params_set_access(pcm_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm_handle, hw_params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(pcm_handle, hw_params, 1);

    // The sample rate for generated speech, cannot modify because rely on model
    unsigned int sample_rate = 22050;
    snd_pcm_hw_params_set_rate_near(pcm_handle, hw_params, &sample_rate, 0);

    // need set the hardware buffer otherwise the sound could be cutted
    err = snd_pcm_hw_params_set_buffer_size(pcm_handle, hw_params, 8192);
    if (err < 0)
    {
        printf("unable set the hardware buffer: %s\n", snd_strerror(err));
    }

    snd_pcm_hw_params(pcm_handle, hw_params);
    snd_pcm_prepare(pcm_handle);

    ttsConfig.pcm_handle = pcm_handle;
}

void alsa_play(snd_pcm_t *pcm_handle, std::vector<int16_t> audioBuffer)
{

    // decrease the play sound otherwise it will play using 100% volume
    for (int i = 0; i < audioBuffer.size(); i++)
    {
        audioBuffer[i] = audioBuffer[i] / 5;
    }

    snd_pcm_prepare(pcm_handle);

    int ret = snd_pcm_writei(pcm_handle, audioBuffer.data(), audioBuffer.size());

    if (ret < 0) // if something wrong for alsa play, need recover and prepare
    {
        fprintf(stderr, "error from writei: %s\n", snd_strerror(ret));
        snd_pcm_recover(pcm_handle, ret, 0);
        snd_pcm_prepare(pcm_handle);
    }
}

// segment the text by punctuation.
// For example: "hello, how are you , i am ok! ", will be segment into 3 chunk for tts process to avoid the long text process
std::vector<std::string> segmentText(const std::string &text)
{
    std::vector<std::string> segments;
    std::string segment;
    std::string punctuation = ".,;:!?，。";

    for (char c : text)
    {
        if (punctuation.find(c) != std::string::npos)
        {
            if (!segment.empty())
            {
                segments.push_back(segment);
                segment.clear();
            }
        }
        else
        {
            segment += c;
        }
    }

    if (!segment.empty())
    {
        segments.push_back(segment);
    }

    return segments;
}

// Stt process funtion, text is the input data for speech generate.
void process_tts(const std::string &text, tts_params &ttsConfig)
{

    // segment the raw text into small chunk for stt accelerate
    std::vector<std::string> segments = segmentText(text);

    for (int i = 0; i < segments.size(); i++)
    {
        std::vector<int16_t> audioChunk;
        audioChunk.clear();
        std::string segment = segments[i];

        piper::textToAudio(ttsConfig.piperConfig, ttsConfig.voice, segment, audioChunk);

        // For any text with punctuation, here will incert a silence chunk. Not in first segment which means not silence at beginning.
        if (i != 0)
        {
            alsa_play(ttsConfig.pcm_handle, ttsConfig.silence_chunk);
        }

        alsa_play(ttsConfig.pcm_handle, audioChunk);
    }
}
