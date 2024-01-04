// #ifndef SHERPA_ONNX_ALSA_CQ_H_

#include "alsa_cq_buffer.h"

int32_t audio_CQ_init(const char *capture_id, int sample_rate, online_params *params, snd_pcm_t *capture_handle)
{
    int err;

    unsigned int rate = 16000;
    snd_pcm_hw_params_t *hw_params;
    snd_pcm_format_t format = SND_PCM_FORMAT_FLOAT_LE;

    if ((err = snd_pcm_open(&capture_handle, capture_id, SND_PCM_STREAM_CAPTURE, 0)) < 0)
    {
        fprintf(stderr, "cannot open audio device %s (%s)\n",
                capture_id,
                snd_strerror(err));
        exit(1);
    }

    if ((err = snd_pcm_hw_params_malloc(&hw_params)) < 0)
    {
        fprintf(stderr, "cannot allocate hardware parameter structure (%s)\n",
                snd_strerror(err));
        exit(1);
    }
    if ((err = snd_pcm_hw_params_any(capture_handle, hw_params)) < 0)
    {
        fprintf(stderr, "cannot initialize hardware parameter structure (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    if ((err = snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0)
    {
        fprintf(stderr, "cannot set access type (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    if ((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, format)) < 0)
    {
        fprintf(stderr, "cannot set sample format (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    if ((err = snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, 0)) < 0)
    {
        fprintf(stderr, "cannot set sample rate (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    if ((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, 1)) < 0)
    {
        fprintf(stderr, "cannot set channel count (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    if ((err = snd_pcm_hw_params(capture_handle, hw_params)) < 0)
    {
        fprintf(stderr, "cannot set parameters (%s)\n",
                snd_strerror(err));
        exit(1);
    }

    snd_pcm_hw_params_free(hw_params);

    if ((err = snd_pcm_prepare(capture_handle)) < 0)
    {
        fprintf(stderr, "cannot prepare audio interface for use (%s)\n",
                snd_strerror(err));
        exit(1);
    }
    

    // continue push the audio stream to buffer
    std::thread audio_thread(audio_CQ_push, params, capture_handle);
    audio_thread.detach();

    return 0;
}

int32_t audio_CQ_push(online_params *params, snd_pcm_t *ahandler)
{
    // printf("params->is_running%d",params->is_running);
    while (params->is_running)
    {

        const size_t buffer_frames = (16000 * 100) / 1000;
        std::vector<float> buffer(buffer_frames);
  
        snd_pcm_t *capture_handle = (snd_pcm_t *)ahandler;
        int err;
        if ((err = snd_pcm_readi(capture_handle, buffer.data(), buffer_frames)) != buffer_frames)
        {
            fprintf(stderr, "read from audio interface failed (%s)\n",
                    err, snd_strerror(err));
            exit(1);
        }

        std::lock_guard<std::mutex> lock(params->m_mutex);
        if (params->CQ_audio_entrance + buffer.size() >= params->audio.CQ_buffer.size())
        {

            const size_t n0 = params->audio.CQ_buffer.size() - params->CQ_audio_entrance;

            memcpy(&params->audio.CQ_buffer[params->CQ_audio_entrance], buffer.data(), n0 * sizeof(float));
            memcpy(&params->audio.CQ_buffer[0], &buffer[n0], (buffer.size() - n0) * sizeof(float));

            params->CQ_audio_entrance = (params->CQ_audio_entrance + buffer.size()) % params->audio.CQ_buffer.size();

            // params->m_audio_len = params->audio.CQ_buffer.size();
        }
        else
        {

            memcpy(&params->audio.CQ_buffer[params->CQ_audio_entrance], buffer.data(), buffer.size() * sizeof(float));

            params->CQ_audio_entrance = (params->CQ_audio_entrance + buffer.size());
            // params->m_audio_len = std::min(params->m_audio_len + buffer.size(), params->audio.CQ_buffer.size());
        }
   
        // fprintf(stdout,"CQ_audio_entrance in push %d \n",params->CQ_audio_entrance);

    }
    return 0;
}

int32_t audio_CQ_get(online_params *params,int get_ms,int keep_ms)
{

    params->audio.pcmf32_new.clear();

    std::lock_guard<std::mutex> lock(params->m_mutex);
    int32_t n_samples = (16000 * get_ms) / 1000;
    int32_t len = audio_CQ_length(params);
    int32_t exit_num=n_samples;
    if(keep_ms !=0)
    {
        exit_num= (16000 * keep_ms) / 1000;
    }

    if (n_samples <= len)
    {
        params->audio.pcmf32_new.resize(n_samples);

        if (params->CQ_audio_exit + n_samples > params->audio.CQ_buffer.size())
        {
            int32_t n0 = params->audio.CQ_buffer.size() - params->CQ_audio_exit;

            memcpy(params->audio.pcmf32_new.data(), &params->audio.CQ_buffer[params->CQ_audio_exit], n0 * sizeof(float));
            memcpy(&params->audio.pcmf32_new[n0], &params->audio.CQ_buffer[0], (n_samples - n0) * sizeof(float));
            params->CQ_audio_exit = (params->CQ_audio_exit + exit_num) % params->audio.CQ_buffer.size();
        }
        else
        {
            memcpy(params->audio.pcmf32_new.data(), &params->audio.CQ_buffer[params->CQ_audio_exit], n_samples * sizeof(float));
            params->CQ_audio_exit = params->CQ_audio_exit + exit_num;
        }
    }
    else
    {
 
        return len - n_samples;
    }
    // fprintf(stdout,"CQ_audio_exit in get %d \n",params->CQ_audio_exit);

    return n_samples;
}

int32_t audio_CQ_view(online_params *params,int start,int end )
{

    params->audio.pcmf32_new.clear();

    std::lock_guard<std::mutex> lock(params->m_mutex);

    int32_t min_sample = (16000 * 500) / 1000;
    int32_t n_samples;

    if(start<end)
    {
        n_samples=end-start;
    }
    else
    {
        n_samples=params->audio.CQ_buffer.size()-start+end;
    }
    
    if(n_samples<min_sample)
    {
        printf("too short to handle\n");
        return -1;
    }
 
    params->audio.pcmf32_new.resize(n_samples);

    if (start + n_samples > params->audio.CQ_buffer.size())
    {
        int32_t n0 = params->audio.CQ_buffer.size() - start;

        memcpy(params->audio.pcmf32_new.data(), &params->audio.CQ_buffer[start], n0 * sizeof(float));
        memcpy(&params->audio.pcmf32_new[n0], &params->audio.CQ_buffer[0], (n_samples - n0) * sizeof(float));
        // params->CQ_audio_exit = (params->CQ_audio_exit + exit_num) % params->audio.CQ_buffer.size();
    }
    else
    {
        memcpy(params->audio.pcmf32_new.data(), &params->audio.CQ_buffer[start], n_samples * sizeof(float));
        // params->CQ_audio_exit = params->CQ_audio_exit + exit_num;
    }

    // fprintf(stdout,"CQ_audio_exit in get %d \n",params->CQ_audio_exit);

    return n_samples;
}

int32_t audio_CQ_length(online_params *params)
{
    uint32_t len = 0;
    if (params->CQ_audio_entrance >= params->CQ_audio_exit)
    {
        len = params->CQ_audio_entrance - params->CQ_audio_exit;
    }
    else
    {
        len += params->audio.CQ_buffer.size() - params->CQ_audio_exit;
        len += params->CQ_audio_entrance;
    }

    return len;
}

int32_t audio_CQ_clear(online_params *params)
{
    uint32_t len = 0;
    params->CQ_audio_entrance=0;
    params->CQ_audio_exit=0;
    memset(params->audio.CQ_buffer.data(), 0, params->audio.CQ_buffer.size() * sizeof(float));
    // params->audio.CQ_buffer.resize(16000 * 30);

    return len;
}

int32_t init_online_audio(online_params *params)
{

  params->is_running = true;

  online_audio audio_buffer;
  audio_buffer.pcmf32_new = std::vector<float>(params->n_samples_30s, 0.0f);
  audio_buffer.CQ_buffer.resize(params->sample_rate * 30);
  params->audio = audio_buffer;
  float value;

  return 0;
}
// #endif  // SHERPA_ONNX_ALSA_CQ_H_
