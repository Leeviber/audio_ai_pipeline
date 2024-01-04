#ifndef SHERPA_ONNX_ALSA_CQ_H_
#define SHERPA_ONNX_ALSA_CQ_H_

#include <alsa/asoundlib.h>
#include <thread>
#include <mutex>
#include <vector>
#ifdef RK3588
    #include "rknn_api.h"
#endif
struct online_audio
{
    std::vector<float> CQ_buffer;
    std::vector<float> pcmf32_new;
};

struct online_params
{
    int32_t step_ms = 0;
    int32_t keep_ms = 0;
    int32_t CQ_audio_entrance = 0;
    int32_t CQ_audio_exit = 0;
    int32_t sample_rate=16000;
    
    #ifdef RK3588
        rknn_context rk_ctx;
    #endif


    std::mutex m_mutex;

    int32_t n_samples_30s = (1e-3 * 30000.0) * sample_rate;
 
    bool is_running = false;

    online_audio audio;
 

};
int32_t audio_CQ_init(const char *capture_id, int sample_rate, online_params *params, snd_pcm_t *capture_handle);
int32_t audio_CQ_push(online_params *params, snd_pcm_t *ahandler);
int32_t audio_CQ_get(online_params *params,int get_ms,int keep_ms);
int32_t audio_CQ_view(online_params *params,int get_ms,int keep_ms);
int32_t audio_CQ_length(online_params *params);
int32_t audio_CQ_clear(online_params *params);
int32_t init_online_audio(online_params *params);

#endif  // SHERPA_ONNX_ALSA_CQ_H_
