#ifndef VAD_STT_ONLINE
#define VAD_STT_ONLINE
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

#include <algorithm>
#include <numeric>
#include <cmath>
#include <math.h>

#include "kiss_fft.h"
#include "kiss_fftr.h"
// #include "kws_engine.h"


#ifndef DATA_STRUCT_H
#define DATA_STRUCT_H

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
#endif

int32_t init_online_audio(online_params *params);

#endif