#ifndef PIPER_KWS
#define PIPER_KWS
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

#include "kws_stt_online.h"
#include "kiss_fft.h"
#include "kiss_fftr.h"
#include "kws_engine.h"

const float eps = std::numeric_limits<float>::epsilon();

struct kws_params
{
    std::string fb_path;
    std::string emb_path;
    char* eff_model_path;
    rknn_context rk_ctx;
    std::vector<std::vector<float>> mfcc_fb;
    std::vector<std::vector<float>> embedding;
};

int32_t init_kws(kws_params *kws_params);

bool kws_process(online_params *params,kws_params *kws_params);

int32_t deinit_kws(kws_params *kws_params);

#endif

 