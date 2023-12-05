#ifndef SHERPA_ONNX_ALSA_CQ_H_
#define SHERPA_ONNX_ALSA_CQ_H_

#include <alsa/asoundlib.h>
#include <thread>
#include <mutex>
#include "vad_stt_online.h"

int32_t audio_CQ_init(const char *capture_id, int sample_rate, online_params *params, snd_pcm_t *capture_handle);
int32_t audio_CQ_push(online_params *params, snd_pcm_t *ahandler);
int32_t audio_CQ_get(online_params *params,int get_ms,int keep_ms);
int32_t audio_CQ_view(online_params *params,int get_ms,int keep_ms);
int32_t audio_CQ_length(online_params *params);
int32_t audio_CQ_clear(online_params *params);

#endif  // SHERPA_ONNX_ALSA_CQ_H_
