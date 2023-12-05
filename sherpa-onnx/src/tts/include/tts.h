#ifndef PIPER_TTS_H_
#define PIPER_TTS_H_
#include "piper.h"
#include <alsa/asoundlib.h>

#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

using json = nlohmann::json;

struct tts_params
{
  piper::PiperConfig piperConfig;
  piper::Voice voice;
  std::string modelPath;
  std::string modelConfigPath;
  std::string outputPath;
  snd_pcm_t *pcm_handle;
  int silence_ms;
  char *play_device;
  std::vector<int16_t> silence_chunk;
};

std::vector<std::string> segmentText(const std::string &text);

void init_tts(tts_params &ttsConfig);

void process_tts(const std::string &line, tts_params &ttsConfig);

#endif