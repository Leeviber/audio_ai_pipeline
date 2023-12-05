#ifndef SPEAKER_ID
#define SPEAKER_ID

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

#include "speaker_id/speaker/speaker_engine.h"
#include "speaker_id/frontend/wav.h"

int32_t init_online_audio(online_params *params);


#endif