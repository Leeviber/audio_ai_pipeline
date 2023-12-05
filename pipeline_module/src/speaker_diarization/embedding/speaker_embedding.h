#ifndef SPEAKER_EMBEDDING
#define SPEAKER_EMBEDDING
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

#include "speaker_help.h"

 
std::vector<std::vector<double>> getEmbedding( std::shared_ptr<wespeaker::SpeakerEngine> engine, const std::vector<std::vector<float>>& dataChunks, 
        const std::vector<std::vector<float>>& masks);
        
#endif