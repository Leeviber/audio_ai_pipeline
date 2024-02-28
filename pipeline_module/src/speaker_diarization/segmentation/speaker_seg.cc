#include "speaker_seg.h"


std::vector<float> max_segmentation_cluster(const std::vector<std::vector<float>>& segmentation,
                                       const std::vector<int>& cluster, int k) 
{
    std::vector<float> maxValues( segmentation.size());
    for (size_t i = 0; i < segmentation.size(); ++i) 
    {
        float maxValue = -std::numeric_limits<float>::infinity();

        for (size_t j = 0; j < cluster.size(); ++j) 
        {
            if (cluster[j] == k) 
            {
                maxValue = std::max(maxValue, segmentation[i][j]);
            }
        }
        maxValues[i] = maxValue;
    }

    return maxValues;
}

std::vector<std::vector<double>> reconstruct( 
        const std::vector<std::vector<std::vector<float>>>& segmentations,
        const SlidingWindow& segmentations_frames,
        const std::vector<std::vector<int>>& hard_clusters, 
        const std::vector<int>& count_data,
        const SlidingWindow& count_frames,
        SlidingWindow& activations_frames)
{
    size_t num_chunks = segmentations.size();
    size_t num_frames = segmentations[0].size();
    size_t local_num_speakers = segmentations[0][0].size();

    // python: num_clusters = np.max(hard_clusters) + 1
    // Note, element in hard_clusters have negative number, don't define num_cluster as size_t
    int num_clusters = 0;
    for( size_t i = 0; i < hard_clusters.size(); ++i )
    {
        for( size_t j = 0; j < hard_clusters[0].size(); ++j )
        {
            if( hard_clusters[i][j] > num_clusters )
                num_clusters = hard_clusters[i][j];
        }
    }
    num_clusters++;
    assert( num_clusters > 0 );

    // python: for c, (cluster, (chunk, segmentation)) in enumerate(...
    std::vector<std::vector<std::vector<double>>> clusteredSegmentations( num_chunks, 
            std::vector<std::vector<double>>( num_frames, std::vector<double>( num_clusters, NAN)));
    for( size_t i = 0; i < num_chunks; ++i ) 
    {
        const auto& cluster = hard_clusters[i];
        const auto& segmentation = segmentations[i];
        for( auto k : cluster )
        {
            if( abs( k + 2 ) < std::numeric_limits<double>::epsilon()) // check if it equals -2
            {
                continue;
            }

            auto max_sc = max_segmentation_cluster( segmentation, cluster, k );
            assert( k < num_clusters );
            assert( max_sc.size() > 0 );
            assert( max_sc.size() == num_frames );
            for( size_t m = 0; m < num_frames; ++m )
            {
                clusteredSegmentations[i][m][k] = max_sc[m];
            }
        }
    }

 

    std::vector<std::vector<double>> diarizationRes;
    to_diarization( clusteredSegmentations, segmentations_frames, 
            count_data, count_frames, activations_frames, diarizationRes );
    return diarizationRes;
}


bool to_diarization( std::vector<std::vector<std::vector<double>>>& segmentations, 
        const SlidingWindow& segmentations_frames,
        const std::vector<int>& count,
        const SlidingWindow& count_frames, 
        SlidingWindow& to_diarization_frames,
        std::vector<std::vector<double>>& binary)
{
    // python: activations = Inference.aggregate(...
    SlidingWindow activations_frames;
    auto activations = PipelineHelper::aggregate( segmentations, 
            segmentations_frames, 
            count_frames, 
            activations_frames, 
            false, 0.0, true );

#ifdef WRITE_DATA
    debugWrite2d( activations, "cpp_to_diarization_activations" );

    /*
    std::ifstream fd("/tmp/py_to_diarization_activations.txt"); //taking file as inputstream
    int cnt = 0;
    for( std::string line; getline( fd, line ); )
    {
        std::string delimiter = ",";
        std::vector<std::string> v = Helper::split(line, delimiter);
        v.pop_back();
        for( size_t i = 0; i < v.size(); ++i )
        {
            activations[cnt][i] = std::stof( v[i] );
        }
        cnt++;
    }
    */
#endif 

    // python: _, num_speakers = activations.data.shape
    size_t num_speakers = activations[0].size();

    // python: count.data = np.minimum(count.data, num_speakers)
    // here also convert 1d to 2d later need pass to crop_segment
    std::vector<std::vector<int>> converted_count( count.size(), std::vector<int>( 1 ));
    for( size_t i = 0; i < count.size(); ++i )
    {
        if( count[i] > num_speakers )
            converted_count[i][0] = num_speakers;
        else
            converted_count[i][0] = count[i];
    }

    // python: extent = activations.extent & count.extent
    // get extent then calc intersection, check extent() of 
    // SlidingWindowFeature and __and__() of Segment
    // Get activations.extent
    double tmpStart = activations_frames.start + (0 - .5) * activations_frames.step + 
        .5 * activations_frames.duration;
    double duration = activations.size() * activations_frames.step;
    double activations_end = tmpStart + duration;
    double activations_start = activations_frames.start;

    // Get count.extent
    tmpStart = count_frames.start + (0 - .5) * count_frames.step + .5 * count_frames.duration;
    duration = count.size() * count_frames.step;
    double count_end = tmpStart + duration;
    double count_start = count_frames.start;

    // __and__(), max of start, min of end
    double intersection_start = std::max( activations_start, count_start );
    double intersection_end = std::min( activations_end, count_end );
    Segment focus( intersection_start, intersection_end );
    SlidingWindow cropped_activations_frames;
    auto cropped_activations = crop_segment( activations, activations_frames, focus, 
            cropped_activations_frames );

    SlidingWindow cropped_count_frames;
    auto cropped_count = crop_segment( converted_count, count_frames, focus, 
            cropped_count_frames );

#ifdef WRITE_DATA
    debugWrite2d( cropped_activations, "cpp_cropped_activations" );
    debugWrite2d( cropped_count, "cpp_cropped_count" );
#endif // WRITE_DATA

    // python: sorted_speakers = np.argsort(-activations, axis=-1)
    std::vector<std::vector<int>> sorted_speakers( cropped_activations.size(),
            std::vector<int>( cropped_activations[0].size()));
    int ss_index = 0;
    for( auto& a : cropped_activations )
    {
        // -activations
        for( size_t i = 0; i < a.size(); ++i ) a[i] = -1.0 * a[i];
        auto indices = Helper::argsort( a );
        sorted_speakers[ss_index++].swap( indices );
    }
#ifdef WRITE_DATA
    debugWrite2d( sorted_speakers, "cpp_sorted_speakers" );
#endif // WRITE_DATA

    assert( cropped_activations.size() > 0 );
    assert( cropped_activations[0].size() > 0 );

    // python: binary = np.zeros_like(activations.data)
    binary.resize( cropped_activations.size(),
        std::vector<double>( cropped_activations[0].size(), 0.0 ));

    // python: for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
    // NOTE: here c is data of count, not sliding window, see __next__ of SlidingWindowFeature
    // in python code
    assert( cropped_count.size() <= sorted_speakers.size());

    // following code based on this cropped_count is one column data, if not 
    // need below code
    assert( cropped_count[0].size() == 1 );
    for( size_t i = 0; i < cropped_count.size(); ++i )
    {
        int k = cropped_count[i][0];
        assert( k <= binary[0].size());
        for( size_t j = 0; j < k; ++j )
        {
            assert( sorted_speakers[i][j] < cropped_count.size());
            binary[i][sorted_speakers[i][j]] = 1.0f;
        }
    }

    to_diarization_frames = cropped_activations_frames;

    return true;
}

std::vector<int16_t> floatToShort(const std::vector<float>& floatVector) {
    // Initialize the resulting int16_t vector
    std::vector<int16_t> shortVector;
    shortVector.reserve(floatVector.size());

    // Determine the scaling factor to convert floats to int16_t
    float scale = static_cast<float>(std::numeric_limits<int16_t>::max());

    // Convert each float value to int16_t
    for (float value : floatVector) {
        // Convert float to int16_t and ensure it is within the range [-32768, 32767]
        int16_t intValue = static_cast<int16_t>(std::round(value * scale));
        if (intValue > std::numeric_limits<int16_t>::max()) {
            intValue = std::numeric_limits<int16_t>::max();
        } else if (intValue < std::numeric_limits<int16_t>::min()) {
            intValue = std::numeric_limits<int16_t>::min();
        }

        // Store the converted value in the resulting vector
        shortVector.push_back(intValue);
    }

    return shortVector;
}


std::vector<float> countSpeakerProbabilities(const std::vector<std::vector<std::vector<double>>> &binarized)
{
    int num_chunk = binarized.size();
    int num_frame = binarized[0].size();
    int num_speaker = binarized[0][0].size();

    std::vector<int> speaker_counts(num_speaker, 0);
    int frame_total = num_chunk * num_frame;
    printf("frame_total%d \n",frame_total);

    // 计算每个说话者在所有 chunk 中为 1 的次数
    for (int i = 0; i < num_chunk; ++i)
    {
        for (int j = 0; j < num_frame; ++j)
        {
            for (int k = 0; k < num_speaker; ++k)
            {
                speaker_counts[k] += binarized[i][j][k];
            }
        }
    }

    // 将次数转换为概率
    std::vector<float> speaker_probabilities(num_speaker, 0.0f);
    for (int k = 0; k < num_speaker; ++k)
    {
        speaker_probabilities[k] = static_cast<float>(speaker_counts[k]) / frame_total;
        printf("speaker_probabilities%f \n",speaker_probabilities[k]);
    }

    return speaker_probabilities;
}

bool shouldProcess(const std::vector<float> &speaker_counts, float threshold)
{
    int count_above_threshold = 0;

    for (int i = 0; i < speaker_counts.size(); ++i)
    {
        if (speaker_counts[i] > threshold)
        {
            count_above_threshold++;
            if (count_above_threshold >= 2)
            {
                return true; // 如果有两个以上的数字大于阈值，则处理数据
            }
        }
    }

    return false; // 如果不满足条件，则不处理数据
}


// 运行分割模型并返回模型输出
std::vector<std::vector<std::vector<double>>> runSegmentationModel(SegmentModel *mm,const std::vector<float> &input_wav,
                                                                   std::vector<std::pair<double, double>> &segments,
                                                                   std::vector<std::vector<std::vector<float>>> &segmentations,
                                                                   SlidingWindow &res_frames
                                                                   )
{
 
    // 运行分割模型
    // SlidingWindow res_frames;
    segmentations = mm->slide(input_wav, res_frames);
    auto segment_data = res_frames.data();
    for (auto seg : segment_data)
    {
        segments.emplace_back(std::make_pair(seg.start, seg.end));
    }
    auto binarized = mm->binarize_swf(segmentations, false);

    return binarized;
}

// Define a struct to represent annotations
Annotation to_annotation( const std::vector<std::vector<double>>& scores,
        const SlidingWindow& frames,
        double onset, double offset, 
        double min_duration_on, double min_duration_off)
{
    // call binarize : pyannote/audio/utils/signal.py: 287
    size_t num_frames = scores.size();
    size_t num_classes = scores[0].size();

    // python: timestamps = [frames[i].middle for i in range(num_frames)]
    std::vector<double> timestamps( num_frames );
    for( size_t i = 0; i < num_frames; ++i )
    {
        double start = frames.start + i * frames.step;
        double end = start + frames.duration;
        timestamps[i] = ( start + end ) / 2;
    }

    // python: socre.data.T
    std::vector<std::vector<double>> inversed( num_classes, std::vector<double>( num_frames ));
    for( size_t i = 0; i < num_frames; ++i )
    {
        for( size_t j = 0; j < num_classes; ++j )
        {
            inversed[j][i] = scores[i][j];
        }
    }

    Annotation active;
    double pad_onset = 0.0;
    double pad_offset = 0.0;
    for( size_t i = 0; i< num_classes; ++i )
    {
        int label = i;
        double start = timestamps[0];
        bool is_active = false;
        if( inversed[i][0] > onset )
        {
            is_active = true;
        }
        for( size_t j = 1; j < num_frames; ++j )
        {
            // currently active
            if(is_active)
            {
                // switching from active to inactive
                if( inversed[i][j] < offset )
                {
                    Segment region(start - pad_onset, timestamps[j] + pad_offset);
                    active.addSegment(region.start, region.end, label);
                    start = timestamps[j];
                    is_active = false;
                }
            }
            else
            {
                if( inversed[i][j] > onset )
                {
                    start = timestamps[j];
                    is_active = true;
                }
            }
        }

        if(is_active)
        {
            Segment region(start - pad_onset, timestamps.back() + pad_offset);
            active.addSegment(region.start, region.end, label);
        }
    }

    // because of padding, some active regions might be overlapping: merge them.
    // also: fill same speaker gaps shorter than min_duration_off
    if( pad_offset > 0.0 || pad_onset > 0.0  || min_duration_off > 0.0 )
        active.support( min_duration_off );

    // remove tracks shorter than min_duration_on
    if( min_duration_on > 0 )
    {
        active.removeShort( min_duration_on );
    }

    return active;
}



 
void mergeSegments(const std::vector<Annotation::Result> &results, const std::vector<float> &input_wav,
                   std::map<int, std::vector<Annotation::Result>> &mergedResults,
                   std::vector<std::vector<float>> &audioSegments,
                   std::vector<Annotation::Result> &allSegment)
{
    // std::map<int, std::vector<Annotation::Result>> mergedResults;

    // Group segments by label
    for (const auto &result : results)
    {
        mergedResults[result.label].push_back(result);
    }
    // std::vector<std::vector<float>> audioSegments;

    // Merge segments within each label and extract audio segments
    for (auto &pair : mergedResults)
    {
        int label = pair.first;
        std::vector<Annotation::Result> &segmentList = pair.second;

        std::vector<Annotation::Result> mergedSegments;
        // mergedSegments.push_back(segmentList[0]);

        // Iterate over segments and merge
        std::vector<float> audioSegment;
        double mergedEnd = -1.0; // Initialize with an invalid value
        for (const auto &segment : segmentList)
        {
            // Calculate the start and end samples based on the sample rate
            int startSample = static_cast<int>(segment.start * 16000);
            int endSample = static_cast<int>(segment.end * 16000);

            // Extract the audio segment based on the start and end samples
            std::vector<float> segmentAudio(input_wav.begin() + startSample, input_wav.begin() + endSample);

            if (mergedEnd != -1.0 && segment.start - mergedEnd <= 1.5)
            {
                // Merge segments if the time difference is less than or equal to 1.5 seconds
                audioSegment.insert(audioSegment.end(), segmentAudio.begin(), segmentAudio.end());
                mergedSegments.back().end = segment.end;
            }
            else
            {
                // Push the previous audio segment if it exists
                if (!audioSegment.empty())
                {
                    audioSegments.push_back(audioSegment);
                    audioSegment.clear();
                }
                // Start a new audio segment
                audioSegment = segmentAudio;
                mergedSegments.push_back(segment);
            }

            // Update the merged end time
            mergedEnd = segment.end;
        }

        // Push the last audio segment
        if (!audioSegment.empty())
        {
            audioSegments.push_back(audioSegment);
        }

        // Store the audio segments in the result map
        pair.second = mergedSegments;
    }

    for (auto &pair : mergedResults)
    {

        std::vector<Annotation::Result> &segmentList = pair.second;

        for (const auto &segment : segmentList)
        {
            allSegment.push_back(segment);
            printf("segment id%d\n", segment.label);
            printf("segment start:%f, end:%f\n", segment.start, segment.end);
        }
    }
}


std::vector<int> mergeAndRenumberNumbers(const std::vector<int> &numbers)
{
    std::unordered_map<int, int> index_map;
    std::vector<int> unique_numbers;

    // 获取唯一的数字并保留其首次出现的顺序
    for (int num : numbers)
    {
        if (index_map.find(num) == index_map.end())
        {
            index_map[num] = static_cast<int>(unique_numbers.size());
            unique_numbers.push_back(num);
        }
    }

    // 根据唯一的数字及其首次出现的顺序进行重新编号
    std::vector<int> result;
    for (int num : numbers)
    {
        result.push_back(index_map[num]);
    }

    return result;
}

std::string secondsToMinutesAndSeconds(double seconds) {
    int minutes = static_cast<int>(seconds) / 60;
    int remainingSeconds = static_cast<int>(seconds) % 60;
    double fractionalSeconds = seconds - static_cast<int>(seconds);
    int roundedFractionalSeconds = static_cast<int>(fractionalSeconds * 100);

    std::string result = std::to_string(minutes) + "分" + std::to_string(remainingSeconds) + "秒" + std::to_string(roundedFractionalSeconds) + "毫秒";
    return result;
}