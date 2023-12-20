 

#include <stdio.h>
#include <chrono>  
#include <string>
#include <vector>

#include "ai_vad.h"        
#include "speaker_diary.h"
#include "speaker_diarization/frontend/wav.h"

// np.max( segmentation[:, cluster == k], axis=1 )
std::vector<float> max_segmentation_cluster(const std::vector<std::vector<float>>& segmentation,
                                       const std::vector<int>& cluster, int k) 
{
    std::vector<float> maxValues( segmentation.size());
    std::cout<<"clu"<<cluster.size()<<std::endl;
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

template<typename T>
std::vector<std::vector<T>> crop_segment( const std::vector<std::vector<T>>& data,
        const SlidingWindow& src, const Segment& focus, SlidingWindow& resFrames )
{
    size_t n_samples = data.size();
    // python: ranges = self.sliding_window.crop(
    // As we pass in Segment, so there would on range returned, here we use following
    // block code to simulate sliding_window.crop <-- TODO: maybe move following block into SlidingWindow class
    // { --> start
        // python: i_ = (focus.start - self.duration - self.start) / self.step
        float i_ = (focus.start - src.duration - src.start) / src.step;

        // python: i = int(np.ceil(i_))
        int rng_start = ceil(i_);
        if( rng_start < 0 )
            rng_start = 0;

        // find largest integer j such that
        // self.start + j x self.step <= focus.end
        float j_ = (focus.end - src.start) / src.step;
        int rng_end = floor(j_) + 1;
    // } <-- end 
    //size_t cropped_num_samples = ( rng_end - rng_start ) * m_sample_rate;
    float start = src[rng_start].start;
    SlidingWindow res( start, src.step, src.duration, n_samples );
    //auto segments = res.data();
    std::vector<Segment> segments;
    segments.push_back( Segment( rng_start, rng_end ));
    
    int n_dimensions = 1;
    // python: for start, end in ranges:
    // ***** Note, I found ranges is always 1 element returned from self.sliding_window.crop
    // if this is not true, then need change following code. Read code:
    // pyannote/core/feature.py:196
    std::vector<std::pair<int, int>> clipped_ranges;
    for( auto segment : segments )
    {
        size_t start = segment.start;
        size_t end = segment.end;

        // if all requested samples are out of bounds, skip
        if( end < 0 || start >= n_samples)
        {
            continue;
        }
        else
        {
            // keep track of non-empty clipped ranges
            // python: clipped_ranges += [[max(start, 0), min(end, n_samples)]]
            clipped_ranges.emplace_back( std::make_pair( std::max( start, 0ul ), std::min( end, n_samples )));
        }
    }
    resFrames = res;
    std::vector<std::vector<T>> cropped_data;

    // python: data = np.vstack([self.data[start:end, :] for start, end in clipped_ranges])
    for( const auto& pair : clipped_ranges )
    {
        for( int i = pair.first; i < pair.second; ++i )
        {
            std::vector<T> tmp;
            for( size_t j = 0; j < data[i].size(); ++j )
                tmp.push_back( data[i][j] );
            cropped_data.push_back( tmp );
        }
    }

    return cropped_data;
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
// pyannote/audio/pipelines/speaker_diarization.py:403, def reconstruct(
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
// Define a struct to represent annotations
struct Annotation 
{
    struct Result
    {
        double start;
        double end;
        int label;
        Result( double start, double end, int label )
            : start( start )
            , end( end )
            , label( label )
        {}
    };
    struct Track 
    {
        std::vector<Segment> segments;
        int label;

        Track( int label )
            : label( label )
        {}

        Track& operator=( const Track& other )
        {
            segments = other.segments;
            label = other.label;

            return *this;
        }

        Track( const Track& other )
        {
            segments = other.segments;
            label = other.label;
        }

        Track( Track&& other )
        {
            segments = std::move( other.segments );
            label = other.label;
        }

        void addSegment(double start, double end ) 
        {
            segments.push_back( Segment( start, end ));
        }

        void support( double collar )
        {
            // Must sort first
            std::sort( segments.begin(), segments.end(), []( const Segment& s1, const Segment& s2 ){
                        return s1.start < s2.start;
                    });
            if( segments.size() == 0 )
                return;
            std::vector<Segment> merged_segments;
            Segment curSeg = segments[0];
            bool merged = true;
            for( size_t i = 1; i < segments.size(); ++i )
            {
                // WHYWHY must assign to tmp object, otherwise
                // in gap function, its value like random
                auto next = segments[i];
                double gap = curSeg.gap( next );
                if( gap < collar )
                {
                    curSeg = curSeg.merge( segments[i] );
                }
                else
                {
                    merged_segments.push_back( curSeg );
                    curSeg = segments[i];
                }
            }
            merged_segments.push_back( curSeg );

            segments.swap( merged_segments );
        }

        void removeShort( double min_duration_on )
        {
            for( size_t i = 1; i < segments.size(); ++i )
            {
                if( segments[i].duration() < min_duration_on )
                {
                    segments.erase( segments.begin() + i );
                    i--;
                }
            }
        }
    };

    std::vector<Track> tracks;

    Annotation()
        : tracks()
    {}

    std::vector<Result> finalResult()
    {
        std::vector<Result> results;
        for( const auto& track : tracks )
        {
            for( const auto& segment : track.segments )
            {
                Result res( segment.start, segment.end, track.label );
                results.push_back( res );
            }
        }
        std::sort( results.begin(), results.end(), []( const Result& s1, const Result& s2 ){
                    return s1.start < s2.start;
                });

        return results;
    }

    void addSegment(double start, double end, int label) 
    {
        for( auto& tk : tracks )
        {
            if( tk.label == label )
            {
                tk.addSegment( start, end );
                return;
            }
        }

        // Not found, create new track
        Track tk( label );
        tk.addSegment( start, end );
        tracks.push_back( tk );
    }

    Annotation& operator=( const Annotation& other )
    {
        tracks = other.tracks;

        return *this;
    }

    Annotation( Annotation&& other )
    {
        tracks = std::move( tracks );
    }

    void removeShort( double min_duration_on )
    {
        for( auto& track : tracks )
        {
            track.removeShort( min_duration_on );
        }
    }

    // pyannote/core/annotation.py:1350
    void support( double collar )
    {
        // python: timeline = timeline.support(collar)
        // pyannote/core/timeline.py:845
        for( auto& track : tracks )
        {
            track.support( collar );
        }
    }
};


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
// int32_t init_online_audio(online_params *params)
// {

//     params->is_running = true;
 

//     online_audio audio_buffer;
//     audio_buffer.pcmf32_new = std::vector<float>(params->n_samples_30s, 0.0f);
//     audio_buffer.CQ_buffer.resize(sample_rate * 30);
//     params->audio = audio_buffer;
//     float value;
 
//     return 0;
// }

int16_t float32ToInt16(float value) {
    // 将 float32 数据范围 [-1, 1] 映射到 int16 数据范围 [-32767, 32767]
    return static_cast<int16_t>(std::round(value * 32767.0f));
}

void convertFloat32ToInt16(const float* floatData, int16_t* intData, size_t numSamples) {
    for (size_t i = 0; i < numSamples; ++i) {
        intData[i] = float32ToInt16(floatData[i]);
    }
}

void saveOneDimArrayToBinaryFile(const std::vector<int>& array, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "Failed to open file for writing." << std::endl;
        return;
    }

    // 获取数组的长度
    int size = array.size();

    // 将一维数组写入文件
    file.write(reinterpret_cast<const char*>(array.data()), size * sizeof(int));

    file.close();
    std::cout << "One-dimensional array saved to binary file: " << filename << std::endl;
}
void saveArrayToBinaryFile(const std::vector<std::vector<std::vector<double>>>& array, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "Failed to open file for writing." << std::endl;
        return;
    }

    // 获取数组的维度
    int dim1 = array.size();
    int dim2 = (dim1 > 0) ? array[0].size() : 0;
    int dim3 = (dim2 > 0) ? array[0][0].size() : 0;

    // 写入数组数据
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                double value = array[i][j][k];
                file.write(reinterpret_cast<const char*>(&value), sizeof(double));
            }
        }
    }

    file.close();
    std::cout << "Array saved to binary file: " << filename << std::endl;
}
 void saveFloatArrayToBinaryFile(const std::vector<std::vector<std::vector<float>>>& array, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "Failed to open file for writing." << std::endl;
        return;
    }

    // 获取数组的维度
    int dim1 = array.size();
    int dim2 = (dim1 > 0) ? array[0].size() : 0;
    int dim3 = (dim2 > 0) ? array[0][0].size() : 0;

    // 写入数组数据
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                float value = array[i][j][k];
                file.write(reinterpret_cast<const char*>(&value), sizeof(float));
            }
        }
    }

    file.close();
    std::cout << "Array saved to binary file: " << filename << std::endl;
}

int main() {

    std::vector<std::string> model_paths;
    std::string onnx_model_path ="./bin/voxceleb_resnet34_LM.onnx";
    model_paths.push_back(onnx_model_path);

    // int embedding_size=256;
    // int feat_dim = 80;
    // int SamplesPerChunk = 32000;
    // std::cout<<"here1"<<std::endl;

    // auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
    //     model_paths, feat_dim, 16000,
    //     embedding_size, SamplesPerChunk);
    // std::vector<float> last_embs(embedding_size, 0);
    // std::vector<float> current_embs(embedding_size, 0);

    int sample_chunk_ms=1000;


    // std::vector<float> enroll_embs(embedding_size, 0);



    int embedding_batch_size = 32;
    double self_frame_step = 0.016875;
    double self_frame_duration = 0.016875;
    double self_frame_start = 0.0;
    size_t min_num_samples = 640;

    int vad_begin = 0;
    int vad_end = 0;

    int32_t vad_activate_sample = (16000 * 500) / 1000;
    int32_t vad_silence_sample = (16000 * 0) / 1000;

    std::string path = "./bin/silero_vad.onnx";
    int test_sr = 16000;
    int test_frame_ms = 96;
    float test_threshold = 0.85f;
    int test_min_silence_duration_ms = 100;
    int test_speech_pad_ms = 0;
    int test_window_samples = test_frame_ms * (test_sr / 1000);

    VadIterator ai_vad(
        path, test_sr, test_frame_ms, test_threshold,
        test_min_silence_duration_ms, test_speech_pad_ms);
 
    ////////////////////////////////////////////////////////////////////////////////
    //// Init ALSA and circular buffer////
    // snd_pcm_t *capture_handle;
    // const char *device_name ="plughw:2,0";   // using arecord -l to checkout the alsa device name
    // ret = audio_CQ_init(device_name, sample_rate, &params, capture_handle);
    // printf("Started\n");
    // ai_vad.reset_states();
 
    wav::WavReader wav_reader("./bin/output.wav");
    int num_channels = wav_reader.num_channels();
    int bits_per_sample = wav_reader.bits_per_sample();
    int sample_rate = wav_reader.sample_rate();
    const float* audio = wav_reader.data();
    int num_samples = wav_reader.num_samples();
    std::vector<float> input_wav{audio, audio + num_samples};
    for( int i = 0; i < num_samples; ++i )
    {
        input_wav[i] = input_wav[i]*1.0f/32768.0;
    }

    printf("input_wav size%d",input_wav.size());

    // printf("sample_rate %d",sample_rate);
    const std::string segmentModel="./bin/seg_model.onnx";
    // int embedding_size = 512;
    auto start_time = std::chrono::high_resolution_clock::now();

    SegmentModel mm(segmentModel);
    std::vector<std::pair<double, double>> segments;
    SlidingWindow res_frames;
    auto segmentations = mm.slide( input_wav, res_frames );
    auto segment_data = res_frames.data();
    for( auto seg : segment_data )
    {
        segments.emplace_back( std::make_pair( seg.start, seg.end ));
    }
    std::cout<<segmentations.size()<<"x"<<segmentations[0].size()<<"x"<<segmentations[0][0].size()<<std::endl;
    // saveFloatArrayToBinaryFile(segmentations, "segmentations.bin");
 
    auto binarized = mm.binarize_swf( segmentations, false );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "完整运行时间为: " << time.count() << " 毫秒" << std::endl;

    assert( binarized.size() == segments.size());

    // estimate frame-level number of instantaneous speakers
    // In python code, binarized in speaker_count function is cacluated with 
    // same parameters as we did above, so we reuse it by passing it into speaker_count
    // SlidingWindow count_frames( num_samples );
    // SlidingWindow pre_frame( self_frame_start, self_frame_step, self_frame_duration );
    // auto count_data = mm.speaker_count( segmentations, binarized, 
    //         pre_frame, count_frames, num_samples );
    // std::cout << "count_data: " << count_data.size() << std::endl;
    // std::cout << "count_data[0]: " << count_data[0]<< std::endl;
    // saveOneDimArrayToBinaryFile(count_data,"count.bin");
    // python: duration = binary_segmentations.sliding_window.duration
    double duration = 5.0;
    size_t num_chunks = binarized.size();
    size_t num_frames = binarized[0].size(); 

    // python: num_samples = duration * self._embedding.sample_rate 
    size_t min_num_frames = ceil(num_frames * min_num_samples / ( duration * 16000 ));
    // std::cout << "min_num_frames: " << min_num_frames << std::endl;
    // std::cout << "num_chunks: " << num_chunks << std::endl;

    // python: clean_frames = 1.0 * ( np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2 )
    // python: clean_segmentations = SlidingWindowFeature( 
    //                               binary_segmentations.data * clean_frames, binary_segmentations.sliding_window )
    saveArrayToBinaryFile(binarized, "binarized.bin");
    printf("binarized (%d,%d,%d)",binarized.size(),binarized[0].size(),binarized[0][0].size());

    auto clean_segmentations = Helper::cleanSegmentations( binarized );
    saveArrayToBinaryFile(clean_segmentations, "clean_segmentations_notsure.bin");

    assert( binarized.size() == clean_segmentations.size());
    std::vector<std::vector<float>> batchData;
    std::vector<std::vector<float>> batchMasks;
    std::vector<std::vector<double>> embeddings;

    printf("clean_segmentations (%d,%d,%d)",clean_segmentations.size(),clean_segmentations[0].size(),clean_segmentations[0][0].size());

    //  for (int i = 0; i < clean_segmentations.size(); ++i) {
    //     for (int j = 0; j < clean_segmentations[i].size(); ++j) {
    //         for (int k = 0; k < clean_segmentations[i][j].size(); ++k) {
    //             if (clean_segmentations[i][j][k] != 0) {
    //                 std::cout << "Non-zero element at index (" << i << "," << j << "," << k << "): " << clean_segmentations[i][j][k] << std::endl;
    //              }
    //         }
    //     }
    // }
    
    //////////////////////////////////////// Embedding //////////////////////////////////////////////////////
    // for( size_t i = 0; i < binarized.size(); ++i )
    // {
    //     auto chunkData = mm.crop( input_wav, segments[i] );
    //     auto& masks = binarized[i];
    //     auto& clean_masks = clean_segmentations[i];
    //     assert( masks[0].size() == 3 );
    //     assert( clean_masks[0].size() == 3 );
    //     // printf("chunkData size %d\n",chunkData.size());
    //     // printf("seg start %f\n",segments[i].first);
    //     // printf("seg end %f\n",segments[i].second);


    //     // python: for mask, clean_mask in zip(masks.T, clean_masks.T):
    //     for( size_t j = 0; j < clean_masks[0].size(); ++j )
    //     {
    //         std::vector<float> used_mask;
    //         float sum = 0.0;
    //         std::vector<float> reversed_clean_mask(clean_masks.size());
    //         std::vector<float> reversed_mask(masks.size());

    //         // python: np.sum(clean_mask)
    //         for( size_t k = 0; k < clean_masks.size(); ++k )
    //         {
 
    //             sum += clean_masks[k][j];
    //             reversed_clean_mask[k] = clean_masks[k][j];
    //             reversed_mask[k] = masks[k][j];
    //         }
    //         // printf("min_num_frames %d\n",min_num_frames);
 
    //         if( sum > min_num_frames )
    //         {
    
    //             used_mask = std::move( reversed_clean_mask );
    //         }
    //         else
    //         {
    //             used_mask = std::move( reversed_mask );
    //         }

    //         // batchify
    //         batchData.push_back( chunkData );
    //         batchMasks.push_back( std::move( used_mask ));
    //         if( batchData.size() == embedding_batch_size )
    //         {
    //             auto embedding = getEmbedding( speaker_engine, batchData, batchMasks );
    //             batchData.clear();
    //             batchMasks.clear();

    //             for( auto& a : embedding )
    //             {
    //                 embeddings.push_back( std::move( a ));
    //             }
    //         }
    //     }
    // }
    // Process remaining
    // if( batchData.size() > 0 )
    // {
    //     auto embedding = getEmbedding( speaker_engine, batchData, batchMasks );
    //     for( auto& a : embedding )
    //     {
    //         embeddings.push_back( std::move( a ));
    //     }
    // }
    // std::cout << "embeddings result size" << embeddings.size()<<","<<embeddings[0].size() << std::endl;
    // auto embeddings1 = Helper::rearrange_up( embeddings, num_chunks );
    // std::cout << "embeddings1 result size" << embeddings1.size()<<","<<embeddings1[0].size()<<","<<embeddings1[0][0].size() << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////



     /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Cluster cst;
    // std::vector<std::vector<int>> hard_clusters; // output 1 for clustering
    // cst.clustering(embeddings1,binarized,hard_clusters);
    // assert( hard_clusters.size() == binarized.size());
    // assert( hard_clusters[0].size() == binarized[0][0].size());
    // std::vector<std::vector<float>> inactive_speakers( binarized.size(),
    //         std::vector<float>( binarized[0][0].size(), 0.0));
    // for( size_t i = 0; i < binarized.size(); ++i )
    // {
    //     for( size_t j = 0; j < binarized[0].size(); ++j )
    //     {
    //         for( size_t k = 0; k < binarized[0][0].size(); ++k )
    //         {
    //             inactive_speakers[i][k] += binarized[i][j][k];
    //         }
    //     }
    // }   
    // for( size_t i = 0; i < inactive_speakers.size(); ++i )
    // {
    //     for( size_t j = 0; j < inactive_speakers[0].size(); ++j )
    //     {
    //         if( abs( inactive_speakers[i][j] ) < std::numeric_limits<double>::epsilon())
    //             hard_clusters[i][j] = -2;
    //     }
    // }

    // SlidingWindow activations_frames;
    // auto discrete_diarization = reconstruct( segmentations, res_frames, 
    //         hard_clusters, count_data, count_frames, activations_frames );


    // float diarization_segmentation_min_duration_off = 0.5817029604921046; // see SegmentModel
    // auto diarization = to_annotation(discrete_diarization, 
    //         activations_frames, 0.5, 0.5, 0.0, 
    //         diarization_segmentation_min_duration_off);

    // std::cout<<"----------------------------------------------------"<<std::endl;
    // auto diaRes = diarization.finalResult();
    // for( const auto& dr : diaRes )
    // {
    //     std::cout<<"["<<dr.start<<" -- "<<dr.end<<"]"<<" --> Speaker_"<<dr.label<<std::endl;
    // }
    // std::cout<<"----------------------------------------------------"<<std::endl;


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // std::ofstream file("data.bin", std::ios::binary);
    // if (file.is_open()) {
    //     // 逐个写入数组元素
    //     for (const auto& vec1 : clean_segmentations) {
    //         for (const auto& vec2 : vec1) {
    //             file.write(reinterpret_cast<const char*>(vec2.data()),
    //                        vec2.size() * sizeof(double));
    //         }
    //     }
    //     // 关闭文件
    //     file.close();
    // } else {
    //     // 处理文件打开失败的情况
    // }
    // std::cout << "Array saved to binary file: "  << std::endl;

    return 0;
}

 
     double self_frame_step = 0.016875;
