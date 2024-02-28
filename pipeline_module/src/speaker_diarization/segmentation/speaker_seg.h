#ifndef SPEAKER_SEG
#define SPEAKER_SEG
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <limits>
#include <algorithm>
#include <chrono>
#include <sstream>  
#include <assert.h>
#include<algorithm>
#include "onnx_model.h"
#include "speaker_help.h"



class Segment
{
public:
    double start;
    double end;
    Segment( double start, double end )
        : start( start )
        , end( end )
    {}

    Segment( const Segment& other )
    {
        start = other.start;
        end = other.end;
    }

    Segment& operator=( const Segment& other )
    {
        start = other.start;
        end = other.end;

        return *this;
    }

    double duration() const 
    {
        return end - start;
    }

    double gap( const Segment& other )
    {
        if( start < other.start )
        {
            if( end >= other.start )
            {
                return 0.0;
            }
            else
            {
                return other.start - end;
            }
        }
        else
        {
            if( start <= other.end )
            {
                return 0.0;
            }
            else
            {
                return start - other.end;
            }
        }
    }

    Segment merge( const Segment& other )
    {
        return Segment(std::min( start, other.start ), std::max( end, other.end ));
    }
};



class SlidingWindow
{
public:
    double start;
    double step;
    double duration;
    size_t num_samples;
    double sample_rate;
    SlidingWindow()
        : start( 0.0 )
        , step( 0.0 )
        , duration( 0.0 )
        , num_samples( 0 )
        , sample_rate( 16000 )
    {
    }

    SlidingWindow( size_t num_samples )
        : start( 0.0 )
        , step( 0.0 )
        , duration( 0.0 )
        , num_samples( num_samples )
        , sample_rate( 16000 )
    {
    }

    SlidingWindow( double start, double step, double duration, size_t num_samples = 0 )
        : start( start )
        , step( step )
        , duration( duration )
        , num_samples( num_samples )
        , sample_rate( 16000 )
    {
    }

    SlidingWindow( const SlidingWindow& other )
    {
        start = other.start;
        step = other.step;
        duration = other.duration;
        num_samples = other.num_samples;
        sample_rate = other.sample_rate;
    }

    SlidingWindow& operator=( const SlidingWindow& other )
    {
        start = other.start;
        step = other.step;
        duration = other.duration;
        num_samples = other.num_samples;
        sample_rate = other.sample_rate;

        return *this;
    }

    size_t closest_frame( double start )
    {
        double closest = ( start - this->start - .5 * duration ) / step;
        if( closest < 0.0 )
            closest = 0.0;
        return Helper::np_rint( closest );
    }

    Segment operator[]( int pos ) const
    {
        int window_size = std::round(duration * sample_rate); // 80000
        int step_size = std::round(step * sample_rate); // 8000
        // python: start = self.__start + i * self.__step
        //double start = this->start + pos * this->step;
        double start = 0.0;
        size_t cur_frames = 0;
        int index = 0;
        while( true )
        {
            if( index == pos )
                return Segment( start, start + duration );
            if( cur_frames + window_size >= num_samples )
            {
                break;
            }
            start += step;
            cur_frames += step_size;
            index++;
        }

        return Segment(0.0, 0.0);
    }

    std::vector<Segment> data()
    {
        std::vector<Segment> segments;
        int window_size = std::round(duration * sample_rate); // 80000
        int step_size = std::round(step * sample_rate); // 8000
        double start = 0.0;
        size_t cur_frames = 0;
        while( true )
        {
            Segment seg( start, start + duration );
            segments.push_back( seg );
            if( cur_frames + window_size >= num_samples )
            {
                break;
            }
            start += step;
            cur_frames += step_size;
        }

        return segments;
    }
};


class PipelineHelper
{
public:
    // pyannote/audio/core/inference.py:411
    // we ignored warm_up parameter since our case use default value( 0.0, 0.0 )
    // so hard code warm_up
    static std::vector<std::vector<double>> aggregate( 
            const std::vector<std::vector<std::vector<double>>>& scoreData, 
            const SlidingWindow& scores_frames, 
            const SlidingWindow& pre_frames, 
            SlidingWindow& post_frames,
            bool hamming = false, 
            double missing = NAN, 
            bool skip_average = false,
            double epsilon = std::numeric_limits<double>::epsilon())
    {
        size_t num_chunks = scoreData.size(); 
        size_t num_frames_per_chunk = scoreData[0].size(); 
        size_t num_classes = scoreData[0][0].size(); 
        size_t num_samples = scores_frames.num_samples;
        assert( num_samples > 0 );

        // create masks 
        std::vector<std::vector<std::vector<double>>> masks( num_chunks, 
                std::vector<std::vector<double>>( num_frames_per_chunk, std::vector<double>( num_classes, 1.0 )));
        auto scores = scoreData;

        // Replace NaN values in scores with 0 and update masks
        // python: masks = 1 - np.isnan(scores)
        // python: scores.data = np.nan_to_num(scores.data, copy=True, nan=0.0)
        for (size_t i = 0; i < num_chunks; ++i) 
        {
            for (size_t j = 0; j < num_frames_per_chunk; ++j) 
            {
                for( size_t k = 0; k < num_classes; ++k )
                {
                    if (std::isnan(scoreData[i][j][k])) 
                    {
                        masks[i][j][k] = 0.0;
                        scores[i][j][k] = 0.0;
                    }
                }
            }
        }

        if( !hamming )
        {
            // python np.ones((num_frames_per_chunk, 1))
            // no need create it, later will directly apply 1 to computation
        }
        else
        {
            // python: np.hamming(num_frames_per_chunk).reshape(-1, 1)
            assert( false ); // no implemented
        }

        // Get frames, we changed this part. In pyannote, it calc frames(self._frames) before calling
        // this function, but in this function, it creates new frames and use it.
        // step = (self.inc_num_samples / self.inc_num_frames) / sample_rate
        // pyannote/audio/core/model.py:243
        // currently cannot find where self.inc_num_samples / self.inc_num_frames from
        /*int inc_num_samples = 270; // <-- this may not be correct
        int inc_num_frames = 1; // <-- this may not be correct
        float frames_step = ( inc_num_samples * 1.0f / inc_num_frames) / g_sample_rate;
        float frames_duration = frames_step;
        float frames_start = scores_frames.start;
        */

        // aggregated_output[i] will be used to store the sum of all predictions
        // for frame #i
        // python: num_frames = ( frames.closest_frame(...)) + 1
        double frame_target = scores_frames.start + scores_frames.duration + (num_chunks - 1) * scores_frames.step;
        SlidingWindow frames( scores_frames.start, pre_frames.step, pre_frames.duration );
        size_t num_frames = frames.closest_frame( frame_target ) + 1;

        // python: aggregated_output: np.ndarray = np.zeros(...
        std::vector<std::vector<double>> aggregated_output(num_frames, std::vector<double>( num_classes, 0.0 ));

        // overlapping_chunk_count[i] will be used to store the number of chunks
        // that contributed to frame #i
        std::vector<std::vector<double>> overlapping_chunk_count(num_frames, std::vector<double>( num_classes, 0.0 ));

        // aggregated_mask[i] will be used to indicate whether
        // at least one non-NAN frame contributed to frame #i
        std::vector<std::vector<double>> aggregated_mask(num_frames, std::vector<double>( num_classes, 0.0 ));
        
        // for our use case, warm_up_window and hamming_window all 1
        double start = scores_frames.start;
        for( size_t i = 0; i < scores.size(); ++i )
        {
            size_t start_frame = frames.closest_frame( start );
            // std::cout<<"start_frame: "<<start_frame<<" with:"<<start<<std::endl;
            start += scores_frames.step; // python: chunk.start
            for( size_t j = 0; j < num_frames_per_chunk; ++j )
            {
                size_t _j = j + start_frame;
                for( size_t k = 0; k < num_classes; ++k )
                {
                    // score * mask * hamming_window * warm_up_window
                    aggregated_output[_j][k] += scores[i][j][k] * masks[i][j][k];
                    overlapping_chunk_count[_j][k] += masks[i][j][k];
                    if( masks[i][j][k] > aggregated_mask[_j][k] )
                    {
                        aggregated_mask[_j][k] = masks[i][j][k];
                    }
                }
            }
        }

#ifdef WRITE_DATA
        debugWrite3d( masks, "cpp_masks_in_aggregate" );
        debugWrite3d( scores, "cpp_scores_in_aggregate" );
        debugWrite2d( aggregated_output, "cpp_aggregated_output" );
        debugWrite2d( aggregated_mask, "cpp_aggregated_mask" );
        debugWrite2d( overlapping_chunk_count, "cpp_overlapping_chunk_count" );
#endif // WRITE_DATA

        post_frames.start = frames.start;
        post_frames.step = frames.step;
        post_frames.duration = frames.duration;
        post_frames.num_samples = num_samples;
        if( !skip_average )
        {
            for( size_t i = 0; i < aggregated_output.size(); ++i )
            {
                for( size_t j = 0; j < aggregated_output[0].size(); ++j )
                {
                    aggregated_output[i][j] /= std::max( overlapping_chunk_count[i][j], epsilon );
                }
            }
        }
        else
        {
            // do nothing
        }

        // average[aggregated_mask == 0.0] = missing
        for( size_t i = 0; i < aggregated_output.size(); ++i )
        {
            for( size_t j = 0; j < aggregated_output[0].size(); ++j )
            {
                if( abs( aggregated_mask[i][j] ) < std::numeric_limits<double>::epsilon() )
                {
                    aggregated_output[i][j] = missing;
                }
            }
        }

        return aggregated_output;

    }

};
// np.max( segmentation[:, cluster == k], axis=1 )
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

class SegmentModel : public OnnxModel 
{
private:
    double m_duration = 5.0;
    double m_step = 0.5;
    int m_batch_size = 1;
    int m_sample_rate = 16000;
    double m_diarization_segmentation_threashold = 0.4442333667381752;
    double m_diarization_segmentation_min_duration_off = 0.5817029604921046;
    size_t m_num_samples = 0;


public:
    SegmentModel(const std::string& model_path)
        : OnnxModel(model_path) {
    }


    // input: batch size x channel x samples count, for example, 32 x 1 x 80000
    // output: batch size x 293 x 3
    std::vector<std::vector<std::vector<float>>> infer( const std::vector<std::vector<float>>& waveform )
    {

        // Create a std::vector<float> with the same size as the tensor
        std::vector<float> audio( waveform.size() * waveform[0].size());
        // std::vector<float> audio( m_batch_size * waveform[0].size());
        for( size_t i = 0; i < waveform.size(); ++i )
        // for( size_t i = 0; i < m_batch_size; ++i )
        {
            for( size_t j = 0; j < waveform[0].size(); ++j )
            {
                audio[i*waveform[0].size() + j] = waveform[i][j];
            }
        }

        // batch_size * num_channels (1 for mono) * num_samples
        const int64_t batch_size = waveform.size();
        // const int64_t batch_size = m_batch_size;
        const int64_t num_channels = 1;
        int64_t input_node_dims[3] = {batch_size, num_channels,
            static_cast<int64_t>(waveform[0].size())};
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(audio.data()), audio.size(),
                input_node_dims, 3);
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.emplace_back(std::move(input_ort));

        auto ort_outputs = session_->Run(
                Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs.data(),
                ort_inputs.size(), output_node_names_.data(), output_node_names_.size());

        const float* outputs = ort_outputs[0].GetTensorData<float>();
        auto outputs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int len1 = outputs_shape[0];
        int len2 = outputs_shape[1];
        int len3 = outputs_shape[2];
        // std::cout<<"output shape:"<<len1<<"x"<<len2<<"x"<<len3<<std::endl;

        len1 = waveform.size();  // <====
        std::vector<std::vector<std::vector<float>>> res( len1, 
                std::vector<std::vector<float>>( len2, std::vector<float>( len3 )));
        for( int i = 0; i < len1; ++i )
        {
            for( int j = 0; j < len2; ++j )
            {
                for( int k = 0; k < len3; ++k )
                {
                    res[i][j][k] = *( outputs + i * len2 * len3 + j * len3 + k );
                }
            }
        }

        return res;
    }
    // 函数接受一个二维数组作为输入，计算每列的平均值，返回一个一维数组
    std::vector<double> 
    calculateColumnAverages(const std::vector<std::vector<double>>& inputArray) {
        // 创建用于存储平均值的数组，初始化为0.0，共三个元素
        std::vector<double> averages(inputArray[0].size(), 0.0);
        // std::cout<<"inputArray.size(): ("<<inputArray.size()<<","<<inputArray[0].size()<<")"<<std::endl;
        // std::cout<<"inputArray: ";
        // 计算每列的和
        for (size_t i = 200; i < inputArray.size(); ++i) {
            for (size_t j = 0; j < inputArray[i].size(); ++j) {
                // std::cout<<","<< inputArray[i][j];
                averages[j] += inputArray[i][j];
            }
            // std::cout<<""<<std::endl;

        }
        // std::cout<<""<<std::endl;

        // 计算每列的平均值
        for (size_t i = 0; i < averages.size(); ++i) {
            averages[i] /= (inputArray.size()-200);
        }

        // 返回存储平均值的数组
        return averages;
    }

    std::vector<std::vector<std::vector<float>>> slide_batch(const std::vector<float>& waveform, 
            SlidingWindow& res_frames )
    {
        int sample_rate = 16000;
        int window_size = std::round(m_duration * sample_rate); // 80000
        int step_size = std::round(m_step * sample_rate); // 8000
        int num_channels = 1;
        size_t num_samples = waveform.size();
        int num_frames_per_chunk = 293; // Need to check with multiple wave files
        size_t i = 0;
        std::vector<std::vector<float>> chunks;
        std::vector<std::vector<std::vector<float>>> outputs;
        while( i + window_size < num_samples )
        {
            // Starting and Ending iterators
            auto start = waveform.begin() + i;
            auto end = start + window_size;

            // To store the sliced vector
            std::vector<float> chunk( window_size, 0.0 );

            // Copy vector using copy function()
            std::copy(start, end, chunk.begin());
            chunks.push_back( chunk ); 
            if( chunks.size() == m_batch_size )
            {
                auto tmp = infer( chunks );
                for( const auto& a : tmp )
                {
                    outputs.push_back( a );
                }
                chunks.clear();
            }

            i += step_size;
        }

        // Process remaining chunks
        if( chunks.size() > 0 )
        {
            auto tmp = infer( chunks );
            for( const auto& a : tmp )
            {
                outputs.push_back( a );
            }
            chunks.clear();
        }

        // Process last chunk if have, last chunk may not equal window_size
        // Make sure at least we have 1 element remaining
        if( i + 1 < num_samples )
        {
            // Starting and Ending iterators
            auto start = waveform.begin() + i;
            auto end = waveform.end();

            // To store the sliced vector, always window_size, for last chunk we pad with 0.0
            std::vector<float> chunk( end - start, 0.0 );

            // Copy vector using copy function()
            std::copy(start, end, chunk.begin());
            chunks.push_back( chunk ); 
            auto tmp = infer( chunks );
            assert( tmp.size() == 1 );

            // Padding
            auto a = tmp[0];
            for( size_t i = a.size(); i < num_frames_per_chunk;  ++i )
            {
                std::vector<float> pad( a[0].size(), 0.0 );
                a.push_back( pad );
            }
            outputs.push_back( a );
        }

        // Calc segments
        res_frames.start = 0.0;
        res_frames.step = m_step;
        res_frames.duration = m_duration;
        res_frames.num_samples = num_samples;
        /*
        float start = 0.0;
        size_t cur_frames = 0;
        while( true )
        {
            std::pair<float, float> seg = { start, start + m_duration };
            segments.push_back( seg );
            if( cur_frames + window_size >= num_samples )
            {
                break;
            }
            start += m_step;
            cur_frames += step_size;
        }
        */

        return outputs;
    }
    // pyannote/audio/core/inference.py:202
    std::vector<std::vector<std::vector<float>>> slide(const std::vector<float>& waveform, 
            SlidingWindow& res_frames )
    {
        int sample_rate = 16000;
        int window_size = std::round(m_duration * sample_rate); // 80000
        int step_size = std::round(m_step * sample_rate); // 8000
        int num_channels = 1;
        size_t num_samples = waveform.size();
        int num_frames_per_chunk = 293; // Need to check with multiple wave files
        size_t i = 0;
        std::vector<std::vector<float>> chunks;
        std::vector<std::vector<std::vector<float>>> outputs;
        static int temp_mex_idx= 0;
        while( i + window_size < num_samples )
        {
 
                // Starting and Ending iterators
            auto start = waveform.begin() + i;
            auto end = start + window_size;

            // To store the sliced vector
            std::vector<float> chunk( window_size, 0.0 );

            // Copy vector using copy function()
            std::copy(start, end, chunk.begin());
            chunks.push_back( chunk ); 
            if( chunks.size() == m_batch_size )
            {    
                // auto start_time = std::chrono::high_resolution_clock::now();

                auto tmp = infer( chunks );

                // auto end_time = std::chrono::high_resolution_clock::now();
                // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                // std::cout << "infer1 函数的运行时间为: " << duration.count() << " 毫秒" << std::endl;
                // std::cout<<"tmp size"<<tmp.size()<<"x"<<tmp[0].size()<<"x"<<tmp[0][0].size()<<std::endl;
                // auto binarized = binarize_swf(tmp,false);
                // std::cout<<"binarized size"<<binarized.size()<<"x"<<binarized[0].size()<<"x"<<binarized[0][0].size()<<std::endl;
                // std::vector<double> result = calculateColumnAverages(binarized[0]);
                // std::cout<<"mean result "<<std::endl<<"speaker 1: "<<result[0]<<std::endl<<" speaker 2: "<<result[1]<<std::endl<<" speaker 3: "<<result[2]<<std::endl;
                // int maxPosition = max_element(result.begin(),result.end()) - result.begin(); 
                // // std::cout<<"maxPosition:"<<maxPosition<<" temp_mex_idx:"<<temp_mex_idx<<" result[maxPosition]:"<<result[maxPosition]<<std::endl;
                // if(maxPosition!=temp_mex_idx && result[maxPosition]>0.2)
                // {
                //     printf("maxPosition %d\n",maxPosition);
                //     temp_mex_idx=maxPosition;
                //     std::cout<<"speaker changeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"<<std::endl;
                //     std::cout<<"time: "<<(float)i/sample_rate<<std::endl;
                // }

                // std::cout<<"maxPosition "<<maxPosition<<std::endl;
                for( const auto& a : tmp )
                {
                    outputs.push_back( a );
                }
                chunks.clear();
            }

            i += step_size;
        }

        // // Process remaining chunks
        if( chunks.size() > 0 )
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            auto tmp = infer( chunks );
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            // std::cout << "infer2 函数的运行时间为: " << duration.count() << " 毫秒" << std::endl;

            for( const auto& a : tmp )
            {
                outputs.push_back( a );
            }
            chunks.clear();
        }

        // Process last chunk if have, last chunk may not equal window_size
        // Make sure at least we have 1 element remaining
        if( i + 1 < num_samples )
        {
            // Starting and Ending iterators
            auto start = waveform.begin() + i;
            auto end = waveform.end();

            // To store the sliced vector, always window_size, for last chunk we pad with 0.0
            std::vector<float> chunk( end - start, 0.0 );

            // Copy vector using copy function()
            std::copy(start, end, chunk.begin());
            chunks.push_back( chunk ); 
            auto start_time = std::chrono::high_resolution_clock::now();

            auto tmp = infer( chunks );
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
 
            assert( tmp.size() == 1 );

            // Padding
            auto a = tmp[0];
            for( size_t i = a.size(); i < num_frames_per_chunk;  ++i )
            {
                std::vector<float> pad( a[0].size(), 0.0 );
                a.push_back( pad );
            }
            outputs.push_back( a );
        }

        // Calc segments
        res_frames.start = 0.0;
        res_frames.step = m_step;
        res_frames.duration = m_duration;
        res_frames.num_samples = num_samples;
        /*
        float start = 0.0;
        size_t cur_frames = 0;
        while( true )
        {
            std::pair<float, float> seg = { start, start + m_duration };
            segments.push_back( seg );
            if( cur_frames + window_size >= num_samples )
            {
                break;
            }
            start += m_step;
            cur_frames += step_size;
        }
        */

        return outputs;
    }

    std::vector<std::vector<std::vector<double>>> binarize_swf(
        const std::vector<std::vector<std::vector<float>>> scores,
        bool initial_state = false ) 
    {
        double onset = m_diarization_segmentation_threashold;

        // TODO: use hlper::rerange_down
        // Imlemenation of einops.rearrange c f k -> (c k) f
        int num_chunks = scores.size();
        int num_frames = scores[0].size();
        int num_classes = scores[0][0].size();
        std::vector<std::vector<double>> data(num_chunks * num_classes, std::vector<double>(num_frames));
        int rowNum = 0;
        for ( const auto& row : scores ) 
        {
            // Create a new matrix with swapped dimensions
            std::vector<std::vector<double>> transposed(num_classes, std::vector<double>(num_frames));

            for (int i = 0; i < num_frames; ++i) {
                for (int j = 0; j < num_classes; ++j) {
                    data[rowNum * num_classes + j][i] = row[i][j];
                }
            }

            rowNum++;
        }
        /*
        for( const auto& d : data )
        {
            for( float e : d )
            {
                std::cout<<e<<",";
            }
            std::cout<<std::endl;
        }
        */

        auto binarized = binarize_ndarray( data, onset, initial_state);

        // TODO: use help::rerange_up
        // Imlemenation of einops.rearrange (c k) f -> c f k - restore
        std::vector<std::vector<std::vector<double>>> restored(num_chunks, 
                std::vector<std::vector<double>>( num_frames, std::vector<double>(num_classes)));
        rowNum = 0;
        for( size_t i = 0; i < binarized.size(); i += num_classes )
        {
            for( size_t j = 0; j < num_classes; ++j )
            {
                for( size_t k = 0; k < num_frames; ++k )
                {
                    restored[rowNum][k][j] = binarized[i+j][k];
                }
            }
            rowNum++;
        }

        return restored;
    }

    std::vector<std::vector<bool>> binarize_ndarray(
        const std::vector<std::vector<double>>& scores,
        double onset = 0.5,
        bool initialState = false
    ) {

        // Scores shape like 2808x293
        size_t rows = scores.size();
        size_t cols = scores[0].size();

        // python: on = scores > onset
        // on is same shape as scores, with true or false inside
        std::vector<std::vector<bool>> on( rows, std::vector<bool>( cols, false ));
        for( size_t i = 0; i < rows; ++i )
        {
            for( size_t j = 0; j < cols; ++j )
            {
                if( scores[i][j] > onset )
                {
                    on[i][j] = true;
                }
     
            }
        }

        // python: off_or_on = (scores < offset) | on
        // off_or_on is same shape as scores, with true or false inside
        // Since onset and offset is same value, it should be true unless score[i][j] == onset
        std::vector<std::vector<bool>> off_or_on( rows, std::vector<bool>( cols, true ));
        for( size_t i = 0; i < rows; ++i )
        {
            for( size_t j = 0; j < cols; ++j )
            {
                if(abs( scores[i][j] - onset ) < std::numeric_limits<double>::epsilon())
                    off_or_on[i][j] = false;
            }
        }

        // python: # indices of frames for which the on/off state is well-defined
        // well_defined_idx = np.array(
        //     list(zip_longest(*[np.nonzero(oon)[0] for oon in off_or_on], fillvalue=-1))
        // ).T
        auto well_defined_idx = Helper::wellDefinedIndex( off_or_on );

        // same_as same shape of as scores
        // python: same_as = np.cumsum(off_or_on, axis=1)
        auto same_as = Helper::cumulativeSum( off_or_on );

        // python: samples = np.tile(np.arange(batch_size), (num_frames, 1)).T
        std::vector<std::vector<int>> samples( rows, std::vector<int>( cols, 0 ));
        for( size_t i = 0; i < rows; ++i )
        {
            for( size_t j = 0; j < cols; ++j )
            {
                samples[i][j] = i;
            }
        }

        // create same shape of initial_state as scores.
        std::vector<std::vector<bool>> initial_state( rows, std::vector<bool>( cols, initialState ));


        // python: return np.where( same_as, on[samples, well_defined_idx[samples, same_as - 1]], initial_state)
        // TODO: delete tmp, directly return
 
        auto tmp = Helper::numpy_where( same_as, on, well_defined_idx, initial_state, samples );
 
        return tmp;
    }

    std::vector<float> crop( const std::vector<float>& waveform, std::pair<double, double> segment) 
    {
        int start_frame = static_cast<int>(std::floor(segment.first * m_sample_rate));
        int frames = static_cast<int>(waveform.size());

        int num_frames = static_cast<int>(std::floor(m_duration * m_sample_rate));
        int end_frame = start_frame + num_frames;

        int pad_start = -std::min(0, start_frame);
        int pad_end = std::max(end_frame, frames) - frames;
        start_frame = std::max(0, start_frame);
        end_frame = std::min(end_frame, frames);
        num_frames = end_frame - start_frame;

        std::vector<float> data(waveform.begin() + start_frame, waveform.begin() + end_frame);

        // Pad with zeros
        data.insert(data.begin(), pad_start, 0.0);
        data.insert(data.end(), pad_end, 0.0);

        return data;
    }

    // pyannote/audio/pipelines/utils/diarization.py:108
    std::vector<int> speaker_count( const std::vector<std::vector<std::vector<float>>>& segmentations,
            const std::vector<std::vector<std::vector<double>>>& binarized,
            const SlidingWindow& pre_frame,
            SlidingWindow& count_frames,
            int num_samples )
    {
 
        // python: trimmed = Inference.trim
        SlidingWindow trimmed_frames;
        SlidingWindow frames( 0.0, m_step, m_duration );
        auto trimmed = trim( binarized, 0.1, 0.1, frames, trimmed_frames );

#ifdef WRITE_DATA
        debugWrite3d( trimmed, "cpp_trimmed" );
#endif // WRITE_DATA

        // python: count = Inference.aggregate(
        // python: np.sum(trimmed, axis=-1, keepdims=True)
        std::vector<std::vector<std::vector<double>>> sum_trimmed( trimmed.size(), 
                std::vector<std::vector<double>>( trimmed[0].size(), std::vector<double>( 1 )));
        for( size_t i = 0; i < trimmed.size(); ++i )
        {
            for( size_t j = 0; j < trimmed[0].size(); ++j )
            {
                double sum = 0.0;
                for( size_t k = 0; k < trimmed[0][0].size(); ++k )
                {
                    sum += trimmed[i][j][k];
                }
                sum_trimmed[i][j][0] = sum;
            }
        }
#ifdef WRITE_DATA
        debugWrite3d( sum_trimmed, "cpp_sum_trimmed" );
#endif // WRITE_DATA
       
        auto count_data = PipelineHelper::aggregate( sum_trimmed, trimmed_frames, 
                pre_frame, count_frames, false, 0.0, false );

#ifdef WRITE_DATA
        debugWrite2d( count_data, "cpp_count_data" );
#endif // WRITE_DATA
       
        // count_data is Nx1, so we convert it to 1d array
        assert( count_data[0].size() == 1 );

        // python: count.data = np.rint(count.data).astype(np.uint8)
        //std::vector<std::vector<int>> res( count_data.size(), std::vector<int>( count_data[0].size()));
        std::vector<int> res( count_data.size());
        for( size_t i = 0; i < res.size(); ++i )
        {
            res[i] = Helper::np_rint( count_data[i][0] );
        }

        return res;
    }

    // pyannote/audio/core/inference.py:540
    // use after_trim_step, after_trim_duration to calc sliding_window later 
    std::vector<std::vector<std::vector<double>>> trim(
            const std::vector<std::vector<std::vector<double>>>& binarized, 
            double left, double right, 
            const SlidingWindow& before_trim, 
            SlidingWindow& trimmed_frames )
    {
        double before_trim_start = before_trim.start;
        double before_trim_step = before_trim.step;
        double before_trim_duration = before_trim.duration;
        size_t chunkSize = binarized.size();
        size_t num_frames = binarized[0].size();

        // python: num_frames_left = round(num_frames * warm_up[0])
        size_t num_frames_left = floor(num_frames * left);

        // python: num_frames_right = round(num_frames * warm_up[1])
        size_t num_frames_right = floor(num_frames * right);
        size_t num_frames_step = floor(num_frames * before_trim_step / before_trim_duration);

        // python: new_data = scores.data[:, num_frames_left : num_frames - num_frames_right]
        std::vector<std::vector<std::vector<double>>> trimed( binarized.size(), 
                std::vector<std::vector<double>>((num_frames - num_frames_right - num_frames_left), 
                std::vector<double>( binarized[0][0].size())));
        for( size_t i = 0; i < binarized.size(); ++i )
        {
            for( size_t j = num_frames_left; j < num_frames - num_frames_right; ++j )
            {
                for( size_t k = 0; k < binarized[0][0].size(); ++k )
                {
                    trimed[i][j - num_frames_left][k] = binarized[i][j][k];
                }
            }
        }

        trimmed_frames.start = before_trim_start + left * before_trim_duration;
        trimmed_frames.step = before_trim_step;
        trimmed_frames.duration = ( 1 - left - right ) * before_trim_duration;
        trimmed_frames.num_samples = num_frames - num_frames_right - num_frames_left;

        return trimed;
    }

}; // SegmentModel


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
#endif