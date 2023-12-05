#include "speaker_embedding.h"
 
int embed_model_size=256;

int16_t floatToInt16(float value)
{
    // return static_cast<int16_t>(std::round(value));
    return static_cast<int16_t>(std::round(value * 32767.0f));

}

std::vector<std::vector<double>> getEmbedding( std::shared_ptr<wespeaker::SpeakerEngine> engine, const std::vector<std::vector<float>>& dataChunks, 
        const std::vector<std::vector<float>>& masks )
{

    // Debug
    static int number = 0;


    size_t batch_size = dataChunks.size();
    size_t num_samples = dataChunks[0].size();

    // python: imasks = F.interpolate(... ) and imasks = imasks > 0.5
    auto imasks = Helper::interpolate( masks, num_samples, 0.5 );


    //masks is [32x293] imask is [32x80000], dataChunks is [32x80000] as welll
    
    // python: signals = pad_sequence(...)
    auto signals = Helper::padSequence( dataChunks, imasks );

    // python: wav_lens = imasks.sum(dim=1)
    std::vector<float> wav_lens( batch_size, 0.0 );
    float max_len = 0;
    int index = 0;
    for( const auto& a : imasks )
    {
        float tmp = std::accumulate(a.begin(), a.end(), 0.0);
        wav_lens[index++] = tmp;
        if( tmp > max_len )
            max_len = tmp;
    }

    // python: if max_len < self.min_num_samples: return np.NAN * np.zeros(...
    if( max_len < 640 )
    {
        // TODO: don't call embedding process, direct return
        // batch_size x 192, where 192 is size of length embedding result for each waveform
        // python: return np.NAN * np.zeros((batch_size, self.dimension))
        std::vector<std::vector<double>> embeddings(batch_size, std::vector<double>( embed_model_size, NAN ));
        return embeddings;
    }


    // python:         
    //      too_short = wav_lens < self.min_num_samples
    //      wav_lens = wav_lens / max_len
    //      wav_lens[too_short] = 1.0 
    std::vector<bool> too_short( wav_lens.size(), false );
    for( size_t i = 0; i < wav_lens.size(); ++i )
    {
        if( wav_lens[i] < 640 )
        {
            wav_lens[i] = 1.0;
            too_short[i] = true;
        }
        else
        {
            wav_lens[i] /= max_len;
        }
    }



    // signals is [32x80000], wav_lens is of length 32 of 1d array, an example for wav_lens
    // [1.0000, 1.0000, 1.0000, 0.0512, 1.0000, 1.0000, 0.1502, ...] 
    // Now call embedding model to get embeddings of batches
    // speechbrain/pretrained/interfaces.py:903
    std::vector<std::vector<int16_t>> signals_int16(signals.size(), std::vector<int16_t>(num_samples));

    std::vector<std::vector<float>> embeddings_f( signals.size(),
        std::vector<float>(embed_model_size, 0.0 ));

    for(int i=0;i<batch_size;i++)
    {
        std::transform(signals[i].begin(), signals[i].end(), signals_int16[i].begin(), floatToInt16);

        std::vector<float> single_emb(embed_model_size, 0.0);
        engine->ExtractEmbedding(signals_int16[i].data(),
                                    num_samples,
                                    &single_emb);
        embeddings_f[i] = single_emb;

    }

    // auto embeddings_f = em.infer( signals, wav_lens );
 
    // Convert float to double 
    size_t col = embeddings_f[0].size();
    std::vector<std::vector<double>> embeddings(embeddings_f.size(), 
            std::vector<double>( col ));

    // python: embeddings[too_short.cpu().numpy()] = np.NAN
    for( size_t i = 0; i < too_short.size(); ++i )
    {
        if( too_short[i] )
        {
            for( size_t j = 0; j < col; ++j )
            {
                embeddings[i][j] = NAN;
            }
        }
        else
        {
            for( size_t j = 0; j < col; ++j )
            {
                embeddings[i][j] = static_cast<double>( embeddings_f[i][j] );
            }
        }
    }
    // std::cout<<"embeddings "<<embeddings.size()<<embeddings[0].size()<<std::endl;

    return embeddings;
}
