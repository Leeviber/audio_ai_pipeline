#include "stt_engine.h"

void STTEngine::init_stt(bool using_whisper)
{
    std::string tokens;
    sherpa_onnx::OfflineModelConfig model_config;

    if (using_whisper)
    {
        // tokens = "./bin/distil-small.en-tokens.txt";
        // std::string encoder_filename = "./bin/distil-small.en-encoder.int8.onnx";
        // std::string decoder_filename = "./bin/distil-small.en-decoder.int8.onnx";

        tokens = "./bin/distil-medium.en-tokens.txt";
        std::string encoder_filename = "./bin/distil-medium.en-encoder.int8.onnx";
        std::string decoder_filename = "./bin/distil-medium.en-decoder.int8.onnx";
     
        sherpa_onnx::OfflineWhisperModelConfig whisper;
        whisper.encoder = encoder_filename;
        whisper.decoder = decoder_filename;
        whisper.language = "en";
        whisper.tail_paddings = 800;

        model_config.model_type = "whisper";
        model_config.whisper = whisper;
    }
    else
    {
        tokens = "./bin/encoder-epoch-30-avg-4-tokens.txt";
        std::string encoder_filename = "./bin/encoder-epoch-30-avg-4.int8.onnx";
        std::string decoder_filename = "./bin/decoder-epoch-30-avg-4.int8.onnx";
        std::string joiner_filename = "./bin/joiner-epoch-30-avg-4.int8.onnx";

        sherpa_onnx::OfflineTransducerModelConfig transducer;
        transducer.encoder_filename = encoder_filename;
        transducer.decoder_filename = decoder_filename;
        transducer.joiner_filename = joiner_filename;

        model_config.model_type = "transducer";
        model_config.transducer = transducer;
    }

    model_config.tokens = tokens;

    sherpa_onnx::OfflineRecognizerConfig config;
    config.model_config = model_config;

    if (!config.Validate())
    {
        fprintf(stderr, "Errors in config!\n");
        return;
    }

    fprintf(stdout, "Creating recognizer ...\n");
    recognizer = std::make_unique<sherpa_onnx::OfflineRecognizer>(config);
}

std::string STTEngine::perform_stt(const std::vector<float> &audioData)
{

    auto s = recognizer->CreateStream();
    s->AcceptWaveform(sampleRate, audioData.data(), audioData.size());
    recognizer->DecodeStream(s.get());
    const std::string text= s->GetResult().text;

    ss.clear();
    ss_pointers.clear();

    return text;
}

void VADChunk::InitVAD(const std::string& model_path, const int window_size) {
    sherpa_onnx::VadModelConfig vad_config;
    sherpa_onnx::SileroVadModelConfig silero_vad;
    silero_vad.model = model_path;
    silero_vad.window_size=(window_size / 1000.0f)*vad_config.sample_rate;
    printf("inside ilero_vad window size%d\n",silero_vad.window_size);
    vad_config.silero_vad = silero_vad;
    vad_ = std::make_unique<sherpa_onnx::VoiceActivityDetector>(vad_config);
}

void VADChunk::PushAudioChunk(const std::vector<float>& audio_chunk) {
    vad_->AcceptWaveform(audio_chunk.data(), audio_chunk.size());
}

void VADChunk::ChunkSTT(STTEngine& stt_interface) {
    while (!vad_->Empty()) {
        auto& segment = vad_->Front();
        std::string text = stt_interface.perform_stt(segment.samples);
        fprintf(stderr, "TEXT: %s\n----\n", text.c_str());
        vad_->Pop();
    }
}
// };
