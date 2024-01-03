#include "sherpa_stt.h"

void STTInterface::init_stt(bool using_whisper)
{
    std::string tokens;
    sherpa_onnx::OfflineModelConfig model_config;

    if (using_whisper)
    {
        tokens = "./bin/distil-small.en-tokens.txt";
        std::string encoder_filename = "./bin/distil-small.en-encoder.int8.onnx";
        std::string decoder_filename = "./bin/distil-small.en-decoder.int8.onnx";
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

std::string STTInterface::perform_stt(const std::vector<float> &audioData)
{

    auto s = recognizer->CreateStream();
    s->AcceptWaveform(sampleRate, audioData.data(), audioData.size());

    ss.push_back(std::move(s));
    ss_pointers.push_back(ss.back().get());
    recognizer->DecodeStreams(ss_pointers.data(), 1);

    const std::string text = ss[0]->GetResult().text;

    ss.clear();
    ss_pointers.clear();

    return text;
}
// };
