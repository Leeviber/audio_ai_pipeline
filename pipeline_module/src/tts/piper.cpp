#ifndef PIPER_H_

#include <array>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <espeak-ng/speak_lib.h>
#include "onnxruntime_cxx_api.h"

#include "json.hpp"
#include "piper.h"
#include "utf8.h"

namespace piper
{

#ifdef _PIPER_VERSION
// https://stackoverflow.com/questions/47346133/how-to-use-a-define-inside-a-format-string
#define _STR(x) #x
#define STR(x) _STR(x)
  const std::string VERSION = STR(_PIPER_VERSION);
#else
  const std::string VERSION = "";
#endif

  // Maximum value for 16-bit signed WAV sample
  const float MAX_WAV_VALUE = 32767.0f;

  const std::string instanceName{"piper"};

  std::string getVersion()
  {
    return VERSION;
  }

  // True if the string is a single UTF-8 codepoint
  bool isSingleCodepoint(std::string s)
  {
    return utf8::distance(s.begin(), s.end()) == 1;
  }

  // Get the first UTF-8 codepoint of a string
  Phoneme getCodepoint(std::string s)
  {
    utf8::iterator character_iter(s.begin(), s.begin(), s.end());
    return *character_iter;
  }

  // Load JSON config information for phonemization
  void parsePhonemizeConfig(json &configRoot, PhonemizeConfig &phonemizeConfig)
  {

    if (configRoot.contains("espeak"))
    {
      auto espeakValue = configRoot["espeak"];
      if (espeakValue.contains("voice"))
      {
        phonemizeConfig.eSpeak.voice = espeakValue["voice"].get<std::string>();
      }
    }

    if (configRoot.contains("phoneme_type"))
    {
      auto phonemeTypeStr = configRoot["phoneme_type"].get<std::string>();
      if (phonemeTypeStr == "text")
      {
        phonemizeConfig.phonemeType = TextPhonemes;
      }
    }

    // phoneme to [id] map
    // Maps phonemes to one or more phoneme ids (required).
    if (configRoot.contains("phoneme_id_map"))
    {
      auto phonemeIdMapValue = configRoot["phoneme_id_map"];
      for (auto &fromPhonemeItem : phonemeIdMapValue.items())
      {
        std::string fromPhoneme = fromPhonemeItem.key();
        if (!isSingleCodepoint(fromPhoneme))
        {
          std::stringstream idsStr;
          for (auto &toIdValue : fromPhonemeItem.value())
          {
            PhonemeId toId = toIdValue.get<PhonemeId>();
            idsStr << toId << ",";
          }
        }

        auto fromCodepoint = getCodepoint(fromPhoneme);
        for (auto &toIdValue : fromPhonemeItem.value())
        {
          PhonemeId toId = toIdValue.get<PhonemeId>();
          phonemizeConfig.phonemeIdMap[fromCodepoint].push_back(toId);
        }
      }
    }

    // phoneme to [phoneme] map
    // Maps phonemes to one or more other phonemes (not normally used).
    if (configRoot.contains("phoneme_map"))
    {
      if (!phonemizeConfig.phonemeMap)
      {
        phonemizeConfig.phonemeMap.emplace();
      }

      auto phonemeMapValue = configRoot["phoneme_map"];
      for (auto &fromPhonemeItem : phonemeMapValue.items())
      {
        std::string fromPhoneme = fromPhonemeItem.key();
        if (!isSingleCodepoint(fromPhoneme))
        {
          // spdlog::error("\"{}\" is not a single codepoint", fromPhoneme);
          // throw std::runtime_error(
          //     "Phonemes must be one codepoint (phoneme map)");
        }

        auto fromCodepoint = getCodepoint(fromPhoneme);
        for (auto &toPhonemeValue : fromPhonemeItem.value())
        {
          std::string toPhoneme = toPhonemeValue.get<std::string>();
          if (!isSingleCodepoint(toPhoneme))
          {
            // throw std::runtime_error(
            //     "Phonemes must be one codepoint (phoneme map)");
          }

          auto toCodepoint = getCodepoint(toPhoneme);
          (*phonemizeConfig.phonemeMap)[fromCodepoint].push_back(toCodepoint);
        }
      }
    }

  } /* parsePhonemizeConfig */

  void parseModelConfig(json &configRoot, ModelConfig &modelConfig)
  {

    modelConfig.numSpeakers = 1;
    if (configRoot.contains("speaker_id_map"))
    {
      if (!modelConfig.speakerIdMap)
      {
        modelConfig.speakerIdMap.emplace();
      }

      auto speakerIdMapValue = configRoot["speaker_id_map"];
      for (auto &speakerItem : speakerIdMapValue.items())
      {
        std::string speakerName = speakerItem.key();
        (*modelConfig.speakerIdMap)[speakerName] =
            speakerItem.value().get<SpeakerId>();
      }
    }

  } /* parseModelConfig */

  void initialize(PiperConfig &config)
  {
    if (config.useESpeak)
    {
      // Set up espeak-ng for calling espeak_TextToPhonemesWithTerminator
      // See: https://github.com/rhasspy/espeak-ng
      // spdlog::debug("Initializing eSpeak");
      int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS,
                                     /*buflength*/ 0,
                                     /*path*/ config.eSpeakDataPath.c_str(),
                                     /*options*/ 0);
      if (result < 0)
      {
        printf("Failed to initialize eSpeak-ng \n");
      }
    }
    else
    {
      printf("Failed to initialize speak module \n");
    }
  }

  void terminate(PiperConfig &config)
  {
    if (config.useESpeak)
    {
      espeak_Terminate();
    }
  }

  void loadModel(std::string modelPath, ModelSession &session)
  {
    session.env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                           instanceName.c_str());
    session.env.DisableTelemetryEvents();

    session.options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);

    session.options.DisableCpuMemArena();
    session.options.DisableMemPattern();
    session.options.DisableProfiling();
    // session.options.SetIntraOpNumThreads(2);
    auto startTime = std::chrono::steady_clock::now();
    session.onnx = Ort::Session(session.env, modelPath.c_str(), session.options);
    auto endTime = std::chrono::steady_clock::now();
  }

  // Load Onnx model and JSON config file
  void loadVoice(PiperConfig &config, std::string modelPath,
                 std::string modelConfigPath, Voice &voice)
  {
    // spdlog::debug("Parsing voice config at {}", modelConfigPath);
    std::ifstream modelConfigFile(modelConfigPath);
    voice.configRoot = json::parse(modelConfigFile);

    parsePhonemizeConfig(voice.configRoot, voice.phonemizeConfig);
    parseModelConfig(voice.configRoot, voice.modelConfig);
    voice.synthesisConfig.speakerId = 0;

    loadModel(modelPath, voice.session);

  } /* loadVoice */

  // Phoneme ids to WAV audio
  void synthesize(std::vector<PhonemeId> &phonemeIds,
                  SynthesisConfig &synthesisConfig, ModelSession &session,
                  std::vector<int16_t> &audioBuffer, SynthesisResult &result)
  {
    // spdlog::debug("Synthesizing audio for {} phoneme id(s)", phonemeIds.size());

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Allocate
    std::vector<int64_t> phonemeIdLengths{(int64_t)phonemeIds.size()};
    std::vector<float> scales{synthesisConfig.noiseScale,
                              synthesisConfig.lengthScale,
                              synthesisConfig.noiseW};

    std::vector<Ort::Value> inputTensors;
    std::vector<int64_t> phonemeIdsShape{1, (int64_t)phonemeIds.size()};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, phonemeIds.data(), phonemeIds.size(), phonemeIdsShape.data(),
        phonemeIdsShape.size()));

    std::vector<int64_t> phomemeIdLengthsShape{(int64_t)phonemeIdLengths.size()};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, phonemeIdLengths.data(), phonemeIdLengths.size(),
        phomemeIdLengthsShape.data(), phomemeIdLengthsShape.size()));

    std::vector<int64_t> scalesShape{(int64_t)scales.size()};
    inputTensors.push_back(
        Ort::Value::CreateTensor<float>(memoryInfo, scales.data(), scales.size(),
                                        scalesShape.data(), scalesShape.size()));

    // From export_onnx.py
    std::array<const char *, 4> inputNames = {"input", "input_lengths", "scales",
                                              "sid"};
    std::array<const char *, 1> outputNames = {"output"};

    // Infer
    auto startTime = std::chrono::steady_clock::now();
    auto outputTensors = session.onnx.Run(
        Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
        inputTensors.size(), outputNames.data(), outputNames.size());
    auto endTime = std::chrono::steady_clock::now();

    if ((outputTensors.size() != 1) || (!outputTensors.front().IsTensor()))
    {
      // throw std::runtime_error("Invalid output tensors");
    }
    auto inferDuration = std::chrono::duration<double>(endTime - startTime);
    result.inferSeconds = inferDuration.count();

    const float *audio = outputTensors.front().GetTensorData<float>();
    auto audioShape =
        outputTensors.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t audioCount = audioShape[audioShape.size() - 1];

    result.audioSeconds = (double)audioCount / (double)synthesisConfig.sampleRate;
    result.realTimeFactor = 0.0;
    if (result.audioSeconds > 0)
    {
      result.realTimeFactor = result.inferSeconds / result.audioSeconds;
    }
    // spdlog::debug("Synthesized {} second(s) of audio in {} second(s)",
    //               result.audioSeconds, result.inferSeconds);

    // Get max audio value for scaling
    float maxAudioValue = 0.01f;
    for (int64_t i = 0; i < audioCount; i++)
    {
      float audioValue = abs(audio[i]);
      if (audioValue > maxAudioValue)
      {
        maxAudioValue = audioValue;
      }
    }

    // We know the size up front
    audioBuffer.reserve(audioCount);

    // Scale audio to fill range and convert to int16
    float audioScale = (MAX_WAV_VALUE / std::max(0.01f, maxAudioValue));
    for (int64_t i = 0; i < audioCount; i++)
    {
      int16_t intAudioValue = static_cast<int16_t>(
          std::clamp(audio[i] * audioScale,
                     static_cast<float>(std::numeric_limits<int16_t>::min()),
                     static_cast<float>(std::numeric_limits<int16_t>::max())));

      audioBuffer.push_back(intAudioValue);
    }

    // Clean up
    for (std::size_t i = 0; i < outputTensors.size(); i++)
    {
      Ort::detail::OrtRelease(outputTensors[i].release());
    }

    for (std::size_t i = 0; i < inputTensors.size(); i++)
    {
      Ort::detail::OrtRelease(inputTensors[i].release());
    }
  }

  // ----------------------------------------------------------------------------

  // Phonemize text and synthesize audio
  void textToAudio(PiperConfig &config, Voice &voice, std::string text,
                   std::vector<int16_t> &audioBuffer)
  {

    std::size_t sentenceSilenceSamples = 0;
    if (voice.synthesisConfig.sentenceSilenceSeconds > 0)
    {
      sentenceSilenceSamples = (std::size_t)(
          voice.synthesisConfig.sentenceSilenceSeconds *
          voice.synthesisConfig.sampleRate * voice.synthesisConfig.channels);
    }

    // Phonemes for each sentence
    std::vector<std::vector<Phoneme>> phonemes;

    if (voice.phonemizeConfig.phonemeType == eSpeakPhonemes)
    {
      // Use espeak-ng for phonemization
      eSpeakPhonemeConfig eSpeakConfig;
      eSpeakConfig.voice = voice.phonemizeConfig.eSpeak.voice;
      phonemize_eSpeak(text, eSpeakConfig, phonemes);
    }
    else
    {
      // Use UTF-8 codepoints as "phonemes"
      CodepointsPhonemeConfig codepointsConfig;
      phonemize_codepoints(text, codepointsConfig, phonemes);
    }

    // Synthesize each sentence independently.
    std::vector<PhonemeId> phonemeIds;
    std::map<Phoneme, std::size_t> missingPhonemes;
    for (auto phonemesIter = phonemes.begin(); phonemesIter != phonemes.end();
         ++phonemesIter)
    {
      std::vector<Phoneme> &sentencePhonemes = *phonemesIter;

      SynthesisResult sentenceResult;

      // Use phoneme/id map from config
      PhonemeIdConfig idConfig;
      idConfig.phonemeIdMap =
          std::make_shared<PhonemeIdMap>(voice.phonemizeConfig.phonemeIdMap);

      // phonemes -> ids
      phonemes_to_ids(sentencePhonemes, idConfig, phonemeIds, missingPhonemes);

      // ids -> audio
      synthesize(phonemeIds, voice.synthesisConfig, voice.session, audioBuffer,
                 sentenceResult);

      // Add end of sentence silence
      if (sentenceSilenceSamples > 0)
      {
        for (std::size_t i = 0; i < sentenceSilenceSamples; i++)
        {
          audioBuffer.push_back(0);
        }
      }

      phonemeIds.clear();
    }

    if (missingPhonemes.size() > 0)
    {

      for (auto phonemeCount : missingPhonemes)
      {
        std::string phonemeStr;
        utf8::append(phonemeCount.first, phonemeStr);
      }
    }

  } /* textToAudio */

} // namespace piper

#endif