
# VAD-KWS-STT demo


## 项目描述

该项目由三个子模块组成，分别是 `src/kws`、`src/litevad` 和 `src/sherpa_stt`，分别对应关键词唤醒（KWS）、活跃说话检测（VAD）和语音到文字转录（STT）。所有的音频处理基于 `src/alsa` 模块，它将从 ALSA 读取的音频片段不断写入一个循环缓冲区，并由不同的子模块使用不同的帧长进行读取。

## 子模块

- `src/kws`：关键词唤醒（KWS）模块
- `src/litevad`：活跃说话检测（VAD）模块
- `src/sherpa_stt`：语音到文字转录（STT）模块

## 依赖

- `src/kws` 模块依赖于 `src/librknn_api`，并需要读取 `src/bin` 中的模型权重和文件。
- `src/sherpa_stt` 模块依赖于 `cmake` 中的 `json`、`kaldi-native-fbank` 和 `onnxruntime`，并需要读取 `src/bin/stt_weight` 的模型权重。

## Demo

该项目包含两个完整的示例流程：

1. `kws_stt_online.cc`：将关键词唤醒（KWS）和语音到文字转录（STT）绑定的演示程序。
2. `vad_stt_online.cc`：将活跃说话检测（VAD）和语音到文字转录（STT）绑定的演示程序。

 