
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


## 热词模块
热词模块实现已集成到模型decode阶段，`vad_stt_tts.cc` 中已添加相应示例，只需要在config里指定热词文件(`bin/hotwords_token.txt`) 及热词阈值就可以打开热词匹配机制。模型的解码已经默认修改成beam search。

`bin/hotwords_token.txt` 文件是一个热词示例，其中每行是由原始热词tokenize化得到的词编码表示，方便模型的直接读取，后续会更新热词tokenizer工具，方便快速从text->token进行解码实现热词 热更新。

## Speaker Track
说话人追踪模块为`src/speaker_id`,该模块基于speaker embedding模型 `./bin/voxceleb_CAM++_LM.onnx`, 获取每秒钟的speaker embedding并进行历史匹配，该pipeline会输出三个tag 到指定txt,分别为 {匹配的ID，是否第一次出现，过去说话概率}。


输出的tag例子：  

{2/0/0.68}. 意味着这一秒的speaker id 为2,是否第一次出现为否，过去60秒的说话概率为0.68。 

{3/1/0.21}. 意味着这一秒的speaker id 为3,是否第一次出现为是，过去60秒的说话概率为0.21。 

{0/0/0}. ID 0 代表给当前这一秒没有人说话，是基于VAD判断的。

## Demo

该项目包含两个完整的示例流程：

1. `kws_stt_online.cc`：将关键词唤醒（KWS）和语音到文字转录（STT）绑定的演示程序。
2. `vad_stt_online.cc`：将活跃说话检测（VAD）和语音到文字转录（STT）绑定的演示程序。
3. `vad_stt_tts.cc`: 融合了AI VAD， STT，TTS模块。更新：为STT 添加了热词机制
4. `speaker_track.cc`: 融合了AI VAD，speaker id模块。
