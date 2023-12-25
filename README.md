

# VAD-KWS-STT demo

<p align="center">
  <img src="./pipeline_module/img/vibe_bot_audio_ai.jpg" alt="图片描述" width="1280" height="320">
</p>

## 项目描述

该项目由五个子模块组成，分别是 `src/kws`、`src/litevad`、`src/tts` 、`src/speaker_id` 和 `src/sherpa_stt`，分别对应关键词唤醒（KWS）、活跃说话检测（VAD）、 语音到文字转录（STT）、文字到语音转录（TTS）、说话人ID (speaker_id)。所有的音频处理基于 `src/alsa` 模块，它将从 ALSA 读取的音频片段不断写入一个循环缓冲区，并由不同的子模块使用不同的帧长进行读取。

## 子模块

- `src/kws`：关键词唤醒（KWS）模块 （NPU）
- `src/litevad`：活跃说话检测（VAD）模块 （CPU）
- `src/sherpa_stt`：语音到文字转录（STT）模块 （CPU）
- `src/tts`：文字到语音转录（TTS）模块 （CPU）
- `src/speaker_id`：说话人识别模块 （NPU&CPU）
- `src/speaker_diarization`：说话人日志模块 （NPU&CPU）

## 模块数据流
| Task | Input | Output |
|------|-------|--------|
| VAD | 1. Audio (float32, 96ms window length, 0 window shift) | 1. Vad_state (flag, speech start and end state) <br> 2. Speech start and end poison in buffer (samples number) |
| KWS | 1. Audio (float32, 1500ms window length, 500 window shift) <br> 2. kws_params (mfcc, rknn ctx, embedding) | 1. Flag (is_Spotting for input audio) |
| STT | 1. Audio (float32, any length audio ≤ 30s, depend on task) | 1. Text |
| TTS | 1. Text | 1. Raw wav data (Samplerate: 22,050Hz) |
| Speaker ID | 1. Audio (int16, any length ≤ 30s) | 1. Float32 vector (512,1) |
| Speaker Diarization | 1. Full audio (float32, 5s window length, 500ms window shift) | 1. Diarization annotation |


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

## 说话人日志模块 (simple speaker diarization)

该模块位于 `pipeline_module/speaker_diarization`。使用了 VAD（语音活动检测）、说话人识别和层次聚类算法。`pipeline_module/speaker_diary_simple.cc` 是demo。

### 功能概述
- **VAD 和分段**：通过识别超过 200 毫秒的silence来识别活跃的语音片段。每个片段会往前计算一个VAD帧（96 毫秒）采样并进行嵌入。短于 2 秒的片段进行补零，长于 2 秒的进行切片分chunk但不重叠。
- **嵌入**：为每个切片计算嵌入，并对活跃片段的嵌入进行平均处理，每个切片只保留一个embedding。对所有embedding进行normalization，以便在后续阈值控制中保持在 [0~1] 范围内。
- **层次聚类**：使用预定义阈值（0.81来自pyannote）进行聚类。保留具有 >= 2 个中心点的聚类作为主要聚类；与pyannote不同的是这里较小的聚类使用余弦相似度进行二次聚类。相似度 < 0.4 的聚类被识别为独立聚类。(代码中使用1减来计算最小，则为>0.6)

### 性能与评估
**优点:**
- 我们的说话人日志处理流程相比于 pyannote 的流程是一个简化版本。去除了说话人重叠模块，显著减少了计算时间，在pyannote的一些测试片段上取得了一致的结果。

**缺点:**
- 当前的 VAD 设置（静音 > 200ms）在提取较不明显的说话者之间的连贯“单人说话片段”方面效果欠佳。这影响了聚类，特别是在多个说话者之间间隔较短、频繁出现非语言声音（如 'em'，'aha'，'yeah'）的情况下，导致了短且不连续的片段，阻碍了有效的嵌入。同理，如果一个连续片段中出现多个说话者，回影响片段的embedding效果从而导致cluster不准

### 未来优化策略
1. **说话人分段**：优先区分连续片段是否来自同一说话人，检测相邻片段之间的说话人变化。探索比 pyannote 更高效的方法。

2. **片段合并**：考虑在没有说话人变化的情况下合并较短片段以获得更长的嵌入，确保更稳定的特征表达  

3. **长音频分段**： 测试发现,对于一些长音频，尤其是多个speaker，之间发生的音频环境变化或单纯因为embeding过多带来的cluater空间过于分散，未来可以考虑将长音频进行分段，例如2分钟为一个独立的cluster,再提取其中的unique speaker embedding 进行全局的cluster，可以一定程度缓解太长的音频带来的speaker id 重复问题


&nbsp;


# 模型文件

此处的模型及参数需要从hugging face 上克隆到本地并存放在bin文件夹下
https://huggingface.co/vibeus/sherpa-onnx-int8
 
在开始之前，请确保已安装 [git-lfs](https://git-lfs.com)。

## 安装

使用以下命令克隆项目：

```shell
git clone https://huggingface.co/vibeus/sherpa-onnx-int8
```

sherpa-onnx-int8文件夹内的所有二进制文件或模型文件为模型所需，请安装到板子合适位置

某些模型文件较大，请使用git lfs

```shell
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/vibeus/sherpa-onnx-int8
git lfs pull
```

 
 
## Demo

该项目包含的示例流程：

1. `pipeline_module/kws_stt_online.cc`：将关键词唤醒（KWS）和语音到文字转录（STT）绑定的演示程序。
2. `pipeline_module/vad_stt_online.cc`：将活跃说话检测（VAD）和语音到文字转录（STT）绑定的演示程序。
3. `pipeline_module/vad_stt_tts.cc`: 融合了AI VAD， STT，TTS模块。更新：为STT 添加了热词机制
4. `pipeline_module/speaker_track.cc`: 融合了AI VAD，speaker id模块。
4. `pipeline_module/speaker_diary_simple.cc`: 融合了AI VAD，speaker id，speaker diarization模块。
