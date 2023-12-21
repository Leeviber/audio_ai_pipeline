import numpy as np
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
 
 

import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
 
import glob


def compute_fbank(wav_path,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank, simlilar to the one in wespeaker.dataset.processor,
        While integrating the wave reading and CMN.
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      sample_frequency=sample_rate,
                      window_type='hamming',
                      use_energy=False)
    # CMN, without CVN
    mat = mat - torch.mean(mat, dim=0)
    return mat

folder_path = "test_audio/audio_output/audio"
file_extension = ".wav"
def sort_by_number(filename):
    # 提取文件名中的数字部分
    number = int(filename.split("/")[-1].split(".")[0].split("audio")[1])
    return number
# 使用 glob 模块获取指定文件夹下的所有 WAV 文件列表
file_list = glob.glob(folder_path + "*" + file_extension)

# 对文件列表按名称进行排序
sorted_file_list = sorted(file_list)
onnx_path="./bin/voxceleb_resnet34_LM.onnx"
# 打印排序后的文件名列表
embedding_array = np.empty((0, 256), dtype=np.float32)
for filename in sorted_file_list:
    print(filename)

    so = ort.SessionOptions()
    so.inter_op_num_threads = 4
    so.intra_op_num_threads = 4
    session = ort.InferenceSession(onnx_path, sess_options=so)
    feats = compute_fbank(filename)
    feats = feats.unsqueeze(0).numpy()  # add batch dimension

    embeddings = session.run(
        output_names=['embs'],
        input_feed={
            'feats': feats
        }
    )
    print(embeddings[0].shape)
    embedding_array = np.append(embedding_array, embeddings[0], axis=0)
print(embedding_array.shape)

array_correc_normed = np.linalg.norm(embedding_array, axis=-1, keepdims=True)
array_correc_normed=embedding_array/array_correc_normed
print(array_correc_normed.shape)

Z = linkage(array_correc_normed, method='centroid', metric="euclidean")

# apply the predefined threshold
clusters = fcluster(Z, 0.8153814381597874, criterion="distance") - 1
print(Z.shape)
print(clusters)
# split clusters into two categories based on their number of items:
# large clusters vs. small clusters
cluster_unique, cluster_counts = np.unique(
    clusters,
    return_counts=True,
)
print("cluster_unique",cluster_unique)
print("cluster_counts",cluster_counts)

large_clusters = cluster_unique[cluster_counts >= 2]
num_large_clusters = len(large_clusters)
print("num_large_clusters",num_large_clusters)

plt.figure()
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
 