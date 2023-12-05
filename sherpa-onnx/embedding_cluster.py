import numpy as np
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def loadArrayFromBinaryFile(filename):
    with open(filename, 'rb') as file:
        # 读取二进制文件中的数据
        data = file.read()

    # 将二进制数据转换为NumPy数组
    array = np.frombuffer(data, dtype=np.float32)

    # 根据数组的维度重新构建NumPy数组
    dim1 = 17
    dim2 = 512
    # dim3 = 3
    array = array.reshape(dim1, dim2)

    return array

array_correct = loadArrayFromBinaryFile('speaker_embedding.bin')
print(array_correct.shape)

###################### spectrul clustering #################33
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import pairwise_distances
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# embeddings = scaler.fit_transform(array_correct)

# # 计算嵌入向量之间的相似度矩阵
# similarity_matrix = pairwise_distances(embeddings, metric='euclidean')

# # 使用谱聚类进行聚类
# spectral_clustering = SpectralClustering(n_clusters=None, affinity='precomputed', assign_labels='discretize')
# spectral_clustering.fit(similarity_matrix)

# # 获取每个嵌入向量的聚类标签
# cluster_labels = spectral_clustering.labels_

# # 获取聚类中心
# cluster_centers = []
# for cluster_id in np.unique(cluster_labels):
#     cluster_indices = np.where(cluster_labels == cluster_id)[0]
#     cluster_embeddings = embeddings[cluster_indices]
#     cluster_center = np.mean(cluster_embeddings, axis=0)
#     cluster_centers.append(cluster_center)


###################################################################3


 



array_correc_normed = np.linalg.norm(array_correct, axis=-1, keepdims=True)
array_correc_normed=array_correct/array_correc_normed
print(array_correct.shape)

Z = linkage(array_correc_normed, method='centroid', metric="euclidean")

        # apply the predefined threshold
clusters = fcluster(Z, 0.8153814381597874, criterion="distance") - 1
print(Z.shape)

print(clusters)
# split clusters into two categories based on their number of items:
# large clusters vs. small clusters
# cluster_unique, cluster_counts = np.unique(
#     clusters,
#     return_counts=True,
# )
# large_clusters = cluster_unique[cluster_counts >= 2]
# num_large_clusters = len(large_clusters)
# print("num_large_clusters",num_large_clusters)

# plt.figure()
# dendrogram(Z)
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Sample Index')
# plt.ylabel('Distance')
# plt.show()

# # k = 2

# # # 根据给定的K值将层次聚类结果划分成K个簇
# clusters = fcluster(Z, k, criterion='distance')

# # 输出每个嵌入向量的聚类结果
# for i, embedding in enumerate(array_correct):
#     cluster_label = clusters[i]
#     print(f"Embedding {i} belongs to cluster {cluster_label}")

# 计算每个聚类的中心点
# cluster_centers = []
# for cluster_id in range(1, k+1):
#     cluster_indices = np.where(clusters == cluster_id)[0]
#     cluster_embeddings = [array_correct[idx] for idx in cluster_indices]
#     cluster_center = np.mean(cluster_embeddings, axis=0)
#     cluster_centers.append(cluster_center)

# # 输出每个聚类的中心点
# for i, center in enumerate(cluster_centers):
#     print(f"Cluster {i+1} center: {center}")