
#ifndef SPEAKER_CLUSTER
#define SPEAKER_CLUSTER
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>

#include "speaker_help.h"
#include "clustering.h"

class Cluster
{
private:
    // Those 2 values extracted from config.yaml under
    // ~/.cache/torch/pyannote/models--pyannote--speaker-diarization/snapshots/xxx/
    float m_threshold = 0.8853814381597874;
    size_t m_min_cluster_size = 15;
    float distance_threshold = 0.7; // larger is relax

public:
    Cluster()
    {
    }

    /*
     * pyannote/audio/pipelines/clustering.py:215, __call__(...
     * embeddings: num of chunks x 3 x 192, where 192 is size each embedding
     * segmentations: num of chunks x 293 x 3, where 293 is size each segment model out[0]
     * and 3 is each segment model output[1]
     * */
    void clustering(const std::vector<std::vector<std::vector<double>>> &embeddings,
                    const std::vector<std::vector<std::vector<double>>> &segmentations,
                    std::vector<std::vector<int>> &hard_clusters,
                    int num_clusters = -1, int min_clusters = -1, int max_clusters = -1)
    {
        // python: train_embeddings, train_chunk_idx, train_speaker_idx = self.filter_embeddings
        std::vector<int> chunk_idx;
        std::vector<int> speaker_idx;
        auto filteredEmbeddings = filter_embeddings(embeddings, chunk_idx, speaker_idx);

        size_t num_embeddings = filteredEmbeddings.size();
        set_num_clusters(static_cast<int>(num_embeddings), num_clusters, min_clusters, max_clusters);

        // do NOT apply clustering when min_clusters = max_clusters = 1
        if (max_clusters < 2)
        {
            size_t num_chunks = embeddings.size();
            size_t num_speakers = embeddings[0].size();
            std::vector<std::vector<int>> hcluster(num_chunks, std::vector<int>(num_speakers, 0));
            hard_clusters.swap(hcluster);
            return;
        }

        // python: train_clusters = self.cluster(
        auto clusterRes = cluster(filteredEmbeddings, min_clusters, max_clusters, num_clusters);

        // python: hard_clusters, soft_clusters = self.assign_embeddings(
        assign_embeddings(embeddings, chunk_idx, speaker_idx, clusterRes, hard_clusters);
    }

    void custom_clustering(const std::vector<std::vector<double>> &embeddings,
                           std::vector<int> &clusterRes,
                           int num_clusters = -1, int min_clusters = -1, int max_clusters = -1)
    {
        // python: train_embeddings, train_chunk_idx, train_speaker_idx = self.filter_embeddings
        std::vector<int> chunk_idx;
        std::vector<int> speaker_idx;
        // auto filteredEmbeddings = filter_embeddings( embeddings, chunk_idx, speaker_idx );

        size_t num_embeddings = embeddings.size();
        set_num_clusters(static_cast<int>(num_embeddings), num_clusters, min_clusters, max_clusters);
        // do NOT apply clustering when min_clusters = max_clusters = 1
        if (max_clusters < 2)
        {
            // size_t num_chunks = embeddings.size();
            // size_t num_speakers = embeddings[0].size();
            // std::vector<std::vector<int>> hcluster( num_chunks, std::vector<int>( num_speakers, 0 ));
            // hard_clusters.swap( hcluster );
            return;
        }
        // printf("min_clusters%d, max_clusters%d,num_clusters%d \n",min_clusters,max_clusters,num_clusters);
        clusterRes = cluster(embeddings, min_clusters, max_clusters, num_clusters);
    }

    // Assign embeddings to the closest centroid
    template <typename T>
    void assign_embeddings(const std::vector<std::vector<std::vector<T>>> &embeddings,
                           const std::vector<int> &chunk_idx,
                           const std::vector<int> &speaker_idx,
                           const std::vector<int> &clusterRes,
                           std::vector<std::vector<int>> &hard_clusters)
    {
        assert(chunk_idx.size() == speaker_idx.size());

        // python: num_clusters = np.max(train_clusters) + 1
        int num_clusters = *std::max_element(clusterRes.begin(), clusterRes.end()) + 1;
        size_t num_chunks = embeddings.size();
        size_t num_speakers = embeddings[0].size();
        size_t dimension = embeddings[0][0].size();

        // python: train_embeddings = embeddings[train_chunk_idx, train_speaker_idx]
        std::vector<std::vector<T>> filtered_embeddings(chunk_idx.size(),
                                                        std::vector<T>(dimension, 0.0));
        for (size_t i = 0; i < chunk_idx.size(); ++i)
        {
            auto tmp = embeddings[chunk_idx[i]][speaker_idx[i]];
            for (size_t j = 0; j < dimension; ++j)
            {
                filtered_embeddings[i][j] = tmp[j];
            }
        }

        // python: centroids = np.vstack([np.mean(train_embeddings[train_clusters == k], axis=0)
        std::vector<std::vector<T>> centroids(num_clusters, std::vector<T>(dimension, 0.0));
        assert(filtered_embeddings.size() == clusterRes.size());
        for (int i = 0; i < num_clusters; ++i)
        {
            size_t mean_count = 0;
            for (size_t j = 0; j < clusterRes.size(); ++j)
            {
                if (i == clusterRes[j])
                {
                    mean_count++;
                    for (size_t k = 0; k < dimension; ++k)
                    {
                        centroids[i][k] += filtered_embeddings[j][k];
                    }
                }
            }
            for (size_t k = 0; k < dimension; ++k)
            {
                centroids[i][k] /= mean_count;
            }
        }
        /*
        for( int i = 0; i < num_clusters; ++i )
        {
            for( size_t k = 0; k < dimension; ++k )
            {
                centroids[i][k] /= dimension;
            }
        }
        */

        // for k in range(num_clusters) compute distance between embeddings and clusters
        //  python: rearrange(embeddings, "c s d -> (c s) d"), where d =192
        auto r1 = Helper::rearrange_down(embeddings);

        // python: cdist(
        auto dist = Helper::cosineSimilarity(r1, centroids);

        // python: e2k_distance = rearrange(
        // N x 3 x 4 for example
        // (c s) k -> c s k
        auto soft_clusters = Helper::rearrange_up(dist, num_chunks);

        // python: soft_clusters = 2 - e2k_distance
        for (size_t i = 0; i < soft_clusters.size(); ++i)
        {
            for (size_t j = 0; j < soft_clusters[0].size(); ++j)
            {
                for (size_t k = 0; k < soft_clusters[0][0].size(); ++k)
                {
                    soft_clusters[i][j][k] = 2.0 - soft_clusters[i][j][k];
                }
            }
        }

        // python: hard_clusters = np.argmax(soft_clusters, axis=2)
        //  N x 3
        hard_clusters = Helper::argmax(soft_clusters);
    }

    std::vector<std::vector<double>> filter_embeddings(
        const std::vector<std::vector<std::vector<double>>> &embeddings,
        std::vector<int> &chunk_idx, std::vector<int> &speaker_idx)
    {
        // **************** max_num_embeddings IS INF in python
        // Initialize vectors to store indices of non-NaN elements

        // Find non-NaN elements and store their indices
        for (int i = 0; i < embeddings.size(); ++i)
        {
            for (int j = 0; j < embeddings[i].size(); ++j)
            {
                if (!std::isnan(embeddings[i][j][0]))
                { // Assuming all elements in the innermost array are NaN or not NaN
                    chunk_idx.push_back(i);
                    speaker_idx.push_back(j);
                }
            }
        }

        // Sample max_num_embeddings embeddings if the number of available embeddings is greater
        /*int num_embeddings = chunk_idx.size();
        if (num_embeddings > max_num_embeddings) {
            // Shuffle the indices
            std::vector<int> indices(num_embeddings);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_shuffle(indices.begin(), indices.end());

            // Sort and select the first max_num_embeddings indices
            std::sort(indices.begin(), indices.begin() + max_num_embeddings);

            // Update chunk_idx and speaker_idx with the selected indices
            chunk_idx.clear();
            speaker_idx.clear();
            for (int i = 0; i < max_num_embeddings; ++i) {
                chunk_idx.push_back(chunk_idx[indices[i]]);
                speaker_idx.push_back(speaker_idx[indices[i]]);
            }
        }*/

        // Create a vector to store the selected embeddings
        std::vector<std::vector<double>> selectedEmbeddings;
        for (int i = 0; i < chunk_idx.size(); ++i)
        {
            selectedEmbeddings.push_back(embeddings[chunk_idx[i]][speaker_idx[i]]);
        }

        return selectedEmbeddings;
    }

    void set_num_clusters(int num_embeddings, int &num_clusters, int &min_clusters, int &max_clusters)
    {
        if (num_clusters != -1)
        {
            min_clusters = num_clusters;
        }
        else
        {
            if (min_clusters == -1)
            {
                min_clusters = 1;
            }
        }
        min_clusters = std::max(1, std::min(num_embeddings, min_clusters));

        if (num_clusters != -1)
        {
            max_clusters == num_clusters;
        }
        else
        {
            if (max_clusters == -1)
            {
                max_clusters = num_embeddings;
            }
        }
        max_clusters = std::max(1, std::min(num_embeddings, max_clusters));
        if (min_clusters > max_clusters)
        {
            min_clusters = max_clusters;
        }
        if (min_clusters == max_clusters)
        {
            num_clusters = min_clusters;
        }
    }

    // pyannote/audio/pipelines/clustering.py:426, cluster(...
    // AgglomerativeClustering
    std::vector<int> cluster(const std::vector<std::vector<double>> &embeddings,
                             int min_clusters, int max_clusters, int num_clusters)
    {
        // python: num_embeddings, _ = embeddings.shape
        size_t num_embeddings = embeddings.size();

        // heuristic to reduce self.min_cluster_size when num_embeddings is very small
        // (0.1 value is kind of arbitrary, though)
        // m_min_cluster_size = std::min( m_min_cluster_size, std::max(static_cast<size_t>( 1 ),
        //             static_cast<size_t>( round(0.1 * num_embeddings))));
        m_min_cluster_size = 1;
        // printf("m_min_cluster_size%d",m_min_cluster_size);
        // linkage function will complain when there is just one embedding to cluster
        // if( num_embeddings == 1 )
        //     return np.zeros((1,), dtype=np.uint8)

        // self.metric == "cosine" and self.method == "centroid"
        // python:
        //    with np.errstate(divide="ignore", invalid="ignore"):
        //        embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
        auto normalizedEmbeddings(embeddings);
        Helper::normalizeEmbeddings(normalizedEmbeddings);

        // python: clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1
        auto clusters = Clustering::cluster(normalizedEmbeddings, m_threshold);
        for (size_t i = 0; i < clusters.size(); ++i)
        {
            clusters[i] -= 1;
        }

        // split clusters into two categories based on their number of items:
        // large clusters vs. small clusters
        // python: cluster_unique, cluster_counts = np.unique(...
        std::unordered_map<int, int> clusterCountMap;
        for (int cluster : clusters)
        {
            clusterCountMap[cluster]++;
        }

        // python: large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        // python: small_clusters = cluster_unique[cluster_counts < min_cluster_size]
        std::vector<int> large_clusters;
        std::vector<int> small_clusters;
        for (const auto &entry : clusterCountMap)
        {
            if (entry.second >= m_min_cluster_size)
            {
                large_clusters.push_back(entry.first);
            }
            else
            {
                small_clusters.push_back(entry.first);
            }
        }
        // printf("large_clusters.size()%d\n",large_clusters.size());
        // printf("small_clusters.size()%d\n",small_clusters.size());

        size_t num_large_clusters = large_clusters.size();

        // force num_clusters to min_clusters in case the actual number is too small
        if (num_large_clusters < min_clusters)
            num_clusters = min_clusters;

        // force num_clusters to max_clusters in case the actual number is too large
        if (num_large_clusters > max_clusters)
            num_clusters = max_clusters;

        if (num_clusters != -1)
            assert(false); // this branch is not implemented

        if (num_large_clusters == 0)
        {
            clusters.assign(clusters.size(), 0);
            return clusters;
        }

        if (small_clusters.size() == 0)
        {
            return clusters;
        }

        std::sort(large_clusters.begin(), large_clusters.end());
        std::sort(small_clusters.begin(), small_clusters.end());

        // re-assign each small cluster to the most similar large cluster based on their respective centroids
        auto large_centroids = Helper::calculateClusterMeans(normalizedEmbeddings, clusters, large_clusters);
        // printf("large_centroids.size()%d\n",large_centroids.size());

        auto small_centroids = Helper::calculateClusterMeans(normalizedEmbeddings, clusters, small_clusters);

        // printf("small_centroids.size()%d\n",small_centroids.size());

        // python: centroids_cdist = cdist(large_centroids, small_centroids, metric=self.metric)
        auto centroids_cdist = Helper::cosineSimilarity(large_centroids, small_centroids);

        // Update clusters based on minimum distances
        // python: for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0))
        for (int small_k = 0; small_k < centroids_cdist[0].size(); ++small_k)
        {
            float minVal = std::numeric_limits<float>::max();
            int large_k = -1;

            // np.argmin
            for (size_t i = 0; i < centroids_cdist.size(); ++i)
            {
                // if (centroids_cdist[i][small_k] < minVal) {
                if (centroids_cdist[i][small_k] < minVal && centroids_cdist[i][small_k] < distance_threshold)
                {

                    minVal = centroids_cdist[i][small_k];
                    large_k = i;
                }
            }
            for (size_t i = 0; i < clusters.size(); ++i)
            {
                if (clusters[i] == small_clusters[small_k])
                {
                    if (large_k != -1)
                    {
                        clusters[i] = large_clusters[large_k];
                    }
                }
            }
        }

        // Find unique clusters and return inverse mapping
        std::vector<int> uniqueClusters;
        std::vector<int> inverseMapping = Helper::findUniqueClusters(clusters, uniqueClusters);

        return inverseMapping;
    }
    std::vector<int> mergeAndRenumber(const std::vector<int> &numbers)
    {
        std::unordered_map<int, int> index_map;
        std::vector<int> unique_numbers;

        // 获取唯一的数字并保留其首次出现的顺序
        for (int num : numbers)
        {
            if (index_map.find(num) == index_map.end())
            {
                index_map[num] = static_cast<int>(unique_numbers.size());
                unique_numbers.push_back(num);
            }
        }

        // 根据唯一的数字及其首次出现的顺序进行重新编号
        std::vector<int> result;
        for (int num : numbers)
        {
            result.push_back(index_map[num]);
        }

        return result;
    }
    void generateTimestamps(const std::vector<int> &renumbered_numbers, const std::vector<std::string> &texts)
    {
        printf("##### Speaker Diarization #####\n");

        for (int i = 0; i < renumbered_numbers.size(); i++)
        {
            printf("Speaker ID: %d, text %s \n", renumbered_numbers[i], texts[i].c_str());
        }
    }
    void generateLabel(const std::vector<int> &renumbered_numbers)
    {
        printf("##### Speaker Diarization #####\n");

        for (int i = 0; i < renumbered_numbers.size(); i++)
        {
            printf("Speaker ID: %d  \n", renumbered_numbers[i]);
        }
    }
    
};

#endif