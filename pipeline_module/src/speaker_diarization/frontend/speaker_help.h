#ifndef SPEAKER_HELP_H
#define SPEAKER_HELP_H

#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <assert.h>

class Helper 
{
public:

    // for string delimiter
    static std::vector<std::string> split(std::string s, std::string delimiter) 
    {
        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        std::string token;
        std::vector<std::string> res;

        while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
            token = s.substr (pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;
            res.push_back (token);
        }

        res.push_back (s.substr (pos_start));
        return res;
    }

    // Mimic python np.rint
    // For values exactly halfway between rounded decimal values, 
    // NumPy rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round to 0.0, etc.
    template <typename T>
    static int np_rint( T val )
    {
        if(abs( val - int( val ) - 0.5 * ( val > 0 ? 1 : -1 )) < std::numeric_limits<double>::epsilon())
        {
            int tmp = std::round( val );
            if( tmp % 2 == 0 )
                return tmp;
            else
                return tmp - 1 * ( val > 0 ? 1 : -1 );
        }
        return std::round( val );
    }

    template <typename T>
    static std::vector<int> argsort(const std::vector<T> &v) 
    {

        // initialize original index locations
        std::vector<int> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        std::stable_sort(idx.begin(), idx.end(),
                [&v](int i1, int i2) {return v[i1] < v[i2];});

        return idx;
    }

    // python: hard_clusters = np.argmax(soft_clusters, axis=2)
    template <typename T>
    static std::vector<std::vector<int>> argmax( std::vector<std::vector<std::vector<T>>>& data )
    {
        std::vector<std::vector<int>> res( data.size(), std::vector<int>( data[0].size()));
        for( size_t i = 0; i < data.size(); ++i )
        {
            for( size_t j = 0; j < data[0].size(); ++j )
            {
                int max_index = 0;
                double max_value = -1.0 * std::numeric_limits<double>::max();
                for( size_t k = 0; k < data[0][0].size(); ++k )
                {
                    if( data[i][j][k] > max_value )
                    {
                        max_index = k;
                        max_value = data[i][j][k];
                    }
                }
                res[i][j] = max_index;
            }
        }

        return res;
    }

    // Define a helper function to find non-zero indices in a vector
    static std::vector<int> nonzeroIndices(const std::vector<bool>& input) 
    {
        std::vector<int> indices;
        for (int i = 0; i < input.size(); ++i) {
            if (input[i]) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    // Function to compute the L2 norm of a vector
    template <typename T>
    static float L2Norm(const std::vector<T>& vec) 
    {
        T sum = 0.0;
        for (T val : vec) 
        {
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    // Function to normalize a 2D vector
    template <typename T>
    static void normalizeEmbeddings(std::vector<std::vector<T>>& embeddings) 
    {
        for (std::vector<T>& row : embeddings) 
        {
            T norm = L2Norm(row);
            if (norm != 0.0) 
            {
                for (T& val : row) 
                {
                    val /= norm;
                }
            }
        }
    }

    // Function to calculate the Euclidean distance between two vectors
    template <typename T>
    static T euclideanDistance(const std::vector<T>& vec1, const std::vector<T>& vec2) {
        T sum = 0.0;
        for (size_t i = 0; i < vec1.size(); ++i) {
            T diff = static_cast<T>(vec1[i]) - static_cast<T>(vec2[i]);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // scipy.cluster.hierarchy.linkage with method: single
    template <typename T>
    static T clusterDistance_single( const std::vector<std::vector<T>>& embeddings,
            const std::vector<std::vector<T>>& distances,
            const std::vector<int>& cluster1, 
            const std::vector<int>& cluster2 )
    {
        T minDistance = 1e9;
        for( size_t i = 0; i < cluster1.size(); ++i )
        {
            if( cluster1[i] == -1 )
                return minDistance;
            // calc each cluster distance
            for( size_t j = 0; j < cluster2.size(); ++j )
            {
                if( cluster2[j] == -1 )
                    break;
                int _i = cluster1[i];
                int _j = cluster2[j];
                if( _i > _j )
                {
                    _i = cluster2[j];
                    _j = cluster1[i];
                }
                assert( _i != _j ); // same point cannot be in cluster1 and cluster2 at same time
                T dis = distances[_i][_j];
                if( dis < minDistance )
                    minDistance = dis;
            }
        }

        return minDistance;
    }
            
    // scipy.cluster.hierarchy.linkage with method: centroid
    template <typename T>
    static T clusterDistance_centroid( const std::vector<std::vector<T>>& embeddings,
            const std::vector<std::vector<T>>& distances,
            const std::vector<int>& cluster1, 
            const std::vector<int>& cluster2 )
    {
        T minDistance = 1e9;
        // d(i ∪ j, k) = αid(i, k) + αjd(j, k) + βd(i, j)
        // αi = |i| / ( |i|+|j| ), αj = |j| / ( |i|+|j| )
        // β = − |i||j| / (|i|+|j|)^2
        for( size_t i = 0; i < cluster1.size(); ++i )
        {
            if( cluster1[i] == -1 )
                return minDistance;
            // calc each cluster distance
            for( size_t j = 0; j < cluster2.size(); ++j )
            {
                if( cluster2[j] == -1 )
                    return minDistance;
                int _i = cluster1[i];
                int _j = cluster2[j];
                if( _i > _j )
                {
                    _i = cluster2[j];
                    _j = cluster1[i];
                }
                assert( _i != _j ); // same point cannot be in cluster1 and cluster2 at same time
                T dis = distances[_i][_j];
                if( dis < minDistance )
                    minDistance = dis;
            }
        }

        return minDistance;
    }

    // Function to calculate the mean of embeddings for large clusters
    template <typename T>
    static std::vector<std::vector<T>> calculateClusterMeans(const std::vector<std::vector<T>>& embeddings,
                                                 const std::vector<int>& clusters,
                                                 const std::vector<int>& largeClusters) 
    {
        std::vector<std::vector<T>> clusterMeans;

        for (int large_k : largeClusters) {
            std::vector<T> meanEmbedding( embeddings[0].size(), 0.0 );
            int count = 0;
            for (size_t i = 0; i < clusters.size(); ++i) {
                if (clusters[i] == large_k) {
                    // Add the embedding to the mean
                    for (size_t j = 0; j < meanEmbedding.size(); ++j) {
                        meanEmbedding[j] += embeddings[i][j];
                    }
                    count++;
                }
            }

            // Calculate the mean by dividing by the count
            if (count > 0) {
                for (size_t j = 0; j < meanEmbedding.size(); ++j) {
                    meanEmbedding[j] /= static_cast<T>(count);
                }
            }

            clusterMeans.push_back(meanEmbedding);
        }

        return clusterMeans;
    }

    // Function to calculate the cosine distance between two vectors
    template <typename T>
    static T cosineDistance(const std::vector<T>& vec1, const std::vector<T>& vec2) 
    {
        if (vec1.size() != vec2.size()) {
            throw std::runtime_error("Vector sizes must be equal.");
        }

        T dotProduct = 0.0;
        T magnitude1 = 0.0;
        T magnitude2 = 0.0;

        for (size_t i = 0; i < vec1.size(); ++i) {
            dotProduct += static_cast<T>(vec1[i]) * static_cast<T>(vec2[i]);
            magnitude1 += static_cast<T>(vec1[i]) * static_cast<T>(vec1[i]);
            magnitude2 += static_cast<T>(vec2[i]) * static_cast<T>(vec2[i]);
        }

        if (magnitude1 == 0.0 || magnitude2 == 0.0) {
            throw std::runtime_error("Vectors have zero magnitude.");
        }

        return 1.0 - (dotProduct / (std::sqrt(magnitude1) * std::sqrt(magnitude2)));
    }

    // Calculate cosine distances between large and small cluster means
    template <typename T>
    static std::vector<std::vector<T>> cosineSimilarity( std::vector<std::vector<T>>& largeClusterMeans,
            std::vector<std::vector<T>>& smallClusterMeans )
    {

        std::vector<std::vector<T>> centroidsCdist( largeClusterMeans.size(),
                std::vector<T>( smallClusterMeans.size()));
        for (size_t i = 0; i < largeClusterMeans.size(); ++i) {
            for (size_t j = 0; j < smallClusterMeans.size(); ++j) {
                T distance = cosineDistance(largeClusterMeans[i], smallClusterMeans[j]);
                centroidsCdist[i][j] = distance;
            }
        }

        return centroidsCdist;
    }

    // Function to find unique clusters and return the inverse mapping
    static std::vector<int> findUniqueClusters(const std::vector<int>& clusters,
                                        std::vector<int>& uniqueClusters) 
    {
        std::vector<int> inverseMapping(clusters.size(), -1);
        int nextClusterIndex = 0;

        // Find unique
        for (size_t i = 0; i < clusters.size(); ++i) 
        {
            std::vector<int>::iterator position = std::find( uniqueClusters.begin(), uniqueClusters.end(), clusters[i] );
            if (position == uniqueClusters.end()) 
            {
                uniqueClusters.push_back(clusters[i]);
            }
        }

        // Sort, python implementation like this
        std::sort(uniqueClusters.begin(), uniqueClusters.end());

        for (size_t i = 0; i < clusters.size(); ++i) 
        {
            std::vector<int>::iterator position = std::find( uniqueClusters.begin(), uniqueClusters.end(), clusters[i] );
            if (position != uniqueClusters.end()) 
            {
                inverseMapping[i] = position - uniqueClusters.begin();
            }
        }

        return inverseMapping;
    }

    // python: embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)
    template <typename T>
    static std::vector<std::vector<std::vector<T>>> rearrange_up( const std::vector<std::vector<T>>& input, int c )
    {
        assert( input.size() > c );
        assert( input.size() % c == 0 );
        size_t dim1 = c;
        size_t dim2 = input.size() / c;
        size_t dim3 = input[0].size();
        std::vector<std::vector<std::vector<T>>> output(c, 
                std::vector<std::vector<T>>( dim2, std::vector<T>(dim3, -1.0f)));
        for( size_t i = 0; i < dim1; ++i )
        {
            for( size_t j = 0; j < dim2; ++j )
            {
                for( size_t k = 0; k < dim3; ++k )
                {
                    output[i][j][k] = input[i*dim2+j][k];
                }
            }
        }

        return output;
    }

    // Imlemenation of einops.rearrange c s d -> (c s) d
    template <typename T>
    static std::vector<std::vector<T>> rearrange_down( const std::vector<std::vector<std::vector<T>>>& input )
    {
        int num_chunks = input.size();
        int num_frames = input[0].size();
        int num_classes = input[0][0].size();
        std::vector<std::vector<T>> data(num_chunks * num_frames, std::vector<T>(num_classes));
        for( size_t i = 0; i < num_chunks; ++i )
        {
            for( size_t j = 0; j < num_frames; ++j )
            {
                for( size_t k = 0; k < num_classes; ++k )
                {
                    data[ i * num_frames + j][k] = input[i][j][k];
                }
            }
        }

        return data;
    }

    // Imlemenation of einops.rearrange c f k -> (c k) f
    template <typename T>
    static std::vector<std::vector<T>> rearrange_other( const std::vector<std::vector<std::vector<T>>>& input )
    {
        int num_chunks = input.size();
        int num_frames = input[0].size();
        int num_classes = input[0][0].size();
        std::vector<std::vector<T>> data(num_chunks * num_classes, std::vector<T>(num_frames));
        int rowNum = 0;
        for ( const auto& row : input ) 
        {
            // Create a new matrix with swapped dimensions
            std::vector<std::vector<T>> transposed(num_classes, std::vector<T>(num_frames));

            for (int i = 0; i < num_frames; ++i) {
                for (int j = 0; j < num_classes; ++j) {
                    data[rowNum * num_classes + j][i] = row[i][j];
                }
            }

            rowNum++;
        }

        return data;
    }

    static std::vector<std::vector<int>> wellDefinedIndex( const std::vector<std::vector<bool>>& off_or_on ) 
    {
        // Find the indices of True values in each row and store them in a vector of vectors
        size_t max_indices = 0;
        std::vector<std::vector<int>> nonzero_indices;
        for (const auto& row : off_or_on) {
            std::vector<int> indices = nonzeroIndices(row);
            if( indices.size() > max_indices )
                max_indices = indices.size();
            nonzero_indices.push_back(indices);
        }

        // Fill missing indices with -1 and create the well_defined_idx vector of vectors
        std::vector<std::vector<int>> well_defined_idx;
        for (const auto& indices : nonzero_indices) {
            if( indices.size() < max_indices )
            {
                std::vector<int> filled_indices(max_indices, -1);
                std::copy(indices.begin(), indices.end(), filled_indices.begin());
                well_defined_idx.push_back(filled_indices);
            }
            else
            {
                well_defined_idx.push_back( indices );
            }
        }

        return well_defined_idx;
    }

    // Function to calculate cumulative sum along axis=1
    static std::vector<std::vector<int>> cumulativeSum(const std::vector<std::vector<bool>>& input) 
    {
        std::vector<std::vector<int>> cumsum;

        for (const auto& row : input) {
            std::vector<int> row_cumsum;
            int running_sum = 0;

            for (bool val : row) {
                running_sum += val ? 1 : 0;
                row_cumsum.push_back(running_sum);
            }

            cumsum.push_back(row_cumsum);
        }

        return cumsum;
    }

    // Define a helper function to calculate np.where
    static std::vector<std::vector<bool>> numpy_where(const std::vector<std::vector<int>>& same_as,
                                    const std::vector<std::vector<bool>>& on,
                                    const std::vector<std::vector<int>>& well_defined_idx,
                                    const std::vector<std::vector<bool>>& initial_state,
                                    const std::vector<std::vector<int>>& samples) 
    {
        assert( same_as.size() == on.size());
        assert( same_as.size() == well_defined_idx.size());
        assert( same_as.size() == initial_state.size());
        assert( same_as.size() == samples.size());
        assert( same_as[0].size() == on[0].size());
        assert( same_as[0].size() == well_defined_idx[0].size());
        assert( same_as[0].size() == initial_state[0].size());
        assert( same_as[0].size() == samples[0].size());
        std::vector<std::vector<bool>> result( same_as.size(), std::vector<bool>( same_as[0].size(), false ));
        for( size_t i = 0; i < same_as.size(); ++i )
        {
            for( size_t j = 0; j < same_as[0].size(); ++j )
            {
                if( same_as[i][j] > 0 )
                {
                    int x = samples[i][j];
                    int y = well_defined_idx[x][same_as[i][j]-1];
                    result[i][j] = on[x][y];
                }
                else
                {
                    result[i][j] = initial_state[i][j];
                }
            }
        }


        return result;
    }

    static std::vector<std::vector<std::vector<double>>> cleanSegmentations(
            const std::vector<std::vector<std::vector<double>>>& data)
    {
        size_t numRows = data.size();
        size_t numCols = data[0].size();
        size_t numChannels = data[0][0].size();

        // Initialize the result with all zeros
        std::vector<std::vector<std::vector<double>>> result(numRows, 
                std::vector<std::vector<double>>(numCols, std::vector<double>(numChannels, 0.0)));
        for (int i = 0; i < numRows; ++i) 
        {
            for (int j = 0; j < numCols; ++j) 
            {
                double sum = 0.0;
                for (int k = 0; k < numChannels; ++k) 
                {
                    sum += data[i][j][k];
                }
                bool keep = false;
                if( sum < 2.0 )
                {
                    keep = true;
                }
                for (int k = 0; k < numChannels; ++k) 
                {
                    if( keep )
                        result[i][j][k] = data[i][j][k];
                }
            }
        }

        return result;
    }

    // Define a function to interpolate 2D arrays (nearest-neighbor interpolation)
    static std::vector<std::vector<bool>> interpolate(const std::vector<std::vector<float>>& masks, 
            int num_samples, float threshold ) 
    {
        int inputHeight = masks.size();
        int inputWidth = masks[0].size();

        std::vector<std::vector<bool>> output(inputHeight, std::vector<bool>(num_samples, false));
        assert( num_samples > inputWidth );
        int scale = num_samples / inputWidth;

        for (int i = 0; i < inputHeight; ++i) 
        {
            for (int j = 0; j < num_samples; ++j) 
            {
                int src_y = j * inputWidth / num_samples;
                if( masks[i][src_y] > threshold )
                    output[i][j] = true;
            }
        }

        return output;
    }

    // Define a function to perform pad_sequence
    static std::vector<std::vector<float>> padSequence(const std::vector<std::vector<float>>& waveforms,
                                                const std::vector<std::vector<bool>>& imasks) 
    {
        // Find the maximum sequence length
        size_t maxLen = 0;
        for (const std::vector<bool>& mask : imasks) 
        {
            maxLen = std::max(maxLen, mask.size());
        }

        // Initialize the padded sequence with zeros
        std::vector<std::vector<float>> paddedSequence(waveforms.size(), std::vector<float>(maxLen, 0.0));

        // Copy the valid data from waveforms based on imasks
        for (size_t i = 0; i < waveforms.size(); ++i) 
        {
            size_t validIndex = 0;
            for (size_t j = 0; j < imasks[i].size(); ++j) 
            {
                if (imasks[i][j]) 
                {
                    paddedSequence[i][validIndex++] = waveforms[i][j];
                }
            }
        }

        return paddedSequence;
    }

};

#endif