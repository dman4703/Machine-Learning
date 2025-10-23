import numpy as np


class HierarchicalClustering(object):

    def __init__(self, points: np.ndarray):
        self.N, self.D = points.shape
        self.points = points
        self.current_iteration = 0
        self.distances, self.cluster_ids = self.create_distances(points)
        self.clustering = np.zeros((self.N - 1, 4))
        self.cluster_sizes = np.zeros(self.N * 2 - 1)
        self.cluster_sizes[: self.N] = 1
    # end init

    def create_distances(self, points: np.ndarray) -> np.ndarray:
        """
        Create the pairwise distance matrix and index map, given points.
            The distance between a cluster and itself should be np.inf
        Args:
            points: N x D numpy array where N is the number of points
        Return:
            distances: N x N numpy array where distances[i][j] is the euclidean distance between points[i, :] and points[j, :].
                       distances[i, i] should always be np.inf in order to calculate the closest clusters more easily
            cluster_ids: (N,) numpy array where index_array[i] gives the cluster id of the i-th column
                         and i-th row of distances. Initially, each point i is assigned cluster id i
        """
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        cluster_ids = np.arange(points.shape[0])
        return distances, cluster_ids
    # end create_distances

    def iterate(self):
        """
        Performs one iteration of the algorithm
            1. Find the two closest clusters using self.distances (if there are multiple minimums, use the first occurence in flattened array)
            2. Replace first cluster's row and col with the newly combined cluster distances in self.distances,
               ensuring distances[i, i] is still np.inf
            3. Delete second cluster's row and col in self.distances
            4. Update self.cluster_ids where new cluster's id should be self.N + self.current_iteration,
               see definition in `create_distances` for more details
            5. Update self.cluster_sizes, where self.cluster_sizes[i] contains the number of points with cluster id i
            6. Update self.clustering, where
               self.clustering[self.current_iteration] = [first cluster id, second cluster id, distance between first and second clusters, size of new cluster]
            7. Update current_iteration
        Hint:
        You'll need to update self.distances, self.cluster_ids, self.cluster_sizes, self.clustering, and self.current_iteration

        While self.distances and self.cluster_ids only keeps information about the current clusters,
            self.cluster_sizes keep track of sizes for all clusters
        """
        minDistance = np.argmin(self.distances)
        rowIdx, colIdx = np.unravel_index(minDistance, self.distances.shape)
        firstCluster = int(self.cluster_ids[rowIdx])
        secondCluster = int(self.cluster_ids[colIdx])
        mergeDistance = float(self.distances[rowIdx, colIdx])

        updatedRow = np.minimum(self.distances[rowIdx, :], self.distances[colIdx, :])
        updatedCol = np.minimum(self.distances[:, rowIdx], self.distances[:, colIdx])
        self.distances[rowIdx, :] = updatedRow
        self.distances[:, rowIdx] = updatedCol
        self.distances[rowIdx, rowIdx] = np.inf

        self.distances = np.delete(self.distances, colIdx, axis=0)
        self.distances = np.delete(self.distances, colIdx, axis=1)

        mergedCluster = self.N + self.current_iteration
        self.cluster_ids[rowIdx] = mergedCluster
        self.cluster_ids = np.delete(self.cluster_ids, colIdx)

        self.cluster_sizes[mergedCluster] = (self.cluster_sizes[firstCluster] + self.cluster_sizes[secondCluster])

        self.clustering[self.current_iteration] = [
            firstCluster,
            secondCluster,
            mergeDistance,
            self.cluster_sizes[mergedCluster],
        ]

        self.current_iteration += 1
    # end iterate

    def fit(self):
        """
        Fits the model on the dataset by calling `iterate`.
        Each call of `iterate` should combine two clusters, logging what was combined in self.clustering

        Return:
            self.clustering, where self.clustering[iteration_index] = [i, j, distance between i and j, size of new cluster]
        """
        while self.current_iteration < self.N - 1:
            self.iterate()
        return self.clustering
    # end fit
