## 5: Hierarchical Clustering [9 pts Grad / 4.5% Bonus for Undergrad]### 5.1 Hierarchical Clustering Implementation [9pts Grad/4.5% Bonus for Undergrad]

Hierarchical Clustering is a bottom-up agglomerative clustering algorithm which iteratively combines the closest pair of clusters. Each point starts off as its own cluster, and in each iteration you'll find the closest clusters and update the distances to the new cluster using single-link clustering, keeping track of the order in which the clusters are combined. In this section, you'll implement the `create_distance_matrix`, `iterate`, and `fit` methods in **hierarchical_clustering.py**. 

The `HierarchicalClustering` class has several instance variables that you may need to create and update in each iteration:
1. `points`: N x D numpy array where N is the number of points, and D is the dimensionality of each point. This is your dataset.
2. `distance`: N x N symmetric numpy array which stores pairwise distances between clusters. The distance between a cluster and itself should be `np.inf` in order to help us calculate the closest pair later
3. `cluster_ids`: (N,) numpy array where index_array[i] gives the cluster id of the i-th column and i-th row of distances. Initially, each point with index `points[i, :]` is assigned cluster id i, and new points are assigned cluster ids starting from `N` and incrementing.
4. `clustering`: (N - 1, 4) numpy array that keeps track of which clusters were merged in each iteration. `clustering[iteration_number]` keeps track of the first cluster id, second cluster id, distance between first and second clusters, and the size of new cluster
5. `cluster_sizes` (2N - 1, ) numpy array that stores the number of points in each cluster, indexed by id. Because there are `N` original clusters corresponding to each point, and each iteration merges two clusters, there will be `2N-1` total clusters created. 

Theses are the following functions you'll have to implement in **hierarchical_clustering.py**:

1. `create_distances`: Creates the initial distance matrix and cluster ids
2. `iterate`: Merges the two closest clusters
3. `fit`: Calls `iterate` multiple times and returns the clusterings

