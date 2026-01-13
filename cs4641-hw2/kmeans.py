import numpy as np


class KMeans(object):

    def __init__(self, points, k, init="random", max_iters=10000, rel_tol=1e-05):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == "random":
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters
    # end init
    def init_centers(self):
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Use np.random.choice to initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        return self.points[np.random.choice(self.points.shape[0], self.K, False)]
    # end init_centers

    def kmpp_init(self):
        """
            Use the intuition that points further away from each other will probably be better initial centers.
            To complete this method, refer to the steps outlined below:.
            1. Sample 1% of the points from dataset, uniformly at random (UAR) and without replacement.
            This sample will be the dataset the remainder of the algorithm uses to minimize initialization overhead.
            2. From the above sample, select only one random point to be the first cluster center.
            3. For each point in the sampled dataset, find the nearest cluster center and record the squared distance to get there.
            4. Examine all the squared distances and take the point with the maximum squared distance as a new cluster center.
            In other words, we will choose the next center based on the maximum of the minimum calculated distance
            instead of sampling randomly like in step 2. You may break ties arbitrarily.
            5. Repeat 3-4 until all k-centers have been assigned. You may use a loop over K to keep track of the data in each cluster.
        Return:
            self.centers : K x D numpy array, the centers.
        Hint:
            You could use functions like np.vstack() here.
        """
        N, D = self.points.shape
        
        #1% sample
        sampleSize = min(N, max(self.K, int(np.ceil(0.01 * N))))
        sample = self.points[np.random.choice(N, sampleSize, False)]
        
        # first center
        centerIndices = []
        centers = []
        first = np.random.randint(sampleSize)
        centerIndices.append(first)
        centers.append(sample[first])
    
        # record min squared distance to cluster
        minSqD = np.full(sampleSize, np.inf, dtype=float)
        def updateMinSqD(center):
            sqD = np.sum((sample - center) ** 2, axis=1)
            np.minimum(minSqD, sqD, out=minSqD)
        # end updateMinSqD
        updateMinSqD(centers[0])

        # new center + repeat
        while len(centers) < self.K:
            nextIndex = int(np.argmax(minSqD))
            if nextIndex not in centerIndices:
                centerIndices.append(nextIndex)
                newCenter = sample[nextIndex]
                centers.append(newCenter)
                updateMinSqD(newCenter)
            # end if
        # end while
        return np.vstack(centers)
    # end kmpp_init

    def update_assignment(self):
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: Do not use loops for the update_assignment function
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison.
        """
        d = pairwise_dist(self.points, self.centers)
        self.assignments = np.argmin(d, axis=1)
        return self.assignments

    def update_centers(self):
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        HINT: If there is an empty cluster then it won't have a cluster center, in that case the number of rows in self.centers can be less than K.
        """
        newCenters = []
        for i in range(self.K):
            inCluster = (self.assignments == i)
            if np.any(inCluster):
                newCenters.append(self.points[inCluster].mean(axis=0))
            # end if
        # end for

        self.centers = np.array(newCenters, dtype=float)
        return self.centers
    # end update_centers

    def get_loss(self):
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        sqD = np.sum((self.points - self.centers[self.assignments]) ** 2, axis=1)
        self.loss = float(np.sum(sqD))
        return self.loss

    def train(self):
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster,
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned,
                     pick a random point in the dataset to be the new center and
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference
                     in loss compared to the previous iteration is less than the given
                     relative tolerance threshold (self.rel_tol).
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!

        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.

        HINT: Do not loop over all the points in every iteration. This may result in time out errors
        HINT: Make sure to care of empty clusters. If there is an empty cluster the number of rows in self.centers can be less than K.
        """
        N, D = self.points.shape
        prevLoss = None
        for _ in range(self.max_iters):
            self.update_assignment()
            nonemptyClusters = np.unique(self.assignments)
            if nonemptyClusters.size < self.K:
                emptyClusters = np.setdiff1d(np.arange(self.K), nonemptyClusters, assume_unique=True)
                randIndices = np.random.randint(0, N, size=emptyClusters.size)
                for c, i in zip(emptyClusters, randIndices):
                    self.centers[c] = self.points[i]
                # end for
                self.update_assignment()
            # end if
            self.update_centers()
            self.get_loss()
            if prevLoss is not None:
                diff = abs(self.loss - prevLoss) / prevLoss
                if diff < self.rel_tol:
                    break
            # end if
            prevLoss = self.loss
        # end for
        return self.centers, self.assignments, self.loss
# end train

def pairwise_dist(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]

    HINT: Do not use loops for the pairwise_distance function
    """
    x2 = np.sum(x**2, axis=1, keepdims=True)
    y2 = np.sum(y**2, axis=1)[np.newaxis, :]
    d2 = x2 + y2 - (2 * np.dot(x, y.T))
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)
# end pairwise_dist

def pairwise_dist_inf(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist[i, j] is the infinity or chebyshev distance between
            x[i, :] and y[j, :]

    HINT: Do not use loops for the pairwise_dist_inf function
    """
    return np.max(np.abs(x[:, None, :] - y[None, :, :]), axis=2)
# end pairwise_dist_inf

def adjusted_rand_statistic(xGroundTruth, xPredicted):
    """
    Args:
        xPredicted : list of length N where N = no. of test samples
        xGroundTruth: list of length N where N = no. of test samples
    Return:
        adjusted rand index value: final coefficient value of type np.float64

    HINT: You can use loops for this function.
    HINT: The idea is to make the comparison of Predicted and Ground truth data points.
        1. Iterate over all distinct pairs of points.
        2. Compare the prediction pair label with the ground truth pair.
        3. Based on the analysis, we can figure out whether both points fall under TP/FP/FN/FP
           i.e. if a pair falls under TP, increment by 2 (one for each point in the pair).
        4. Then calculate the adjusted rand index value
    """
    y = np.asarray(xGroundTruth)
    yHat = np.asarray(xPredicted)
    N = y.shape[0]

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(N - 1):
        for j in range(i+1, N):
            correctY = (y[i] == y[j])
            correctYHat = (yHat[i] == yHat[j])
            if correctY and correctYHat:
                TP += 1
            elif not correctY and not correctYHat:
                TN += 1
            elif not correctY and correctYHat:
                FP += 1
            else:
                FN += 1
        #end for
    # end for

    return np.float64((2 * (TP * TN - FP * FN)) / (((TP + FN) * (FN + TN)) + ((TP + FP) * (FP + TN))) )
# end adjusted_rand_statistic

def silhouette_score(X, labels):
    """
    Args:
        X : N x D numpy array, where N is # points and D is the dimensionality
        labels : 1D numpy array of predicted labels of length N where N = no. of test samples
    Return:
        silhouette score: final coefficient value of type np.float64

    HINT: You can use loops for this function.
    HINT: The idea is to calculate the mean distance between a point and the other points
    in its cluster (the intra cluster distance) and the mean distance between a point and the
    other points in its closest cluster (the inter cluster distance)
        1.  Calculate the pairwise_dist between all points to each other (N x N)
        2.  Loop over all points in the provided data (X) and for each point calculate:

            Intra Cluster Distances (point p to the other points in its own cluster)
                a. Identify all points in the same cluster as p (excluding p itself)
                b. Calculate the mean pairwise_dist between p and the other points
                c. If there are no other points in the same cluster, assign an
                   intra-cluster distance of 0

            Inter Cluster Distances (point p to the points in the closest cluster)
                a. Loop over all clusters except for p's cluster
                b. For each cluster, identify all points belonging to it
                c. Calculate the mean pairwise_dist between p and those points
                d. Set the inter-cluster distance to the minimum mean pairwise_dist
                   among all clusters. Again, if there are no other clusters, use
                   an inter-cluster distance of 0.

        3. Calculate the silhouette scores for each point using
                s_i = (mu_out(x_i) - mu_in(x_i)) / max(mu_out(x_i), mu_in(x_i))
        4. Average the silhouette score across all points

    Note: Refer to the Clustering Evaluation slides from Lecture
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    d = pairwise_dist(X, X)
    silScores = np.zeros(N, dtype=float)
    uniqueClusters = np.unique(labels)
    pointsByCluster = {}
    for c in uniqueClusters:
        indices = np.where(labels == c)[0]
        pointsByCluster[c] = indices
    # end for
    for i in range(N):
        ci = labels[i]
        sameCluster = pointsByCluster[ci]
        if sameCluster.size <= 1:
            a = 0.0
        else: 
            a = np.sum(d[i, sameCluster]) / (sameCluster.size - 1)
        # end if
        b = np.inf
        for cj in uniqueClusters:
            if cj == ci:
                continue
            # end if
            otherCluster = pointsByCluster[cj]
            if otherCluster.size == 0:
                continue
            # end if
            meanD = np.mean(d[i, otherCluster])
            b = min(b, meanD)
        # end for
        if not np.isfinite(b):
            b = 0.0
        # end if
        
        silScores[i] = (b - a) / max(a, b)
    # end for
    return np.float64(np.mean(silScores))
# end silhouette_score
