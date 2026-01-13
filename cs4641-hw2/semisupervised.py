import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

SIGMA_CONST = 1e-06
LOG_CONST = 1e-32


def complete_(data):
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        labeled_complete: n x (D+1) array (n <= N) where values contain both complete features and labels
    """
    if data.size == 0:
        return data
    # end if
    features = data[:, :-1]
    label = data[:, -1]
    hasMissingFeature = np.any(np.isnan(features), axis=1)
    hasLabel = ~np.isnan(label)
    mask = (~hasMissingFeature) & hasLabel
    return data[mask]
# end complete_

def incomplete_(data):
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        labeled_incomplete: n x (D+1) array (n <= N) where values contain incomplete features but complete labels
    """
    if data.size == 0:
        return data
    # end if
    features = data[:, :-1]
    label = data[:, -1]
    hasMissingFeature = np.any(np.isnan(features), axis=1)
    hasLabel = ~np.isnan(label)
    mask = hasMissingFeature & hasLabel
    return data[mask]
# end incomplete_

def unlabeled_(data):
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        unlabeled_complete: n x (D+1) array (n <= N) where values contain complete features but incomplete labels
    """
    if data.size == 0:
        return data
    # end if
    features = data[:, :-1]
    label = data[:, -1]
    hasMissingFeature = np.any(np.isnan(features), axis=1)
    hasLabel = ~np.isnan(label)
    mask = (~hasMissingFeature) & (~hasLabel)
    return data[mask]
# end unlabeled_

class CleanData(object):

    def __init__(self):
        pass
    # end init

    def pairwise_dist(self, x, y):
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
            dist: N x M array, where dist[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
        """
        if x.size == 0 or y.size == 0:
            return np.zeros((x.shape[0], y.shape[0]))
        # end if
        x2 = np.sum(x**2, axis=1, keepdims=True)
        y2 = np.sum(y**2, axis=1)[np.newaxis, :]
        d2 = x2 + y2 - (2 * np.dot(x, y.T))
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)
    # end pairwise_dist

    def __call__(self, incomplete_points, complete_points, K, **kwargs):
        """
        Function to clean or "fill in" NaN values in incomplete data points based on
        the average value for that feature for the K-nearest neighbors in the complete data points.

        Args:
            incomplete_points: N_incomplete x (D+1) numpy array, the incomplete labeled observations
            complete_points:   N_complete   x (D+1) numpy array, the complete labeled observations
            K: integer, corresponding to the number of nearest neighbors you want to base your calculation on
            kwargs: any other args you want
        Return:
            clean_points: (N_complete + N_incomplete) x (D+1) numpy array, containing both the complete points and recently filled points

        Notes:
            (1) The first D columns are features, and the last column is the class label
            (2) There may be more than just 2 class labels in the data (e.g. labels could be 0,1,2 or 0,1,2,...,M)
            (3) There will be at most 1 missing feature value in each incomplete data point (e.g. no points will have more than one NaN value)
            (4) You want to find the k-nearest neighbors, from the complete dataset, with the same class labels;
            (5) There may be missing values in any of the features. It might be more convenient to address each feature at a time.
            (6) Do NOT use a for-loop over N_incomplete; you MAY use a for-loop over the M labels and the D features (e.g. omit one feature at a time)
            (7) You do not need to order the rows of the return array clean_points in any specific manner
        """
        if incomplete_points.size == 0:
            return complete_points.astype(float)
        # end if
        D = incomplete_points.shape[1] - 1
        incomplete = incomplete_points.astype(float).copy()
        complete = complete_points.astype(float)
        uniqueLabels = np.unique(incomplete[:, -1])

        for label in uniqueLabels:
            sameLabelMask = complete[:, -1] == label
            sameLabel = complete[sameLabelMask]

            for i in range(D):
                rowsToFill = (incomplete[:, -1] == label) & (np.isnan(incomplete[:, i]))
                if not np.any(rowsToFill):
                    continue
                # end if
                rowIdx = np.where(rowsToFill)[0]
                completeFeatures = np.ones(D, dtype=bool)
                completeFeatures[i] = False
                x = incomplete[rowIdx, :D][:, completeFeatures]
                y = sameLabel[:, :D][:, completeFeatures]
                if y.shape[0] == 0:
                    continue
                # end if
                d = self.pairwise_dist(x, y)
                k = min(max(int(K), 1), y.shape[0])
                neighborIdx = np.argpartition(d, k - 1, axis=1)[:, :k]
                neighborFeatureValues = sameLabel[:, i]
                assignedValues = np.mean(neighborFeatureValues[neighborIdx], axis=1)
                incomplete[rowIdx, i] = assignedValues
        # end for
        cleanedPoints = np.concatenate((complete, incomplete), axis=0)
        return cleanedPoints
# end __call__

def median_clean_data(data):
    """
    Args:
        data: N x (D+1) numpy array where only last column is guaranteed non-NaN values and is the labels
    Return:
        median_clean: N x (D+1) numpy array where each NaN value in data has been replaced by the median feature value
    Notes:
        (1) When taking the median of any feature, do not count the NaN value
        (2) Return all values to max one decimal point
        (3) The labels column will never have NaN values
    """
    if data.size == 0:
        return data
    # end if
    dataF = data.astype(float)
    features = dataF[:, :-1]
    labels = dataF[:, -1]

    featureMedians = np.nanmedian(features, axis=0)

    for i in range(features.shape[1]):
        missing = np.isnan(features[:, i])
        if np.any(missing):
            features[missing, i] = featureMedians[i]
        # end if
    # end for
    cleanedPoints = np.concatenate([features, labels.reshape(-1, 1)], axis=1)
    cleanedPoints = np.round(cleanedPoints, 1)
    return cleanedPoints
# end median_clean_data

class SemiSupervised(object):

    def __init__(self):
        pass
    # end init
    def softmax(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array where softmax has been applied row-wise to input logit
        """
        shiftedRowMax = logit - np.max(logit, axis=1, keepdims=True)
        expShifted = np.exp(shiftedRowMax)
        p = np.exp(shiftedRowMax) / np.sum(expShifted, axis=1, keepdims=True)
        return p
    # end softmax

    def logsumexp(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:])
        """
        rowMax = np.max(logit, axis=1, keepdims=True)
        shiftedRowMax = logit - rowMax
        sumExp = np.sum(np.exp(shiftedRowMax), axis=1, keepdims=True)
        s = rowMax + np.log(sumExp)
        return s
    # end logsumexp

    def normalPDF(self, logit, mu_i, sigma_i):
        """
        Args:
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: 1xN numpy array, the probability distribution of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        variances = np.maximum(np.diagonal(sigma_i), SIGMA_CONST)
        diff = logit - mu_i
        q = -0.5 * np.sum((diff * diff) / variances, axis=1)
        c = -0.5 * np.sum(np.log(2 * np.pi * variances))
        pdf = np.exp(c + q)
        return pdf
    # end normalPDF

    def _init_components(self, points, K, **kwargs):
        """
        Args:
            points: Nx(D+1) numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K; contains the prior probabilities of each class k
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.

        Hint:
            1. Given that the data is labeled, what's the best estimate for pi?
            2. Using the labels, you can look at individual clusters and estimate the best value for mu, sigma
        """
        features = points[:, :-1].astype(float)
        labels = points[:, -1]
        hasLabel = ~np.isnan(labels)
        labeledFeatures = features[hasLabel]
        knownLabels = labels[hasLabel].astype(int)
        numLabels = labeledFeatures.shape[0]
        D = features.shape[1]

        # Priors from labeled counts
        counts = np.bincount(knownLabels, minlength=K).astype(float)
        pi = counts / np.maximum(np.sum(counts), LOG_CONST)

        # Class-conditional means and diagonal covariances from labeled data
        mu = np.zeros((K, D), dtype=float)
        sigma = np.zeros((K, D, D), dtype=float)
        globalMean = np.mean(labeledFeatures, axis=0)
        globalVar = np.maximum(np.var(labeledFeatures, axis=0), SIGMA_CONST)
        for k in range(K):
            belongsToCluster = knownLabels == k
            if np.any(belongsToCluster):
                clusterFeatures = labeledFeatures[belongsToCluster]
                mu[k] = np.mean(clusterFeatures, axis=0)
                varDiag = np.var(clusterFeatures, axis=0)
            else:
                mu[k] = globalMean
                varDiag = globalVar
            # end if
            varDiag = np.maximum(varDiag, SIGMA_CONST)
            sigma[k] = np.diag(varDiag)
        # end for
        return pi, mu, sigma
    # end _init_components

    def _ll_joint(self, points, pi, mu, sigma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
        """
        N = points.shape[0]
        K = pi.shape[0]
        ll = np.empty((N, K), dtype=float)
        for k in range(K):
            pk = self.normalPDF(points, mu[k], sigma[k])
            # Numerical stability
            logPk = np.log(pk + LOG_CONST)
            logPi = np.log(pi[k] + LOG_CONST)
            ll[:, k] = logPi + logPk
        # end for
        return ll
    # end _ll_joint

    def _E_step(self, points, pi, mu, sigma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        jointLl = self._ll_joint(points, pi, mu, sigma)
        gamma = self.softmax(jointLl)
        return gamma
    # end _E_step
    
    def _M_step(self, points, gamma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.

        Hint:  There are formulas in the slide.
        """
        N, D = points.shape
        K = gamma.shape[1]
        nK = np.maximum(np.sum(gamma, axis=0), LOG_CONST)

        piNew = nK / N
        muNew = np.dot(gamma.T, points) / nK[:, None]
        sigmaNew = np.zeros((K, D, D), dtype=float)

        for k in range(K):
            diff = points - muNew[k]
            varDiag = np.sum(gamma[:, k][:, None] * (diff * diff), axis=0) / nK[k]
            varDiag = np.maximum(varDiag, SIGMA_CONST)
            sigmaNew[k] = np.diag(varDiag)
        # end for
        return piNew, muNew, sigmaNew
    # end _M_step
    
    def __call__(
        self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs
    ):
        """
        Args:
            points: N x (D+1) numpy array, where
                - N is # points,
                - D is the number of features,
                - the last column is the point labels (when available) or NaN for unlabeled points
            K: integer, number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            pi, mu, sigma: (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint: Look at Table 1 in the paper
        """
        features = points[:, :-1].astype(float)
        labels = points[:, -1]
        N, D = features.shape
        hasLabel = ~np.isnan(labels)
        unlabeled = np.isnan(labels)
        if np.any(hasLabel):
            knownLabels = labels[hasLabel].astype(int)
        else:
            knownLabels = np.array([], dtype=int)
        # end if
        pi, mu, sigma = self._init_components(points, K, **kwargs)

        prevLoss = None
        for _ in range(max_iters):
            # E-step: unlabeled points
            gammaUnlabeled = self._E_step(features[unlabeled], pi, mu, sigma)

            # Labeled points
            if np.any(hasLabel):
                numLabeled = int(np.sum(hasLabel))
                gammaLabeled = np.zeros((numLabeled, K), dtype=float)
                gammaLabeled[np.arange(numLabeled), knownLabels] = 1.0
            else:
                gammaLabeled = np.zeros((0, K), dtype=float)
            # end if
            # Combine gamma in original order
            gamma = np.zeros((N, K), dtype=float)
            if np.any(hasLabel):
                gamma[hasLabel] = gammaLabeled
            # end if
            if np.any(unlabeled):
                gamma[unlabeled] = gammaUnlabeled
            # end if
            # M-step with all points
            pi, mu, sigma = self._M_step(features, gamma)

            # Compute semi-supervised loss for convergence
            lL = self._ll_joint(features, pi, mu, sigma)
            if np.any(unlabeled):
                lossUnlabeled = -np.sum(self.logsumexp(lL[unlabeled]))
            else:
                lossUnlabeled = 0.0
            # end if
            lossLabeled = 0.0
            if np.any(hasLabel):
                rows = np.arange(np.sum(hasLabel))
                llLabeled = lL[hasLabel]
                lossLabeled = -np.sum(llLabeled[rows, knownLabels])
            # end if
            loss = float(lossUnlabeled + lossLabeled)

            if prevLoss is not None:
                absChange = abs(loss - prevLoss)
                relChange = absChange / max(abs(prevLoss), LOG_CONST)
                if absChange < abs_tol or relChange < rel_tol:
                    break
                # end if
            # end if
            prevLoss = loss
        # end for
        return pi, mu, sigma
    # end __call__

class ComparePerformance(object):

    def __init__(self):
        pass

    @staticmethod
    def accuracy_semi_supervised(training_data, validation_data, K: int) -> float:
        """
        Train a classification model using your SemiSupervised object on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data

        Args:
            training_data: N_t x (D+1) numpy array, where
                - N_t is the number of data points in the training set,
                - D is the number of features, and
                - the last column represents the labels (when available) or a flag that allows you to separate the unlabeled data.
            validation_data: N_v x(D+1) numpy array, where
                - N_v is the number of data points in the validation set,
                - D is the number of features, and
                - the last column are the labels
            K: integer, number of clusters for SemiSupervised object
        Return:
            accuracy: floating number

        Note: validation_data will NOT include any unlabeled points
        """
        pi, mu, sigma = SemiSupervised()(training_data, K)
        classification_probs = SemiSupervised()._E_step(
            validation_data[:, :-1], pi, mu, sigma
        )
        classification = np.argmax(classification_probs, axis=1)
        semi_supervised_score = accuracy_score(validation_data[:, -1], classification)
        return semi_supervised_score

    @staticmethod
    def accuracy_GNB(training_data, validation_data) -> float:
        """
        Train a Gaussion Naive Bayes classification model (sklearn implementation) on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data

        Args:
            training_data: N_t x (D+1) numpy array, where
                - N is the number of data points in the training set,
                - D is the number of features, and
                - the last column represents the labels
            validation_data: N_v x (D+1) numpy array, where
                - N_v is the number of data points in the validation set,
                - D is the number of features, and
                - the last column are the labels
        Return:
            accuracy: floating number

        Note: both training_data and validation_data will NOT include any unlabeled points
        """
        gnb_model = GaussianNB()
        gnb_model.fit(training_data[:, :-1], training_data[:, -1])
        gnb_score = gnb_model.score(validation_data[:, :-1], validation_data[:, -1])
        return gnb_score

    @staticmethod
    def accuracy_comparison():
        all_data = np.loadtxt("data/data.csv", delimiter=",")
        labeled_complete = complete_(all_data)
        labeled_incomplete = incomplete_(all_data)
        unlabeled = unlabeled_(all_data)
        cleaned_data = CleanData()(labeled_incomplete, labeled_complete, 10)
        cleaned_and_unlabeled = np.concatenate((cleaned_data, unlabeled), 0)
        labeled_data = np.concatenate((labeled_complete, labeled_incomplete), 0)
        median_cleaned_data = median_clean_data(labeled_data)
        print(f"All Data shape:                 {all_data.shape}")
        print(f"Labeled Complete shape:         {labeled_complete.shape}")
        print(f"Labeled Incomplete shape:       {labeled_incomplete.shape}")
        print(f"Labeled shape:                  {labeled_data.shape}")
        print(f"Unlabeled shape:                {unlabeled.shape}")
        print(f"Cleaned data shape:             {cleaned_data.shape}")
        print(f"Cleaned + Unlabeled data shape: {cleaned_and_unlabeled.shape}")
        validation = np.loadtxt("data/validation.csv", delimiter=",")
        accuracy_complete_data_only = ComparePerformance.accuracy_GNB(
            labeled_complete, validation
        )
        accuracy_cleaned_data = ComparePerformance.accuracy_GNB(
            cleaned_data, validation
        )
        accuracy_median_cleaned_data = ComparePerformance.accuracy_GNB(
            median_cleaned_data, validation
        )
        accuracy_semi_supervised = ComparePerformance.accuracy_semi_supervised(
            cleaned_and_unlabeled, validation, 2
        )
        print("===COMPARISON===")
        print(
            f"Supervised with only complete data, GNB Accuracy: {np.round(100.0 * accuracy_complete_data_only, 3)}%"
        )
        print(
            f"Supervised with KNN clean data, GNB Accuracy:     {np.round(100.0 * accuracy_cleaned_data, 3)}%"
        )
        print(
            f"Supervised with Median clean data, GNB Accuracy:    {np.round(100.0 * accuracy_median_cleaned_data, 3)}%"
        )
        print(
            f"SemiSupervised Accuracy:                          {np.round(100.0 * accuracy_semi_supervised, 3)}%"
        )
