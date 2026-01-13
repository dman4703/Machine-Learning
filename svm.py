import numpy as np
from sklearn.svm import SVC


def feature_construction(X: np.ndarray) -> np.ndarray:
    """
    Given the data, perform some transformation that will make the data linearly separable.

    Arguments:
        X: (10, 2) np.array - The data shown in the notebook.
    Returns:
        X_aug: (10, D) np.array - Some transformation of the data
    Hints:
        - The data may come shuffled, so don't handcode anything with specific indices.
        - Adding more features can't hurt separability. While you can carefully construct a great 1 feature boundary, it may be easier to just construct additional features.
    """
    featureOne = X[:, 0]
    featureTwo = X[:, 1]

    xAug = np.column_stack([
        featureOne,
        featureTwo,
        featureOne ** 2,
        featureTwo ** 2,
        featureOne * featureTwo,
    ])

    return xAug
# end feature_construction


def kernel_construction(X: np.ndarray, phi: callable) -> np.ndarray:
    """
    Given a dataset and a callable feature map, construct a kernel matrix, K.
    Simply, K[i,j] = phi(x_i) . phi(x_j)

    Args:
      X: np.ndarray(N, D)[float]; the dataset
      phi: callable; takes (D,) returns (D',), some feature engineering map
    Returns:
      K: np.ndarray(N, N)[float]; the resultant kernel
    Hints:
      - You can do smart broadcasting or symmetric speedup, or you can just loop and calculate elementwise.
    """
    numSamples = X.shape[0]
    K = np.zeros((numSamples, numSamples))

    for i in range(numSamples):
      for j in range(numSamples):
          phiXi = phi(X[i])
          phiXj = phi(X[j])
          K[i, j] = np.dot(phiXi, phiXj)
      # end for
    # end for
    return K
# end kernel_construction


def rbf_kernel(X: np.ndarray, gamma: float) -> np.ndarray:
    """
    Given a dataset and the gamma hyperparameter, build the radial basis function kernel.
    K[i, j] = exp(-gamma * ||x_i - x_j||^2)

    Args:
      X: np.ndarray(N, D)[float]; the dataset
      phi: callable; takes (D,) returns (D',), some feature engineering map
    Returns:
      K: np.ndarray(N, N)[float]; the resultant kernel
    Hints:
      - You can do smart broadcasting or symmetric speedup, or you can just loop and calculate elementwise.
    """
    numSamples = X.shape[0]
    K = np.zeros((numSamples, numSamples))

    for i in range(numSamples):
      for j in range(numSamples):
          diff = X[i] - X[j]
          sqDist = np.dot(diff, diff)
          K[i, j] = np.exp(-gamma * sqDist)
      # end for
    # end for
    return K
# end rbf_kernel
