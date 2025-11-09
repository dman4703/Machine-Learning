from typing import List, Tuple

import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted values
            label: (N, 1) numpy array, the ground truth values
        Return:
            A float value denoting the error between real and predicted
        """
        return float(np.sqrt(np.mean(np.square(pred - label))))
    # end rmse

    def construct_polynomial_feats(self, x: np.ndarray, degree: int) -> np.ndarray:
        """
        Given a feature matrix x, create a new feature matrix
        which is all powers of the features up to the provided degree.

        Args:
            x:
                1-dimensional case: (N,) numpy array
                D-dimensional case: (N, D) numpy array
                where N is the number of instances and D is the dimensionality of each instance
            degree: the max polynomial degree
        Return:
            feat:
                when x is 1-dimensional: (N, degree+1) numpy array
                Remember to include the bias term!
                [[1.0, x1, x1^2, x1^3, ...,],
                 [1.0, x2, x2^2, x2^3, ...,],
                 ...]
                when x is D-dimensional: (N, degree+1, D) numpy array
                Remember to include the bias term!
                [[[ 1.0        1.0]
                  [ x_{1,1}    x_{1,2}]
                  [ x_{1,1}^2  x_{1,2}^2]
                  [ x_{1,1}^3  x_{1,2}^3]
                 ]

                 [[ 1.0        1.0]
                  [ x_{2,1}    x_{2,2}]
                  [ x_{2,1}^2  x_{2,2}^2]
                  [ x_{2,1}^3  x_{2,2}^3]]

                 [[ 1.0        1.0]
                  [ x_{3,1}    x_{3,2}]
                  [ x_{3,1}^2  x_{3,2}^2]
                  [ x_{3,1}^3  x_{3,2}^3]]
                ]
        Hints: It is acceptable to loop over the degrees.
        """
        if x.ndim == 1:
            N = x.shape[0]
            feats = np.empty((N, degree + 1), dtype=float)
            for d in range(degree + 1):
                if d == 0:
                    feats[:, d] = 1.0
                else:
                    feats[:, d] = np.power(x, d)
                # end if
            # end for
            return feats
        # end if

        N, D = x.shape
        feats = np.empty((N, degree + 1, D), dtype=float)
        for d in range(degree + 1):
            if d == 0:
                feats[:, d, :] = 1.0
            else:
                feats[:, d, :] = np.power(x, d)
            # end if
        # end for
        return feats
    # end construct_polynomial_feats

    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """
        Using regression weights, predict the values for each data point in the xtest array

        Args:
            xtest: (N,1+D) numpy array, where N is the number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            weight: (1+D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        return np.dot(xtest, weight)
    # end predict

    def linear_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray) -> np.ndarray:
        """
        Fit a linear regression model using the closed form solution

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
        Hints:
            - For pseudo inverse, you should use the numpy linear algebra function (np.linalg.pinv)
        """
        return np.dot(np.linalg.pinv(xtrain), ytrain)
    # end linear_fit_closed

    def linear_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a linear regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        Hints:
            - RMSE loss should be recorded AFTER the gradient update in each iteration.
        """
        N, M = xtrain.shape
        weight = np.zeros((M, 1), dtype=float)
        lossPerEpoch: List[float] = []

        for _ in range(epochs):
            pred = np.dot(xtrain, weight)
            error = pred - ytrain
            gradient = np.dot(xtrain.T, error) / N
            weight = weight - (learning_rate * gradient)
            lossPerEpoch.append(self.rmse(np.dot(xtrain, weight), ytrain))
        # end for
        return weight, lossPerEpoch
    # end linear_fit_GD
    
    def linear_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a linear regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step
        Hints:
            - RMSE loss should be recorded AFTER the gradient update in each iteration.
            - Keep in mind that the number of epochs is the number of complete passes
            through the training dataset. SGD updates the weight for one datapoint at
            a time. For each epoch, you'll need to go through all of the points.

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
        """
        N, M = xtrain.shape
        weight = np.zeros((M, 1), dtype=float)
        lossPerStep: List[float] = []

        for _ in range(epochs):
            for i in range(N):
                xi = xtrain[i : i + 1, :]
                yi = ytrain[i : i + 1, :]
                predi = np.dot(xi, weight)
                errori = predi - yi
                gradienti = xi.T * float(errori)
                weight = weight - (learning_rate * gradienti)
                lossPerStep.append(self.rmse(np.dot(xtrain, weight), ytrain))
            # end for
        # end for
        return weight, lossPerStep
    # end linear_fit_SGD

    def linear_fit_MBGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        batch_size: int = 5,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a linear regression model using mini-batch gradient descent. Think of
        mini-batch gradient descent as an "in-between" of linear_fit_GD and
        linear_fit_SGD, where instead of processing one data point at a time, we are
        processing a "batch" of data per iteration. MBGD is generally used more in
        the industry due to its practicality.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            batch_size: int, size of each batch (set to 5)
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weights: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step
        Hints:
            - RMSE loss should be recorded AFTER the gradient update in each iteration.
            - Keep in mind that the number of epochs is the number of complete passes
            through the training dataset. MBGD updates the weight for a group of datapoints at
            a time. For each epoch, you'll need to go through all of the points.
        """
        N, M = xtrain.shape
        weight = np.zeros((M, 1), dtype=float)
        lossPerStep: List[float] = []

        for _ in range(epochs):
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                Xb = xtrain[start:end, :]
                yb = ytrain[start:end, :]
                B = end - start
                predb = np.dot(Xb, weight)
                errorb = predb - yb
                gradientb = np.dot(Xb.T, errorb) / B
                weight = weight - (learning_rate * gradientb)
                lossPerStep.append(self.rmse(np.dot(xtrain, weight), ytrain))
            # end for
        # end for
        return weight, lossPerStep
    # end linear_fit_MBGD
    
    def ridge_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float
    ) -> np.ndarray:
        """
        Fit a ridge regression model using the closed form solution

        Args:
            xtrain: (N,1+D) numpy array, where N is
                    number of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
        Return:
            weight: (1+D,1) numpy array, the weights of ridge regression model
        Hints:
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """
        M = xtrain.shape[1]
        I = np.eye(M, dtype=float)
        I[0, 0] = 0.0
        A = np.dot(xtrain.T, xtrain) + (c_lambda * I)
        weight = np.dot(np.linalg.inv(A), (np.dot(xtrain.T, ytrain)))
        return weight
    # end ridge_fit_closed

    def ridge_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 500,
        learning_rate: float = 1e-07,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a ridge regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        Hints:
            - RMSE loss should be recorded AFTER the gradient update in each iteration.
            - You should avoid applying regularization to the bias term in the gradient update
        """
        N, M = xtrain.shape
        weight = np.zeros((M, 1), dtype=float)
        lossPerEpoch: List[float] = []

        for _ in range(epochs):
            pred = np.dot(xtrain, weight)
            error = pred - ytrain
            gradient = np.dot(xtrain.T, error) / N
            regW = weight.copy()
            regW[0, 0] = 0.0
            gradient += (c_lambda / N) * regW
            weight = weight - (learning_rate * gradient)
            lossPerEpoch.append(self.rmse(np.dot(xtrain, weight), ytrain))
        # end for
        return weight, lossPerEpoch
    # end ridge_fit_GD
    
    def ridge_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a ridge regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step
        Hints:
            - RMSE loss should be recorded AFTER the gradient update in each iteration.
            - Keep in mind that the number of epochs is the number of complete passes
            through the training dataset. SGD updates the weight for one datapoint at
            a time. For each epoch, you'll need to go through all of the points.
            - You should avoid applying regularization to the bias term in the gradient update

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
        """
        N, M = xtrain.shape
        weight = np.zeros((M, 1), dtype=float)
        lossPerStep: List[float] = []

        for _ in range(epochs):
            for i in range(N):
                xi = xtrain[i : i + 1, :]
                yi = ytrain[i : i + 1, :]
                predi = np.dot(xi, weight)
                errori = predi - yi
                gradienti = xi.T * float(errori)
                regw = weight.copy()
                regw[0, 0] = 0.0
                gradienti += (c_lambda / N) * regw
                weight = weight - (learning_rate * gradienti)
                lossPerStep.append(self.rmse(np.dot(xtrain, weight), ytrain))
            # end for
        # end for
        return weight, lossPerStep
    # end ridge_fit_SGD
    
    def ridge_fit_MBGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        batch_size: int = 5,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a ridge regression model using mini-batch gradient descent. Think of
        mini-batch gradient descent as an "in-between" of linear_fit_GD and
        linear_fit_SGD, where instead of processing one data point at a time, we are
        processing a "batch" of data per iteration. MBGD is generally used more in
        the industry due to its practicality.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float, value of regularization constant
            batch_size: int, size of each batch (set to 5)
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weights: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step
        Hints:
            - RMSE loss should be recorded AFTER the gradient update in each iteration.
            - Keep in mind that the number of epochs is the number of complete passes
            through the training dataset. MBGD updates the weight for a group of datapoints at
            a time. For each epoch, you'll need to go through all of the points.
            - You should avoid applying regularization to the bias term in the gradient update
        """
        N, M = xtrain.shape
        weight = np.zeros((M, 1), dtype=float)
        lossPerStep: List[float] = []

        for _ in range(epochs):
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                Xb = xtrain[start:end, :]
                yb = ytrain[start:end, :]
                B = end - start
                predb = np.dot(Xb, weight)
                errorb = predb - yb
                gradientb = np.dot(Xb.T, errorb) / B
                regw = weight.copy()
                regw[0, 0] = 0.0
                gradientb += (c_lambda / N) * regw
                weight = weight - (learning_rate * gradientb)
                lossPerStep.append(self.rmse(np.dot(xtrain, weight), ytrain))
            # end for
        # end for
        return weight, lossPerStep
    # end ridge_fit_MBGD
    
    def ridge_cross_validation(
        self, X: np.ndarray, y: np.ndarray, kfold: int = 5, c_lambda: float = 100
    ) -> List[float]:
        """
        For each of the k-folds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the RMSE for each fold

        Args:
            X : (N,1+D) numpy array, where N is the number of instances
                and D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            kfold: int, number of folds you should take while implementing cross validation.
            c_lambda: float, value of regularization constant
        Returns:
            loss_per_fold: list[float], RMSE loss for each kfold
        Hints:
            - np.concatenate might be helpful.
            - Use ridge_fit_closed for this function.
            - Look at 3.5 to see how this function is being used.
            - For kfold=5:
                split X and y into 5 equal-size folds
                use 80 percent for training and 20 percent for test
        """
        N = X.shape[0]
        foldSize = N // kfold
        losses: List[float] = []

        for k in range(kfold):
            start = k * foldSize
            end = start + foldSize
            Xtest = X[start:end, :]
            ytest = y[start:end, :]
            Xtrain = np.concatenate((X[:start, :], X[end:, :]), axis=0)
            ytrain = np.concatenate((y[:start, :], y[end:, :]), axis=0)

            weight = self.ridge_fit_closed(Xtrain, ytrain, c_lambda)
            preds = self.predict(Xtest, weight)
            losses.append(self.rmse(preds, ytest))
        # end for
        return losses
    # end ridge_cross_validation
    
    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        FUNCTION PROVIDED TO STUDENTS

        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N, 1+D) numpy array, where N is the number of instances and
                D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants (lambdas) to search from
            kfold: int, Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the average RMSE error achieved using the best_lambda
            error_list: list[float] list of average RMSE loss for each lambda value given in lambda_list
        """
        best_error = None
        best_lambda = None
        error_list = []
        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            mean_err = np.mean(err)
            error_list.append(mean_err)
            if best_error is None or mean_err < best_error:
                best_error = mean_err
                best_lambda = lm
        return best_lambda, best_error, error_list
