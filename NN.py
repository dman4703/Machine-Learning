import numpy as np

"""
We are going to use the California housing dataset provided by sklearn
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
to train a 2-layer fully connected neural net. We are going to build the neural network from scratch.
"""


class NeuralNet:
    def __init__(
        self,
        y,
        use_dropout,
        use_adam,
        lr=0.01,
        batch_size=64,
        dropout_prob=0.3,
    ):
        """
        This method initializes the class, it is implemented for you.
        Args:
                y (np.ndarray): labels
                use_dropout (bool): flag to enable dropout
                use_adam (bool): flag to use adam
                lr (float): learning rate
                batch_size (int): batch size to use for training
                dropout_prob (float): dropout probability
        """
        self.y = y  # ground truth labels

        # OTHER HYPERPARAMTERS
        self.y_hat = np.zeros((self.y.shape[0], 3))  # estimated labels
        self.dimensions = [8, 15, 7, 3]  # dimensions of different layers
        self.alpha = 0.05
        self.eps = 1e-8

        # DROPOUT
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob

        # PARAMETERS
        self.parameters = {}  # dictionary for different layer variables
        self.cache = (
            {}
        )  # cache for holding variables during forward propagation to use them in back prop
        self.loss = []  # list to store loss values
        self.batch_y = []  # list of y batched numpy arrays

        # TRAINING HYPERPARAMETERS
        self.batch_size = batch_size  # batch size

        # NEURAL NETWORK INFORMATION
        self.learning_rate = lr  # learning rate
        self.sample_count = self.y.shape[0]  # number of training samples we have
        self._estimator_type = "regression"
        self.neural_net_type = "Softsign -> Softsign -> Softmax"

        # Adam parameters
        self.use_adam = use_adam
        self.t = 1
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.first_moment = {}
        self.second_moment = {}

    def init_parameters(self, param=None):
        """
        This method initializes the neural network variables, it is already implemented for you.
        Check it and relate to the mathematical description above.
        You are going to use these variables in forward and backward propagation.

        Args:
                param (dict): Optional dictionary of parameters to use instead of initializing.
        """
        if param is None:
            np.random.seed(0)
            self.parameters["theta1"] = np.random.randn(
                self.dimensions[0], self.dimensions[1]
            ) / np.sqrt(self.dimensions[0])
            self.parameters["b1"] = np.zeros((self.dimensions[1]))
            self.parameters["theta2"] = np.random.randn(
                self.dimensions[1], self.dimensions[2]
            ) / np.sqrt(self.dimensions[1])
            self.parameters["b2"] = np.zeros((self.dimensions[2]))
            self.parameters["theta3"] = np.random.randn(
                self.dimensions[2], self.dimensions[3]
            ) / np.sqrt(self.dimensions[2])
            self.parameters["b3"] = np.zeros((self.dimensions[3]))

        else:
            self.parameters = param
            self.parameters["theta1"] = self.parameters["theta1"]
            self.parameters["theta2"] = self.parameters["theta2"]
            self.parameters["theta3"] = self.parameters["theta3"]
            self.parameters["b1"] = self.parameters["b1"]
            self.parameters["b2"] = self.parameters["b2"]
            self.parameters["b3"] = self.parameters["b3"]

        for parameter in self.parameters:
            self.first_moment[parameter] = np.zeros_like(self.parameters[parameter])
            self.second_moment[parameter] = np.zeros_like(self.parameters[parameter])

    def softmax(self, u):
        """
        Performs softmax function function element-wise.
        To prevent overflow, begin by subtracting each row in u by its maximum!
        Input:
                u (np.ndarray: (N, D)): logits
        Output:
                o (np.ndarray: (N, D)): N probability distributions over D classes
        """
        shiftedU = u - np.max(u, axis=1, keepdims=True)
        expScores = np.exp(shiftedU)
        sumExpScores = np.sum(expScores, axis=1, keepdims=True)
        return expScores / sumExpScores
    # end softmax

    def softsign(self, u):
        """
        The Softsign activation function.
        It is defined as: Softsign(x) = x / (1 + abs(x)).

        Args:
            u (np.ndarray): input array with any shape.

        Returns:
           o (np.ndarray): output, same shape as input u.
        """
        return u / (1 + np.abs(u))
    # end softsign

    def derivative_softsign(self, x):
        """
        Derivative of the Softsign activation function.

        Args:
                x (np.ndarray): Input array.

        Returns:
                np.ndarray: Derivative of Softsign.
        """
        return 1.0 / (1.0 + np.abs(x)) ** 2
    # end derivative_softsign

    @staticmethod
    def _dropout(u, prob):
        """
        Implement the dropout layer. Refer to the description for implementation details.
        Args:
                u (np.ndarray: (N, D)): input to dropout layer
                prob: the probability of dropping an unit
        Returns:
                u_after_dropout (np.ndarray: (N, D)): output of dropout layer
                dropout_mask (np.ndarray: (N, D)): dropout mask indicating which units were dropped

        Hint: scale the units after dropout
                  use np.random.choice to sample from Bernoulli(prob) the inactivated nodes for each iteration
        """
        if prob <= 0:
            dropoutMask = np.ones_like(u)
            return u, dropoutMask
        # end if
        dropoutMask = np.random.choice([0, 1], size=u.shape, p=[prob, 1 - prob])
        uAfterDropout = (u * dropoutMask) / (1.0 - prob)
        return uAfterDropout, dropoutMask
    # end _dropout

    def cross_entropy_loss(self, y, y_hat):
        """
        Computes cross entropy loss.
        Refer to the description in the notebook and implement the appropriate mathematical equation.
        To avoid log(0) errors, add a small constant 1e-15 to the input to np.log
        Args:
                y (np.ndarray: (N, D)): one-hot ground truth labels
                y_hat (np.ndarray: (N, D)): predictions
        Returns:
                loss (float): average cross entropy loss
        """
        smallConstant = 1e-15
        logPredictions = np.log(y_hat + smallConstant)
        return -np.sum(y * logPredictions) / y.shape[0]
    # end cross_entropy_loss

    def forward(self, x, use_dropout):
        """
        We have provided an outline of the function you can use below. Fill in the "..." with the appropriate code.
        Check init_parameters method and use variables from there as well as other implemented methods.
        Implement the appropriate mathematical equations as described in the notebook.
        NOTE: Implement dropout only on the first layer!

        Args:
                x (np.ndarray: (N, M)): input to neural network
                use_dropout (bool): true if using dropout in forward.
        Returns:
                o3 (np.ndarray: (N, D)): output of neural network

        Note: The shapes of the variables in self.cache should be:
                u1 (np.ndarray: (N, K)): output after first linear layer
                o1 (np.ndarray: (N, K)): output after applying activation and dropout (if specified)
                u2 (np.ndarray: (N, R)): output after second linear layer
                o2 (np.ndarray: (N, R)): output of nueral network
                u3 (np.ndarray: (N, D)): output after third linear layer
                o3 (np.ndarray: (N, D)): output of nueral network

        where,
                N: Number of datapoints
                M: Number of input features
                K: Size of the first hidden layer
                R: Size of the second hidden layer
                D: Number of output features

        HINT 1: Refer to this guide: https://static.us.edusercontent.com/files/gznuqr6aWHD8dPhiusG2TG53 for more detail on the forward pass.

        Outline:

        self.cache["X"] = x
        u1 = ...
        o1 = ...

        if use_dropout:
                o1, dropout_mask = ...
                self.cache["mask"] = dropout_mask

        self.cache["u1"], self.cache["o1"] = u1, o1

        u2 = ...
        o2 = ...
        self.cache["u2"], self.cache["o2"] = u2, o2

        u3 = ...
        o3 = ...
        self.cache["u3"], self.cache["o3"] = u3, o3

        return o3
        """
        self.cache["X"] = x

        theta1 = self.parameters["theta1"]
        b1 = self.parameters["b1"]
        theta2 = self.parameters["theta2"]
        b2 = self.parameters["b2"]
        theta3 = self.parameters["theta3"]
        b3 = self.parameters["b3"]

        u1 = np.dot(x, theta1) + b1
        o1 = self.softsign(u1)

        if use_dropout:
            o1, dropoutMask = self._dropout(o1, self.dropout_prob)
            self.cache["mask"] = dropoutMask
        # end if

        self.cache["u1"], self.cache["o1"] = u1, o1

        u2 = np.dot(o1, theta2) + b2
        o2 = self.softsign(u2)
        self.cache["u2"], self.cache["o2"] = u2, o2

        u3 = np.dot(o2, theta3) + b3
        o3 = self.softmax(u3)
        self.cache["u3"], self.cache["o3"] = u3, o3

        return o3
    # end forward

    def compute_gradients(self, y, yh):
        """
        Compute the gradients for each layer given the predicted outputs and ground truths.
        The dropout mask you stored at forward may be helpful.

        Args:
            y (np.ndarray: (N, D)): ground truth values
            yh (np.ndarray: (N, D)): predicted outputs

        Returns:
            gradients (dict): dictionary with the following mapping and shapes:
                dLoss_theta3 (np.ndarray: (R, D)): gradients for theta3
                dLoss_b3 (np.ndarray: (D,)): gradients for b3
                dLoss_theta2 (np.ndarray: (K, R)): gradients for theta2
                dLoss_b2 (np.ndarray: (R,)): gradients for b2
                dLoss_theta1 (np.ndarray: (M, K)): gradients for theta1
                dLoss_b1 (np.ndarray: (K,)): gradients for b1
                where,
                N: Number of datapoints
                M: Number of input features
                K: Size of the first hidden layer
                R: Size of the second hidden layer
                D: Number of output features


        Note: You will have to use the cache (self.cache) to retrieve the values
        from the forward pass!

        HINT 1: Refer to this guide: https://static.us.edusercontent.com/files/gznuqr6aWHD8dPhiusG2TG53 for more detail on computing gradients.

        HINT 2: Division by N only needs to occur ONCE for any derivative that requires a division
        by N. Make sure you avoid cascading divisions by N where you might accidentally divide your
        derivative by N^2 or greater.

        HINT 3: Here's an outline of the function you can use. Fill in the "..." with the appropriate code:

        Note that we only implemented drop out on the first hidden layer.

        dLoss_u3 = yh - y

        dLoss_theta3 = ...
        dLoss_b3 = ...
        dLoss_o2 = ...


        dLoss_u2 = ...

        dLoss_theta2 = ...
        dLoss_b2 = ...
        dLoss_o1 = ...

        if self.use_dropout:
                dLoss_u1 = ...
        else:
                dLoss_u1 = ...

        dLoss_theta1 = ...
        dLoss_b1 = ...

        """
        numSamples = y.shape[0]

        theta1 = self.parameters["theta1"]
        theta2 = self.parameters["theta2"]
        theta3 = self.parameters["theta3"]

        X = self.cache["X"]
        u1 = self.cache["u1"]
        o1 = self.cache["o1"]
        u2 = self.cache["u2"]
        o2 = self.cache["o2"]

        dLoss_u3 = yh - y

        dLossTheta3 = np.dot(o2.T, dLoss_u3) / numSamples
        dLossB3 = np.sum(dLoss_u3, axis=0) / numSamples
        dLoss_o2 = np.dot(dLoss_u3, theta3.T)

        dLoss_u2 = dLoss_o2 * self.derivative_softsign(u2)

        dLossTheta2 = np.dot(o1.T, dLoss_u2) / numSamples
        dLossB2 = np.sum(dLoss_u2, axis=0) / numSamples
        dLoss_o1 = np.dot(dLoss_u2, theta2.T)

        if self.use_dropout:
            dropoutMask = self.cache["mask"]
            dLoss_u1 = (
                dLoss_o1
                * self.derivative_softsign(u1)
                * dropoutMask
                / (1.0 - self.dropout_prob)
            )
        else:
            dLoss_u1 = dLoss_o1 * self.derivative_softsign(u1)
        # end if
        
        dLossTheta1 = np.dot(X.T, dLoss_u1) / numSamples
        dLossB1 = np.sum(dLoss_u1, axis=0) / numSamples

        gradients = {
            "dLoss_theta3": dLossTheta3,
            "dLoss_b3": dLossB3,
            "dLoss_theta2": dLossTheta2,
            "dLoss_b2": dLossB2,
            "dLoss_theta1": dLossTheta1,
            "dLoss_b1": dLossB1,
        }

        return gradients
    # end compute_gradients

    def update_weights(self, dLoss):
        """
        Update the weights in the network using the gradients of the parameters (dLoss). You should update self.parameters.
        For every parameter:
        - if self.use_adam is true:
            - self.first_moment (M), self.second_moment (V), self.beta1, self.beta2, self.t have been initialized for you.
            - You can either increment self.t here (once at the end), or increment it in every iteration in gradient_descent/minibatch_gradient_descent.
            - Refer to init_parameters to understand how to index into M & V to get M_t and V_t for every parameter t.
            - Update self.first_moment and self.second_moment using dLoss
            - Normalize the first & second moments and calculate the learning update
            - Update the parameters using the learning update and learning rate
        - else:
            - Update the parameters using dLoss and learning rate

        Args:
                dLoss (dict): dictionary that maps layer names (strings) to gradients (numpy arrays)
                use_adam (bool): flag to use adam or not

        Return:
                None

        Hint: You must update self.parameters, self.first_moment and self.second_moment if use_adam = True.
        Otherwise, you only need to update self.parameters.
        """
        learningRate = self.learning_rate

        if self.use_adam:
            for paramName in ["theta3", "b3", "theta2", "b2", "theta1", "b1"]:
                if paramName in dLoss:
                    gradientParam = dLoss[paramName]
                else:
                    gradientParam = dLoss["dLoss_" + paramName]
                # end if
                firstMoment = self.first_moment[paramName]
                secondMoment = self.second_moment[paramName]

                firstMoment = self.beta1 * firstMoment + (1.0 - self.beta1) * gradientParam
                secondMoment = (self.beta2 * secondMoment + (1.0 - self.beta2) * (gradientParam ** 2))

                self.first_moment[paramName] = firstMoment
                self.second_moment[paramName] = secondMoment

                mHat = firstMoment / (1.0 - (self.beta1 ** self.t))
                vHat = secondMoment / (1.0 - (self.beta2 ** self.t))

                updateStep = learningRate * mHat / (np.sqrt(vHat) + self.eps)
                self.parameters[paramName] = (self.parameters[paramName] - updateStep)
            # end for
            self.t += 1
        else:
            # Vanilla gradient descent update
            for paramName in ["theta3", "b3", "theta2", "b2", "theta1", "b1"]:
                if paramName in dLoss:
                    gradientParam = dLoss[paramName]
                else:
                    gradientParam = dLoss["dLoss_" + paramName]
                # end if
                self.parameters[paramName] = (self.parameters[paramName] - learningRate * gradientParam)
            # end for
        # end if
    # end update_weights

    def backward(self, y, yh):
        """
        Implement a complete backward pass using compute_gradients and update_weights

        Args:
                y (np.ndarray: (N, D)): ground truth labels
                yh (np.ndarray: (N, D)): neural network predictions
                use_dropout (bool): flag to use dropout
                use_adam (bool): flag to use adam

        Note: This function does not return anything. It should update the weights in the network using the gradients.
        """
        gradients = self.compute_gradients(y, yh)
        self.update_weights(gradients)
    # end backward

    def gradient_descent(self, x, y, iter=60000, local_test=False):
        """
        This function is an implementation of the gradient descent algorithm.
        Notes:
        1. GD considers all examples in the dataset in one go and learns a gradient from them.
        2. One iteration here is one round of forward and backward propagation on the complete dataset.
        3. Append loss at multiples of 1000 i.e. at 0th, 1000th, 2000th .... iterations to self.loss
        4. Increment self.t at every iteration if you haven't done so in update_weights.
        **For LOCAL TEST append and print out loss at every iteration instead of every 1000th multiple.

        Args:
                x (np.ndarray: N x M): input
                y (np.ndarray: N x D): ground truth labels
                iter (int): number of iterations to train for
                use_adam (bool): flag to use adam or not
                local_test (bool): flag to indicate if local test is being run or not

        HINT: Here's an outline of the function you can use.

        self.init_parameters()
        for i in range(iter):
            # TODO: implement training loop

            # Print every one iteration for local test, and every 1000th iteration for AG and 1.3
            print_multiple = 1 if local_test else 1000
            if i % print_multiple == 0:
                    print("Loss after iteration %i: %f" % (i, loss))
                    self.loss.append(loss)
            self.t += 1 // if you didn't already increment self.t in update_weights
        """
        self.init_parameters()

        for i in range(iter):
            yh = self.forward(x, self.use_dropout)
            self.y_hat = yh

            loss = self.cross_entropy_loss(y, yh)
            self.backward(y, yh)

            printMultiple = 1 if local_test else 1000
            if i % printMultiple == 0:
                print("Loss after iteration %i: %f" % (i, loss))
                self.loss.append(loss)
            # end if
        # end for
    # end gradient_descent

    def minibatch_gradient_descent(self, x, y, iter=60000, local_test=False):
        """
        This function is an implementation of the batch gradient descent algorithm

        Notes:
        1. MBGD loops over all mini batches in the dataset one by one and learns a gradient
        2. One iteration here is one round of forward and backward propagation on one minibatch.
           You will use self.batch_size to index into x and y to get a batch. This batch will be
           fed into the forward and backward functions.

        3. Append and printout loss at multiples of 1000 iterations i.e. at 0th, 1000th, 2000th .... iterations.
           **For LOCAL TEST append and print out loss at every iteration instead of every 1000th multiple.

        4. Append the y batched numpy array to self.batch_y at every 1000 iterations i.e. at 0th, 1000th,
           2000th .... iterations. We will use this to determine if batching is done correctly.
           **For LOCAL TEST append the y batched array at every iteration instead of every 1000th multiple

        5. We expect a noisy plot since learning on a batch adds variance to the
           gradients learnt
        6. Be sure that your batch size remains constant (see notebook for more detail). Please
           batch your data in a wraparound manner. For example, given a dataset of 9 numbers,
           [1, 2, 3, 4, 5, 6, 7, 8, 9], and a batch size of 6, the first iteration batch will
           be [1, 2, 3, 4, 5, 6], the second iteration batch will be [7, 8, 9, 1, 2, 3],
           the third iteration batch will be [4, 5, 6, 7, 8, 9], etc...
        7. Increment self.t at every iteration if you haven't done so in update_weights.

        Args:
                x (np.ndarray: N x M): input data
                y (np.ndarray: N x D): ground truth labels
                use_adam (bool): flag to use adam or not
                iter (int): number of BATCHES to iterate through
                local_test (bool): True if calling local test, default False for autograder and Q1.3
                                this variable can be used to switch between autograder and local test requirement for
                                appending/printing out loss and y batch arrays
        HINT:  Here's an outline of the function you can use.

        self.init_parameters()
        for i in range(iter):
            # TODO: implement training loop

            # Print every one iteration for local test, and every 1000th iteration for AG and 1.3
            print_multiple = 1 if local_test else 1000
            if i % print_multiple == 0:
                    print("Loss after iteration %i: %f" % (i, loss))
                    self.loss.append(loss)
                    self.batch_y.append(y_batch)
            self.t += 1 // if you didn't increment self.t in update_weights
        """
        self.init_parameters()

        sampleCount = x.shape[0]
        batchSize = self.batch_size

        for i in range(iter):
            batchIndices = (
                np.arange(i * batchSize, i * batchSize + batchSize) % sampleCount
            )

            xBatch = x[batchIndices]
            yBatch = y[batchIndices]

            yhBatch = self.forward(xBatch, self.use_dropout)
            loss = self.cross_entropy_loss(yBatch, yhBatch)

            self.backward(yBatch, yhBatch)

            printMultiple = 1 if local_test else 1000
            if i % printMultiple == 0:
                print("Loss after iteration %i: %f" % (i, loss))
                self.loss.append(loss)
                self.batch_y.append(yBatch)
            # end if
        # end for
    # end minibatch_gradient_descent

    def predict(self, x):
        """
        This function predicts new data points
        It is implemented for you

        Args:
                x (np.ndarray: (N, M)): input data
        Returns:
                y (np.ndarray: (N,)): predictions
        """
        yh = self.forward(x, False)  # (N, D) = forward((N, M))
        pred = np.argmax(yh, axis=1)  # (N,) = argmax((N, D), axis=1)
        return pred
