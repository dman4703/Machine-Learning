import pickle
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from cnn import CNN
from cnn_image_transformations import (
    TransformedDataset,
    create_testing_transformations,
    create_training_transformations,
)
from lstm import LSTM, LSTMOutputAdapter
from NN import NeuralNet as dlnet
from rnn import RNN, RNNOutputAdapter
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC
from svm import kernel_construction, rbf_kernel
from utilities.utils import get_housing_dataset


class TestNN(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.setUp()

        # sample training data
        self.x_train = np.array(
            [
                [
                    0.72176308,
                    0.43961601,
                    0.13553666,
                    0.55713544,
                    0.87702101,
                    0.12019972,
                    0.04842653,
                    0.01573553,
                ],
                [
                    0.53498397,
                    0.81056978,
                    0.17524362,
                    0.00521916,
                    0.2053607,
                    0.90502607,
                    0.99638276,
                    0.45936163,
                ],
                [
                    0.84114195,
                    0.78107371,
                    0.62526833,
                    0.18139081,
                    0.28554493,
                    0.86342263,
                    0.11350829,
                    0.82592072,
                ],
                [
                    0.43286995,
                    0.13815595,
                    0.71456809,
                    0.985452,
                    0.60177364,
                    0.87152055,
                    0.85442663,
                    0.7442592,
                ],
                [
                    0.54714474,
                    0.45039175,
                    0.43588923,
                    0.53943311,
                    0.70734352,
                    0.67388256,
                    0.29136773,
                    0.19560766,
                ],
                [
                    0.5617591,
                    0.86315884,
                    0.34730499,
                    0.13892525,
                    0.53279486,
                    0.79825459,
                    0.37465092,
                    0.23443029,
                ],
                [
                    0.4233198,
                    0.0020612,
                    0.4777035,
                    0.78088463,
                    0.8208675,
                    0.76655747,
                    0.72102559,
                    0.79251294,
                ],
                [
                    0.74503529,
                    0.25137268,
                    0.76440309,
                    0.5790357,
                    0.03791042,
                    0.82510481,
                    0.64463256,
                    0.08997057,
                ],
                [
                    0.81644094,
                    0.51437913,
                    0.75881908,
                    0.96191336,
                    0.56525617,
                    0.70372399,
                    0.75134392,
                    0.56722149,
                ],
            ]
        )
        self.y_train = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]).T

    def setUp(self):
        self.nn = dlnet(y=np.random.randn(1, 30), use_dropout=False, use_adam=False)

    def assertAllClose(self, student, truth, msg=None):
        self.assertTrue(np.allclose(student, truth), msg=msg)

    def assertDictAllClose(self, student, truth):
        for key in truth:
            if key not in student:
                self.fail("Key " + key + " missing.")
            self.assertAllClose(student[key], truth[key], msg=(key + " is incorrect."))

        for key in student:
            if key not in truth:
                self.fail("Extra key " + key + ".")

    def test_softsign(self):
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        u_copy = u.copy()
        student = self.nn.softsign(u)

        truth = np.array(
            [
                [
                    0.63821235,
                    0.28579449,
                    0.49462738,
                    0.69144309,
                    0.65127122,
                    -0.49425419,
                ],
                [
                    0.48720274,
                    -0.13145982,
                    -0.09356154,
                    0.29108106,
                    0.12590742,
                    0.59254745,
                ],
                [
                    0.432153,
                    0.10847618,
                    0.30741363,
                    0.25019176,
                    0.5990504,
                    -0.17023346,
                ],
                [
                    0.23842464,
                    -0.46065353,
                    -0.7185469,
                    0.39526563,
                    0.46364483,
                    -0.42600156,
                ],
                [
                    0.69416665,
                    -0.59256275,
                    0.0437563,
                    -0.15767048,
                    0.6051768,
                    0.59503657,
                ],
            ]
        )
        self.assertAllClose(student, truth)
        self.assertAllClose(u, u_copy)
        print_success_message("test_softsign")

    def test_d_softsign(self):
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        student = self.nn.derivative_softsign(u)
        truth = np.array(
            [
                [
                    0.1308903,
                    0.51008952,
                    0.25540149,
                    0.09520737,
                    0.12161176,
                    0.25577882,
                ],
                [
                    0.26296103,
                    0.75436205,
                    0.82163069,
                    0.50256607,
                    0.76403784,
                    0.16601758,
                ],
                [
                    0.32245022,
                    0.79481472,
                    0.47967589,
                    0.56221239,
                    0.16076058,
                    0.68851251,
                ],
                [
                    0.57999702,
                    0.29089462,
                    0.07921585,
                    0.36570366,
                    0.28767687,
                    0.32947421,
                ],
                [
                    0.09353404,
                    0.16600511,
                    0.91440202,
                    0.70951902,
                    0.15588536,
                    0.16399538,
                ],
            ]
        )
        self.assertAllClose(student, truth)
        print_success_message("test_d_softsign")

    def test_silu(self):
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        u_copy = u.copy()
        student = self.nn.silu(u)

        truth = np.array(
            [
                [
                    1.50600053,
                    0.2395843,
                    0.71140327,
                    2.02545844,
                    1.61763291,
                    -0.26721929,
                ],
                [
                    0.68514007,
                    -0.06996226,
                    -0.04894825,
                    0.2468647,
                    0.07719997,
                    1.17891447,
                ],
                [
                    0.51870733,
                    0.06453415,
                    0.27039224,
                    0.19441639,
                    1.22019903,
                    -0.0920934,
                ],
                [
                    0.18083851,
                    -0.25501111,
                    -0.18439194,
                    0.42996694,
                    0.60820579,
                    -0.23937115,
                ],
                [
                    2.05717158,
                    -0.27535592,
                    0.02340263,
                    -0.08485796,
                    1.26057691,
                    1.19452976,
                ],
            ]
        )
        self.assertAllClose(student, truth)
        self.assertAllClose(u, u_copy)
        print_success_message("test_silu")

    def test_d_silu(self):
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        student = self.nn.derivative_silu(u)
        truth = np.array(
            [
                [
                    1.07401955,
                    0.69486452,
                    0.92117203,
                    1.09858542,
                    1.08265444,
                    0.07927933,
                ],
                [0.91219594, 0.42460936, 0.44848207, 0.69967328, 0.5717735, 1.03387651],
                [0.8467463, 0.56068773, 0.71485409, 0.6637922, 1.04036474, 0.39813594],
                [
                    0.65401393,
                    0.11970306,
                    -0.09884819,
                    0.80494919,
                    0.88386697,
                    0.16036434,
                ],
                [
                    1.09901368,
                    -0.03389202,
                    0.52287128,
                    0.40695176,
                    1.04627518,
                    1.03638487,
                ],
            ]
        )
        self.assertAllClose(student, truth)
        print_success_message("test_d_silu")

    def test_leaky_relu(self):
        alpha = 0.05
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        u_copy = u.copy()
        student = self.nn.leaky_relu(alpha, u)
        truth = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.04886389,
                ],
                [
                    0.95008842,
                    -0.00756786,
                    -0.00516094,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.01025791,
                ],
                [
                    0.3130677,
                    -0.04270479,
                    -0.12764949,
                    0.6536186,
                    0.8644362,
                    -0.03710825,
                ],
                [
                    2.26975462,
                    -0.07271828,
                    0.04575852,
                    -0.00935919,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        self.assertAllClose(student, truth)
        self.assertAllClose(u, u_copy)
        print_success_message("test_leaky_relu")

    def test_softmax(self):
        input = np.array([[2, 0, 1], [1, 0, 2]])

        actual = self.nn.softmax(input)

        expected = np.array(
            [[0.66524096, 0.09003057, 0.24472847], [0.24472847, 0.09003057, 0.66524096]]
        )

        assert np.allclose(actual, expected, atol=0.1)
        print_success_message("test_softmax")

    def test_d_leaky_relu(self):
        alpha = 0.05
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        student = self.nn.derivative_leaky_relu(alpha, u)
        truth = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.05],
                [1.0, 0.05, 0.05, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.05],
                [1.0, 0.05, 0.05, 1.0, 1.0, 0.05],
                [1.0, 0.05, 1.0, 0.05, 1.0, 1.0],
            ]
        )
        self.assertAllClose(student, truth)
        print_success_message("test_d_leaky_relu")

    def test_softmax(self):
        input = np.array([[2, 0, 1], [1, 0, 2]])

        actual = self.nn.softmax(input)

        expected = np.array(
            [[0.66524096, 0.09003057, 0.24472847], [0.24472847, 0.09003057, 0.66524096]]
        )

        assert np.allclose(actual, expected, atol=0.1)
        print_success_message("test_softmax")

    def test_d_silu(self):
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        student = self.nn.derivative_silu(u)
        truth = np.array(
            [
                [
                    1.07401955,
                    0.69486452,
                    0.92117203,
                    1.09858542,
                    1.08265444,
                    0.07927933,
                ],
                [0.91219594, 0.42460936, 0.44848207, 0.69967328, 0.5717735, 1.03387651],
                [0.8467463, 0.56068773, 0.71485409, 0.6637922, 1.04036474, 0.39813594],
                [
                    0.65401393,
                    0.11970306,
                    -0.09884819,
                    0.80494919,
                    0.88386697,
                    0.16036434,
                ],
                [
                    1.09901368,
                    -0.03389202,
                    0.52287128,
                    0.40695176,
                    1.04627518,
                    1.03638487,
                ],
            ]
        )
        self.assertAllClose(student, truth)
        print_success_message("test_d_silu")

    def test_dropout(self):
        np.random.seed(0)
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        student, _ = self.nn._dropout(u, prob=0.3)

        truth = np.array(
            [
                [2.52007479, 0.57165316, 1.39819711, 3.201276, 2.66793999, -1.39611126],
                [
                    1.35726917,
                    -0.21622459,
                    -0.1474555,
                    0.58656929,
                    0.20577653,
                    2.07753359,
                ],
                [1.08719676, 0.17382146, 0.0, 0.0, 0.0, -0.29308323],
                [
                    0.44723957,
                    -1.22013677,
                    -3.64712831,
                    0.93374086,
                    1.23490886,
                    -1.06023574,
                ],
                [0.0, -2.07766524, 0.0, -0.2674055, 2.18968459, 2.09908396],
            ]
        )

        self.assertAllClose(student, truth)
        print_success_message("test_dropout")

    def test_loss(self):
        y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])

        # Model's predicted probabilities for each class
        yh = np.array(
            [
                [0.8, 0.15, 0.05],
                [0.1, 0.7, 0.2],
                [0.05, 0.1, 0.85],
                [0.9, 0.05, 0.05],
                [0.1, 0.3, 0.6],
            ]
        )

        # Calculate Cross-Entropy
        student = self.nn.cross_entropy_loss(y, yh)

        truth = 0.2717047128349055

        self.assertAllClose(student, truth)
        print_success_message("test_loss")

    def test_forward_without_dropout(self):
        # load nn parameters
        file = open("data/test_data/nn_param_new.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(file))
        file.close()

        np.random.seed(42)

        x = np.random.rand(10, 3)

        student = self.nn.forward(x, use_dropout=False)

        truth = np.array(
            [
                [0.36450874, 0.33744198, 0.29804928],
                [0.40793771, 0.31060955, 0.28145274],
                [0.32367739, 0.35412677, 0.32219584],
                [0.41217376, 0.30437785, 0.28344839],
                [0.41617887, 0.30771078, 0.27611035],
                [0.36647134, 0.32538377, 0.30814489],
                [0.39351613, 0.3150166, 0.29146727],
                [0.35722217, 0.32982926, 0.31294857],
                [0.37627323, 0.333526, 0.29020077],
                [0.38936348, 0.3262192, 0.28441732],
            ]
        )
        self.assertAllClose(student, truth)
        print_success_message("test_forward_without_dropout")

    def test_forward_with_dropout(self):
        # control random seed
        np.random.seed(0)

        # load nn parameters
        file = open("data/test_data/nn_param_new.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(file))
        file.close()

        np.random.seed(42)

        x = np.random.rand(10, 3)

        student = self.nn.forward(x, use_dropout=True)

        truth = np.array(
            [
                [0.38083144, 0.37202949, 0.24713907],
                [0.42249378, 0.33856044, 0.23894578],
                [0.34116015, 0.34053483, 0.31830502],
                [0.42545811, 0.24746995, 0.32707194],
                [0.43159953, 0.30505192, 0.26334855],
                [0.37145432, 0.3548629, 0.27368278],
                [0.40436912, 0.25965551, 0.33597537],
                [0.35755991, 0.3236327, 0.31880739],
                [0.39935786, 0.35870271, 0.24193942],
                [0.39165996, 0.30246166, 0.30587838],
            ]
        )
        self.assertAllClose(student, truth)
        print_success_message("test_forward")

    def test_compute_gradients_without_dropout(self):
        # Load updated parameters
        nn_param = open("data/test_data/nn_param_new.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        # Load updated cache
        cache = open("data/test_data/test_compute_gradients_cache_new.pickle", "rb")
        self.nn.cache = pickle.load(cache)
        cache.close()

        # Transpose cache data as needed
        for p in self.nn.cache:
            self.nn.cache[p] = self.nn.cache[p]

        # Update y and yh to new dimensions (match the network architecture)
        y = np.array(
            [
                [1, 0, 0],  # Class 0 for Sample 1
                [0, 1, 0],  # Class 1 for Sample 2
                [0, 0, 1],  # Class 2 for Sample 3
                [1, 0, 0],  # Class 0 for Sample 4
                [0, 1, 0],  # Class 1 for Sample 5
                [1, 0, 0],  # Class 0 for Sample 6
                [0, 0, 1],  # Class 2 for Sample 7
                [0, 1, 0],  # Class 1 for Sample 8
                [1, 0, 0],  # Class 0 for Sample 9
                [0, 0, 1],  # Class 2 for Sample 10
            ]
        )

        yh = np.array(
            [
                [0.45114982, 0.27534103, 0.27350915],
                [0.50694454, 0.25828013, 0.23477533],
                [0.376928, 0.31075548, 0.31231653],
                [0.64020962, 0.17592685, 0.18386353],
                [0.60049779, 0.21113381, 0.1883684],
                [0.38496115, 0.30412801, 0.31091084],
                [0.47722575, 0.26188546, 0.26088879],
                [0.36382012, 0.31778386, 0.31839602],
                [0.42644384, 0.30231422, 0.27124194],
                [0.44405581, 0.29595013, 0.25999406],
            ]
        )  # (10,3)

        # Compute the gradients without dropout
        student = self.nn.compute_gradients(y, yh)
        # print(student)
        # print("theta3 gradient shape:", student["theta3"].shape)

        # Update the expected truth values for theta1, theta2, theta3, and biases
        truth = {
            "dLoss_theta1": np.array(
                [
                    [0.00895215, 0.00766287, 0.00082727, -0.00719649, 0.00384498],
                    [0.00184395, 0.0037988, 0.01529954, -0.00751128, -0.00906929],
                    [0.00608964, 0.00585077, 0.00106683, -0.00675632, 0.0049104],
                ]
            ),
            "dLoss_b1": np.array(
                [0.01865414, 0.01262648, 0.0118904, -0.01248499, 0.00588595]
            ),
            "dLoss_theta2": np.array(
                [
                    [0.00014873, 0.021338, 0.01161551, 0.0036835, -0.00072898],
                    [0.00093171, 0.04741728, 0.01448796, 0.01261177, 0.00288854],
                    [0.00274418, 0.03154303, 0.01204491, 0.01167746, 0.00169214],
                    [0.00149538, 0.04055873, -0.01106509, 0.01588332, 0.00814872],
                    [-0.00183606, 0.03907579, 0.00040928, 0.02024615, 0.00599554],
                ]
            ),
            "dLoss_b2": np.array(
                [-0.00025202, 0.05614122, 0.01991378, 0.01096406, 0.00131081]
            ),
            "dLoss_theta3": np.array(
                [
                    [0.11512291, -0.0211416, -0.09398131],
                    [-0.00472817, -0.01578938, 0.02051755],
                    [0.03034595, -0.02774979, -0.00259616],
                    [0.0655147, 0.00383751, -0.06935222],
                    [-0.02385389, 0.02962021, -0.00576631],
                ]
            ),
            "dLoss_b3": np.array([0.06722364, -0.0286501, -0.03857354]),
        }
        # Compare the computed gradients with the truth
        self.assertDictAllClose(student, truth)
        print_success_message("test_compute_gradients_without_dropout")

    def test_compute_gradients_with_dropout(self):
        nn = dlnet(y=np.random.randn(1, 30), use_dropout=True, use_adam=False)
        nn_param = open("data/test_data/nn_param_new.pickle", "rb")
        nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        cache = open("data/test_data/test_compute_gradients_cache_new.pickle", "rb")
        nn.cache = pickle.load(cache)
        cache.close()

        for p in nn.cache:
            nn.cache[p] = nn.cache[p]

        y = np.array(
            [
                [1, 0, 0],  # Class 0 for Sample 1
                [0, 1, 0],  # Class 1 for Sample 2
                [0, 0, 1],  # Class 2 for Sample 3
                [1, 0, 0],  # Class 0 for Sample 4
                [0, 1, 0],  # Class 1 for Sample 5
                [1, 0, 0],  # Class 0 for Sample 6
                [0, 0, 1],  # Class 2 for Sample 7
                [0, 1, 0],  # Class 1 for Sample 8
                [1, 0, 0],  # Class 0 for Sample 9
                [0, 0, 1],  # Class 2 for Sample 10
            ]
        )
        yh = np.array(
            [
                [0.44241676, 0.33622821, 0.22135503],
                [0.60897122, 0.22194954, 0.16907924],
                [0.36570828, 0.2890999, 0.34519182],
                [0.62958633, 0.12340082, 0.24701284],
                [0.74252438, 0.13642363, 0.12105198],
                [0.37250028, 0.35025135, 0.27724837],
                [0.41267473, 0.2645114, 0.32281387],
                [0.35629951, 0.3272992, 0.31640129],
                [0.44909499, 0.32278295, 0.22812206],
                [0.41330537, 0.28034051, 0.30635412],
            ]
        )
        student = nn.compute_gradients(y, yh)
        # print(student)

        truth = {
            "dLoss_theta1": np.array(
                [
                    [0.01257918, 0.00438516, 0.00041564, -0.00033564, -0.00586452],
                    [0.00369215, 0.0012352, 0.01411022, -0.00648098, -0.00873815],
                    [0.00435854, 0.00428483, 0.00087675, -0.00299805, -0.00618496],
                ]
            ),
            "dLoss_b1": np.array(
                [0.02841737, 0.00543574, 0.01463225, -0.00314271, -0.00901249]
            ),
            "dLoss_theta2": np.array(
                [
                    [0.00117768, 0.02191615, 0.01140168, 0.00332251, -0.00193981],
                    [0.00095694, 0.04510351, 0.01295194, 0.01295712, 0.0021269],
                    [0.00352138, 0.03066893, 0.01248627, 0.01129413, 0.00045637],
                    [0.00320332, 0.04393486, -0.0147236, 0.01645948, 0.00765219],
                    [-0.00091297, 0.03927441, 0.00206941, 0.01807345, 0.00367213],
                ]
            ),
            "dLoss_b2": np.array(
                [0.0012974, 0.05698888, 0.01684716, 0.01104701, -0.00039955]
            ),
            "dLoss_theta3": np.array(
                [
                    [0.11615164, -0.02704535, -0.08910628],
                    [0.00698212, -0.01659459, 0.00961246],
                    [0.03556941, -0.02808471, -0.0074847],
                    [0.0702694, 0.00220021, -0.07246961],
                    [-0.02613038, 0.0342784, -0.00814802],
                ]
            ),
            "dLoss_b3": np.array([0.07930819, -0.03477125, -0.04453694]),
        }

        self.assertDictAllClose(student, truth)
        print_success_message("test_compute_gradients_with_dropout")

    def test_update_weights(self):
        nn_param = open("data/test_data/nn_param_new.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        dLoss_file = open("data/test_data/dLoss_new.pickle", "rb")
        dLoss = pickle.load(dLoss_file)
        dLoss_file.close()

        # for p in dLoss:
        #     dLoss[p] = dLoss[p].T

        self.nn.update_weights(dLoss)
        student = self.nn.parameters

        theta1 = np.array(
            [
                [1.00083557, 0.2270293, 0.55528726, 1.27137136, 1.05955953],
                [-0.55445887, 0.53903292, -0.08587255, -0.05856124, 0.23295317],
                [0.08172316, 0.82508247, 0.43177496, 0.06903235, 0.25182592],
            ]
        )

        b1 = np.array([-0.00333674, -0.01494079, 0.00205158, -0.00313068, 0.00854096]).T

        theta2 = np.array(
            [
                [0.17475359, 0.66163629, -0.10039393, 0.14742978, -0.40466077],
                [-1.1271881, 0.29184954, 0.38845946, -0.34723408, 1.00037154],
                [-0.65196158, 0.01668221, -0.07483331, 0.70528767, 0.66059634],
                [0.06773111, 0.15681651, -0.40905365, -0.88196584, -0.15256802],
                [0.08040691, 0.5644029, 0.55478332, -0.19272557, -0.13009738],
            ]
        )

        b2 = np.array([0.00438074, 0.01252795, -0.0077749, 0.01613898, 0.0021274]).T

        theta3 = np.array(
            [
                [-0.45997248, -0.63892035, -0.75795918],
                [0.8842196, -0.22764156, -0.2001961],
                [-0.56093229, 0.34467954, -0.71541384],
                [-0.09151293, -0.39374022, 0.17662359],
                [-0.22030754, -0.51073194, -0.01437774],
            ]
        )
        b3 = np.array([0.00401781, 0.01630198, -0.00462782]).T

        truth = {
            "theta1": theta1,
            "b1": b1,
            "theta2": theta2,
            "b2": b2,
            "theta3": theta3,
            "b3": b3,
        }

        self.assertDictAllClose(student, truth)
        print_success_message("test_update_weights")

    def test_update_weights_with_adam(self):
        nn = dlnet(y=np.random.randn(1, 30), use_dropout=True, use_adam=True)
        nn_param = open("data/test_data/nn_param_new.pickle", "rb")
        nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        dLoss_file = open("data/test_data/dLoss_new.pickle", "rb")
        dLoss = pickle.load(dLoss_file)
        dLoss_file.close()

        nn.update_weights(dLoss)
        student = nn.parameters
        # print(student)

        theta1 = np.array(
            [
                [1.0084761, 0.22103087, 0.55507464, 1.28378029, 1.06823511],
                [-0.55423165, 0.5385338, -0.07738613, -0.04959343, 0.22705916],
                [0.0731636, 0.8296252, 0.42938534, 0.0602491, 0.24626456],
            ]
        )

        b1 = np.array([-0.01, -0.01, 0.01, -0.01, 0.01]).T

        theta2 = np.array(
            [
                [0.1592237, 0.65817247, -0.10174956, 0.15000813, -0.39196323],
                [-1.13173175, 0.28230712, 0.39658762, -0.34190629, 1.00506513],
                [-0.6604121, 0.01046383, -0.07371116, 0.6954797, 0.66711722],
                [0.0592946, 0.15911942, -0.40702986, -0.87583911, -0.14559104],
                [0.07992138, 0.56020272, 0.54772062, -0.18321782, -0.1251939],
            ]
        )

        b2 = np.array([0.01, 0.01, -0.01, 0.01, 0.01]).T

        theta3 = np.array(
            [
                [-0.45892714, -0.64505133, -0.75306723],
                [0.88241328, -0.21792339, -0.20591278],
                [-0.57026712, 0.33770426, -0.71175706],
                [-0.08514035, -0.39046482, 0.18302806],
                [-0.218439, -0.51799476, -0.02260348],
            ]
        )

        b3 = np.array([0.01, 0.01, -0.01]).T

        truth = {
            "theta1": theta1,
            "b1": b1,
            "theta2": theta2,
            "b2": b2,
            "theta3": theta3,
            "b3": b3,
        }

        self.assertDictAllClose(student, truth)
        print_success_message("test_update_weights_with_adam")

    def test_backward(self):
        nn = dlnet(y=np.random.randn(1, 30), use_dropout=True, use_adam=False)
        nn_param = open("data/test_data/nn_param_new.pickle", "rb")
        nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        cache = open("data/test_data/test_compute_gradients_cache_new.pickle", "rb")
        nn.cache = pickle.load(cache)
        cache.close()

        for p in nn.cache:
            nn.cache[p] = nn.cache[p]

        y = np.array(
            [
                [1, 0, 0],  # Class 0 for Sample 1
                [0, 1, 0],  # Class 1 for Sample 2
                [0, 0, 1],  # Class 2 for Sample 3
                [1, 0, 0],  # Class 0 for Sample 4
                [0, 1, 0],  # Class 1 for Sample 5
                [1, 0, 0],  # Class 0 for Sample 6
                [0, 0, 1],  # Class 2 for Sample 7
                [0, 1, 0],  # Class 1 for Sample 8
                [1, 0, 0],  # Class 0 for Sample 9
                [0, 0, 1],  # Class 2 for Sample 10
            ]
        )
        yh = np.array(
            [
                [0.44241676, 0.33622821, 0.22135503],
                [0.60897122, 0.22194954, 0.16907924],
                [0.36570828, 0.2890999, 0.34519182],
                [0.62958633, 0.12340082, 0.24701284],
                [0.74252438, 0.13642363, 0.12105198],
                [0.37250028, 0.35025135, 0.27724837],
                [0.41267473, 0.2645114, 0.32281387],
                [0.35629951, 0.3272992, 0.31640129],
                [0.44909499, 0.32278295, 0.22812206],
                [0.41330537, 0.28034051, 0.30635412],
            ]
        )
        nn.backward(y, yh)
        student = nn.parameters

        expected_theta1 = np.array(
            [
                [1.0183503, 0.23098702, 0.56507048, 1.29378365, 1.07829375],
                [-0.56426857, 0.54852145, -0.08752723, -0.05952862, 0.23714654],
                [0.08312001, 0.83958235, 0.43937657, 0.07027908, 0.25632641],
            ]
        )

        expected_b1 = np.array(
            [
                -2.84173687e-04,
                -5.43573952e-05,
                -1.46322467e-04,
                3.14270516e-05,
                9.01249158e-05,
            ]
        )

        expected_theta2 = np.array(
            [
                [0.14921192, 0.66795331, -0.09186358, 0.13997491, -0.38194383],
                [-1.14174132, 0.29185609, 0.3864581, -0.33203586, 1.01504386],
                [-0.65044732, 0.02015714, -0.08383603, 0.68536676, 0.65711265],
                [0.06926256, 0.16868007, -0.39688262, -0.88600371, -0.15566757],
                [0.06993051, 0.54980997, 0.53769992, -0.17339855, -0.13523062],
            ]
        )

        expected_b2 = np.array(
            [
                -1.29740089e-05,
                -5.69888788e-04,
                -1.68471616e-04,
                -1.10470059e-04,
                3.99553870e-06,
            ]
        )

        expected_theta3 = np.array(
            [
                [-0.47008866, -0.63478087, -0.76217616],
                [0.87234346, -0.22775744, -0.19600891],
                [-0.56062281, 0.3479851, -0.72168221],
                [-0.09584304, -0.40048682, 0.17375275],
                [-0.2281777, -0.52833755, -0.012522],
            ]
        )

        expected_b3 = np.array([-0.00079308, 0.00034771, 0.00044537])

        expected = {
            "theta1": expected_theta1,
            "b1": expected_b1,
            "theta2": expected_theta2,
            "b2": expected_b2,
            "theta3": expected_theta3,
            "b3": expected_b3,
        }
        self.assertDictAllClose(student, expected)
        print_success_message("test_backward")

    def test_gradient_descent(self):
        x_train, y_train, x_test, y_test = get_housing_dataset()

        nn = dlnet(y_train, lr=0.01, use_dropout=False, use_adam=False)

        nn.gradient_descent(x_train, y_train, iter=3, local_test=True)

        gd_loss = np.array([1.086892, 1.086749, 1.086607])
        gd_loss_test = np.allclose(np.array(nn.loss), gd_loss, rtol=1e-1)
        print("\nYour GD losses works within the expected range:", gd_loss_test)
        self.assertTrue(gd_loss_test)

    # def test_stochastic_gradient_descent(self):
    #     x_train, y_train, x_test, y_test = get_housing_dataset()

    #     nn = dlnet(y_train, lr=0.01, use_dropout=False, use_adam=False)

    #     nn.stochastic_gradient_descent(x_train, y_train, iter=3, local_test=True)

    #     gd_loss = np.array([1.030836, 1.090258, 1.210505])
    #     gd_loss_test = np.allclose(np.array(nn.loss), gd_loss, rtol=1e-1)
    #     print("\nYour SGD losses works within the expected range:", gd_loss_test)

    def test_minibatch_gradient_descent(self):
        x_train, y_train, x_test, y_test = get_housing_dataset()

        np.random.seed(0)
        nn = dlnet(y_train, lr=0.01, batch_size=6, use_dropout=False, use_adam=False)

        bgd_loss = np.array([1.11102194, 1.10795187, 1.07966684])

        batch_y = np.array(
            [
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
            ]
        )
        batch_y = batch_y.reshape((3, 6, 3))

        nn.minibatch_gradient_descent(x_train, y_train, iter=3, local_test=True)

        batch_y_test = np.allclose(np.array(nn.batch_y), batch_y, rtol=1e-1)
        print("Your batch_y works within the expected range:", batch_y_test)

        bgd_loss_test = np.allclose(np.array(nn.loss), bgd_loss, rtol=1e-2)
        print(
            "\nYour mini-batch GD losses works within the expected range:",
            bgd_loss_test,
        )
        self.assertTrue(bgd_loss_test)

    def test_gradient_descent_with_adam(self):
        gd_loss_with_adam = [1.086276, 1.080218, 1.074970]
        np.random.seed(0)
        x_train, y_train, x_test, y_test = get_housing_dataset()
        nn = dlnet(y_train, lr=0.01, use_dropout=False, use_adam=True)
        nn.gradient_descent(x_train, y_train, iter=3, local_test=True)
        gd_loss_test_with_adam = np.allclose(
            np.array(nn.loss), gd_loss_with_adam, rtol=1e-2
        )
        print(
            "\nYour GD losses works within the expected range:",
            gd_loss_test_with_adam,
        )
        self.assertTrue(gd_loss_test_with_adam)

    def test_minibatch_gradient_descent_with_adam(self):
        np.random.seed(0)
        x_train, y_train, x_test, y_test = get_housing_dataset()

        nn = dlnet(y_train, lr=0.01, use_dropout=False, use_adam=True)

        nn.minibatch_gradient_descent(x_train, y_train, iter=3, local_test=True)

        gd_loss = np.array([1.080808, 1.095893, 1.071157])
        gd_loss_test = np.allclose(np.array(nn.loss), gd_loss, rtol=1e-2)
        print("\nYour GD losses works within the expected range:", gd_loss_test)
        self.assertTrue(gd_loss_test)


class TestCNN(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.student_cnn = CNN()

    def test_model_architecture(self):
        arbitrary_input = torch.randn((1, 1, 84, 84))
        try:
            output = self.student_cnn.forward(arbitrary_input)
            output = output.squeeze()
            self.assertTrue(
                output.ndim == 1 and output.shape[0] == 4,
                f"Expected output to contain 4 classes. Yours contains {output.shape[0]} classes.",
            )
        except RuntimeError as e:
            if "size mismatch" in str(e) or "shape" in str(e):
                self.assertTrue(False, f"Model hidden layers incompatible: {str(e)}")
            else:
                self.assertTrue(False, f"Runtime Error: {str(e)}")
        print_success_message("test_model_architecture")

    def test_cnn_train_loss_plot(self, trainer):
        (
            train_loss,
            train_acc,
            test_loss,
            test_acc,
        ) = trainer.get_training_history()

        self.assertTrue(
            len(train_loss) == len(train_acc), "len(train_loss) != len(train_acc)"
        )

        THRESHOLD = 3
        increasing_edges = 0
        decreasing_edges = 0

        N = 3
        # Check if train loss is decreasing and train accuracy is increasing
        for i in range(len(train_loss) - N):
            j = i + N

            if train_loss[j] > train_loss[i]:
                increasing_edges += 1

            if train_acc[j] < train_acc[i]:
                decreasing_edges += 1

        self.assertTrue(
            increasing_edges < THRESHOLD,
            f"In train loss plot: {increasing_edges} increasing edges >= {THRESHOLD} threshold\n{train_loss}",
        )
        self.assertTrue(
            decreasing_edges < THRESHOLD,
            f"In train accuracy plot: {decreasing_edges} decreasing edges >= {THRESHOLD} threshold\n{train_acc}",
        )
        print_success_message("test_cnn_train_loss_plot")

    def test_cnn_test_loss_plot(self, trainer):
        (
            train_loss,
            train_acc,
            test_loss,
            test_acc,
        ) = trainer.get_training_history()

        self.assertTrue(
            len(test_loss) == len(test_acc), "len(train_loss) != len(train_acc)"
        )

        THRESHOLD = 3
        increasing_edges = 0
        decreasing_edges = 0
        N = 3

        # Check if train loss is decreasing and train accuracy is increasing
        for i in range(len(test_loss) - N):
            j = i + N

            if test_loss[j] > test_loss[i]:
                increasing_edges += 1

            if test_acc[j] < test_acc[i]:
                decreasing_edges += 1

        self.assertTrue(
            increasing_edges < THRESHOLD,
            f"In test loss plot: {increasing_edges} increasing edges >= {THRESHOLD} threshold\n{test_loss}",
        )
        self.assertTrue(
            decreasing_edges < THRESHOLD,
            f"In test accuracy plot: {decreasing_edges} decreasing edges >= {THRESHOLD} threshold\n{test_acc}",
        )
        print_success_message("test_cnn_test_loss_plot")

    def test_cnn_confusion_matrix(self, trainer, testloader):
        classes = ["glioma", "meningioma", "notumor", "pituitary"]
        y_pred, y_pred_classes, y_gt_classes = trainer.predict(testloader)
        y_pred_prob = torch.max(y_pred, dim=1).values
        # print(f"Test accuracy: {accuracy_score(y_gt_classes, y_pred_classes)}")

        num_classes = len(torch.unique(y_gt_classes))
        correct_per_class = torch.zeros(num_classes)
        total_per_class = torch.zeros(num_classes)

        # Count correct predictions and total samples for each class
        for pred, gt in zip(y_pred_classes, y_gt_classes):
            total_per_class[gt] += 1
            if pred == gt:
                correct_per_class[gt] += 1

        # Calculate accuracy percentage for each class
        accuracy_per_class = torch.where(
            total_per_class > 0,
            (correct_per_class / total_per_class),
            torch.zeros_like(total_per_class),
        )
        ACCURACY_THRESHOLD = 0.5
        # print(accuracy_per_class)
        ConfusionMatrixDisplay.from_predictions(
            y_gt_classes, y_pred_classes, normalize="true", display_labels=classes
        )
        plt.show()
        self.assertTrue(
            (accuracy_per_class >= ACCURACY_THRESHOLD).all(),
            f"In confusion matrix, diagonal entries should be >= {ACCURACY_THRESHOLD}",
        )
        print_success_message("test_cnn_confusion_matrix")


class TestRNN(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.model = RNN(vocab_size=34, max_input_len=40)

    def test_rnn_architecture(self):
        self.model.set_hyperparameters()
        self.model.define_model()

        embedding_dim = self.model.hp["embedding_dim"]
        rnn_units = self.model.hp["rnn_units"]

        expected_model = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.model.vocab_size, embedding_dim=embedding_dim
            ),
            nn.RNN(input_size=embedding_dim, hidden_size=rnn_units, batch_first=True),
            RNNOutputAdapter(),
            nn.Linear(in_features=rnn_units, out_features=self.model.vocab_size),
        )

        passed, feedback = assert_network_layers_match(self.model.model, expected_model)
        if passed:
            print_success_message("test_rnn_architecture")
        else:
            print_fail_message("test_rnn_architecture", feedback)


class TestLSTM(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.model = LSTM(vocab_size=34, max_input_len=40)

    def test_lstm_architecture(self):
        self.model.set_hyperparameters()
        self.model.define_model()

        embedding_dim = self.model.hp["embedding_dim"]
        lstm_units = self.model.hp["lstm_units"]

        expected_model = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.model.vocab_size, embedding_dim=embedding_dim
            ),
            nn.LSTM(input_size=embedding_dim, hidden_size=lstm_units, batch_first=True),
            LSTMOutputAdapter(),
            nn.Linear(in_features=lstm_units, out_features=self.model.vocab_size),
        )

        passed, feedback = assert_network_layers_match(self.model.model, expected_model)
        if passed:
            print_success_message("test_lstm_architecture")
        else:
            print_fail_message("test_lstm_architecture", feedback)


class TestSVM(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_kernel(self):
        N, D = 50, 2
        X = np.random.randn(N, D)
        y = np.sign(np.random.randn(N))

        def phi(datum):
            """
            Takes a data point of dimension 2 and maps it to a new feature space.
            """
            projected = np.zeros((5,))
            projected[0] = datum[0]
            projected[1] = datum[1]
            projected[2] = datum[0] * datum[1]
            projected[3] = datum[0] ** 2
            projected[4] = datum[1] ** 2
            return projected

        K = kernel_construction(X, phi)

        # Just testing properties, the autograder will test exact values.
        self.assertIsInstance(K, np.ndarray)
        self.assertEqual(
            len(K.shape),
            2,
            f"Kernel matrices are 2Darrays as (N, N), got shape: {K.shape}.",
        )
        self.assertEqual(
            K.shape[0],
            K.shape[1],
            f"Kernel matrices are pairwise dot products of the data. It should be a square matrix (N, N), got shape: {K.shape}.",
        )
        self.assertEqual(
            K.shape[0],
            N,
            f"You computed the wrong amount of pairwise similarities. It should be a square matrix (N, N), got shape: {K.shape}.",
        )
        for i in range(N):
            for j in range(N):
                self.assertAlmostEqual(
                    K[i, j],
                    K[j, i],
                    f"The kernel should be symmetric, since the dot product is symmetric. K[{i},{j}]:({K[i,j]}) != K[{j},{i}]:({K[j,i]})",
                )

        print_success_message("test_kernel")

    def test_rbf_kernel(self):
        x_1 = np.array(
            [
                -2.127,
                -1.472,
                -1.819,
                -2.139,
                -1.978,
                -2.033,
                -1.452,
                -2.302,
                -2.043,
                -2.627,
                0.561,
                -0.252,
                1.777,
                -0.613,
                0.787,
                0.104,
                1.386,
                0.489,
                -0.180,
                0.971,
                1.428,
                1.066,
                1.302,
                0.365,
                0.637,
                0.327,
                0.640,
                0.186,
                -0.726,
                1.177,
                0.598,
                -0.630,
                1.462,
                0.092,
                1.051,
                4.327,
                3.825,
                2.874,
                2.750,
                3.738,
                2.717,
                3.855,
                3.145,
                3.683,
                3.249,
                3.494,
                3.007,
                4.250,
                3.088,
                3.281,
                7.456,
                2.000,
                4.314,
                9.363,
                7.870,
                -3.509,
                4.200,
                -0.205,
                -0.012,
                2.042,
                0.976,
                6.217,
                3.444,
                0.886,
                2.175,
                1.734,
                6.376,
                -0.420,
                1.652,
                -3.016,
            ]
        )
        x_2 = np.array(
            [
                2.116,
                0.480,
                1.174,
                2.689,
                2.241,
                -1.172,
                1.140,
                -0.181,
                -0.123,
                0.492,
                -3.063,
                0.784,
                1.037,
                -0.890,
                2.723,
                -1.745,
                0.054,
                -0.224,
                1.839,
                1.763,
                0.185,
                0.453,
                -1.065,
                -2.376,
                -0.417,
                0.187,
                1.476,
                1.442,
                -0.464,
                -0.362,
                -1.258,
                -1.704,
                -2.047,
                2.340,
                -0.611,
                0.874,
                0.154,
                1.367,
                -1.481,
                0.482,
                -0.821,
                -1.044,
                -0.694,
                -0.373,
                0.067,
                -1.398,
                1.080,
                0.558,
                -1.843,
                1.785,
                2.447,
                5.653,
                5.864,
                4.257,
                7.269,
                3.545,
                5.045,
                4.812,
                6.532,
                6.469,
                5.154,
                5.378,
                4.112,
                3.519,
                4.652,
                5.156,
                6.230,
                6.202,
                4.612,
                4.697,
            ]
        )
        y = np.array(
            [
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ]
        )
        X = np.column_stack((x_1, x_2))
        gamma = 1e-3

        K = rbf_kernel(
            X.copy(), gamma
        )  # if you're reading this and interested, this is an intentionally low gamma to induce misclassification

        # Just testing properties, the autograder will test exact values.
        self.assertIsInstance(K, np.ndarray)
        self.assertEqual(
            len(K.shape),
            2,
            f"Kernel matrices are 2Darrays as (N, N), got shape: {K.shape}.",
        )
        self.assertEqual(
            K.shape[0],
            K.shape[1],
            f"Kernel matrices are pairwise dot products of the data. It should be a square matrix (N, N), got shape: {K.shape}.",
        )
        self.assertEqual(
            K.shape[0],
            X.shape[0],
            f"You computed the wrong amount of pairwise similarities. It should be a square matrix (N, N), got shape: {K.shape}.",
        )
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.assertAlmostEqual(
                    K[i, j],
                    K[j, i],
                    f"The kernel should be symmetric, since the dot product is symmetric. K[{i},{j}]:({K[i,j]}) != K[{j},{i}]:({K[j,i]})",
                )
        # This test may not work, since scikit might be doing something unique.
        svm_model = SVC(kernel="precomputed")
        svm_model.fit(K, y)
        y_pred = svm_model.predict(K)
        student_acc = accuracy_score(y, y_pred)
        svm_model = SVC(kernel="rbf", gamma=gamma)
        svm_model.fit(X, y)
        y_pred = svm_model.predict(X)
        scikit_acc = accuracy_score(y, y_pred)
        self.assertAlmostEqual(
            student_acc,
            scikit_acc,
            msg=f"This test may be oversensitive, but your rbf kernel performs differently than expected.",
        )

        print_success_message("test_rbf_kernel")


def assert_network_layers_match(actual_model, expected_model):
    """
    Compares two PyTorch nn.Sequential models to see if their architectures match
    with the same rigor as a Keras .get_config() comparison.

    This function checks for layer count, layer type (semantic equality), and the
    critical configuration parameters of each layer (literal equality).

    Args:
        actual_model (nn.Sequential): The student's implemented model.
        expected_model (nn.Sequential): The reference (teacher's) model.

    Returns:
        A tuple (bool, str):
        - True if the models match, False otherwise.
        - A detailed feedback string explaining the first mismatch found, or None if they match.
    """

    student_total_params = sum(param.numel() for param in actual_model.parameters())
    teacher_total_params = sum(param.numel() for param in expected_model.parameters())

    if student_total_params != teacher_total_params:
        feedback = "\nMismatch in parameter count between actual and expected models!\n"
        return False, feedback

    # 3. If all checks pass
    return True, None


def get_layer_type(layer_name):
    split = layer_name.split("_")
    if len(split) > 0:
        return split[0]
    return layer_name


def print_success_message(test_name):
    print(test_name + " passed!")


def print_fail_message(test_name, feedback=None):
    print(test_name + " failed")
    print(feedback)
