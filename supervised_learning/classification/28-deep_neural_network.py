#!/usr/bin/env python3
"""Task 28"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """DeepNeuralNetwork"""

    def __init__(self, nx, layers, activation='sig'):
        """Initializes DeepNeuralNetwork"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if min(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for lix, layer_size in enumerate(layers, 1):
            if type(layer_size) is not int:
                raise TypeError("layers must be a list of positive integers")
            w = np.random.randn(layer_size, nx) * np.sqrt(2 / nx)
            self.__weights["W{}".format(lix)] = w
            self.__weights["b{}".format(lix)] = np.zeros((layer_size, 1))
            nx = layer_size

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

    def activate(self, Z):
        """Applies the activation function to Z"""
        if self.__activation == 'sig':
            return 1 / (1 + np.exp(-Z))
        elif self.__activation == 'tanh':
            return np.tanh(Z)

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data,
               where nx is number of input features & m is number of examples.

        Returns:
            The output of the neural network and the cache, respectively.
        """
        self.__cache["A0"] = X

        for i in range(self.__L):
            Z = np.matmul(self.__weights["W{}".format(i + 1)],
                          self.__cache["A{}".format(i)]) + self.__weights[
                              "b{}".format(i + 1)]

            sm_act = np.sum(np.exp(Z), axis=0, keepdims=True)
            if i == self.__L - 1:
                # softmax activation for last layer
                self.__cache["A{}".format(i + 1)] = np.exp(Z) / sm_act
            else:
                # activation function for hidden layers
                self.__cache["A{}".format(i + 1)] = self.activate(Z)

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Arguments:
            Y: correct labels for input data
            A: activated output of neuron for each example

        Returns:
            The cost
        """
        m = Y.shape[1]
        return -1 / m * np.sum(Y * np.log(A))

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        Arguments:
            X: input data
            Y: correct labels

        Returns:
            Neuron's prediction and cost of the network
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A == np.amax(A, axis=0), 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        Arguments:
            Y: correct labels for the input data
            cache: dict containing all intermediary values of the network
            alpha: learning rate
        """
        W_tmp = None
        m = Y.shape[1]

        # first iteration
        A = cache["A{}".format(self.__L)]
        dZ = A - Y

        for i in reversed(range(1, self.__L + 1)):
            if W_tmp is not None:
                A = cache["A{}".format(i)]
                dZ = np.matmul(W_tmp.T, dZ) * (A * (1 - A))

            dW = np.matmul(dZ, cache["A{}".format(i - 1)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            W_tmp = self.__weights.get("W{}".format(i))

            s_w = self.__weights["W{}".format(i)] - alpha * dW
            s_w_2 = self.__weights["b{}".format(i)] - alpha * db

            self.__weights["W{}".format(i)] = s_w
            self.__weights["b{}".format(i)] = s_w_2

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the deep neural network

        Arguments:
            X: input data
            Y: correct labels for the input data
            iterations: number of iterations to train over
            alpha: learning rate
            verbose: defines whether or not to print information
                     about the training
            graph: defines whether or not to graph information
                   about the training once the training has completed
            step: number of iterations between printing and graphing
                  information

        Returns:
            Evaluation of training data after iterations of training
            have occurred
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 1 or step > iterations:
            step = iterations

        costs = []
        for i in range(iterations + 1):
            self.forward_prop(X)

            if i % step == 0 or i == iterations:
                costs.append(self.cost(Y,
                                       self.__cache["A{}".format(self.__L)]))

                if verbose:
                    print("Cost after {} iterations: {}".format(i, costs[-1]))

                if graph:
                    plt.plot(np.arange(0, i + 1, step), costs)
                    plt.xlabel("iteration")
                    plt.ylabel("cost")
                    plt.title("Training Cost")

            if i == iterations:
                return self.evaluate(X, Y)

            self.gradient_descent(Y, self.__cache, alpha)

        plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format

        Arguments:
            filename: file to which the object should be saved
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object

        Arguments:
            filename: file from which the object should be loaded

        Returns:
            The loaded object, or None if filename doesn't exist
        """
        try:
            with open(filename, 'rb') as file:
                obj = pickle.load(file)
                if isinstance(obj, DeepNeuralNetwork):
                    return obj
                return None
        except FileNotFoundError:
            return None
