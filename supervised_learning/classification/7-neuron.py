#!/usr/bin/env python3
"""Task 7"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """Initialize the Neuron class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output"""
        return self.__A

    def forward_prop(self, X):
        """Perform forward propagation"""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Calculate the cost using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions"""
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Perform one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m

        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=3000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """Train the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 1 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []
        iterations_list = []
        for iteration in range(iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if verbose and iteration % step == 0:
                cost = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(iteration, cost))

            if graph and iteration % step == 0:
                cost = self.cost(Y, A)
                costs.append(cost)
                iterations_list.append(iteration)

        if graph:
            plt.plot(iterations_list, costs, 'b-')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        prediction, cost = self.evaluate(X, Y)
        return prediction, cost
