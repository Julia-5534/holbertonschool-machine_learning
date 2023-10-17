#!/usr/bin/env python3
"""Task 5"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian Optimization Class"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initializes Bayesian Optimization"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.sampled_points = set()

    def acquisition(self):
        """Calculates acquisition function values for
        all points in the search space. Calculates the
        next best sample location"""
        mu_s, sigma_s = self.gp.predict(self.X_s)

        with np.errstate(divide='warn'):
            if self.minimize:
                imp = np.min(self.gp.Y) - mu_s - self.xsi
            else:
                imp = mu_s - np.max(self.gp.Y) - self.xsi

            Z = imp / sigma_s
            EI = imp * norm.cdf(Z) + sigma_s * norm.pdf(Z)
            EI[sigma_s == 0.0] = 0.0
            X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Finds next sample point by maxing (or min-ing) the acquire
        function and evaluates the black-box function at this point and
        updates the Gaussian process with the new sample."""
        for _ in range(iterations):
            next_sample, _ = self.acquisition()
            next_sample_tuple = tuple(next_sample)
            if next_sample_tuple in self.sampled_points:
                break
            next_output = self.f(next_sample)
            self.gp.update(next_sample, next_output)
            self.sampled_points.add(next_sample_tuple)

        X_opt = np.array(next_sample)
        Y_opt = np.array(next_output)
        return X_opt, Y_opt
