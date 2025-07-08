"""
This module contains the implementation of the data generating processes
from the paper:
1.Kennedy, E. H. Towards optimal doubly robust estimation of heterogeneous
causal effects. Electron. J. Stat. 17, (2023).
"""

import numpy as np
import pandas as pd
from scipy.special import expit


class DGPRLearner:
    def __init__(
        self,
        scenario: str = "poly",
        random_state: int = None,
        d=500,
        alpha=50,
        beta=50,
        gamma=50,
    ) -> None:
        """

        Parameters
        ----------
        scenario : str, optional
            Scenario to simulate, by default 'poly'
        random_state : int, optional
            Random state, by default None
        d : int, optional
            Number of dimensions, by default 500
        alpha : int, optional
            Number of important features for the propensity score,
            by default 50
        beta : int, optional
            Number of important features for the response functino ,
            by default 50

        """
        self.scenario = scenario
        self.rng = np.random.default_rng(random_state)
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.init_scenario()

    def init_scenario(self):
        if self.scenario == "poly":
            self.propensity = self._propensity_poly
            self.response = np.vectorize(self._response_poly)
        elif self.scenario == "high_dim":
            self.important_features_a = self.rng.choice(
                self.d, self.alpha, replace=False
            )
            self.important_features_mu = self.rng.choice(
                self.d, self.beta, replace=False
            )
            self.propensity = self._propensity_high_dim
            self.response = self._response_high_dim
        elif self.scenario == "high_dim_2":
            self.important_features_a = self.rng.choice(
                self.d, self.alpha, replace=False
            )
            self.important_features_mu = self.rng.choice(
                self.d, self.beta, replace=False
            )
            self.imp_var_tau = self.rng.choice(self.d, self.gamma, replace=False)

            self.coef_a = self.rng.choice([-2, -1, 1, 2], self.alpha) * 5
            self.coef_mu = self.rng.choice([-2, -1, 1, 2], self.beta)
            self.coef_tau = self.rng.choice([-2, -1, 1, 2], self.gamma)

            self.propensity = self._propensity_high_dim_2
            self.response = self._response_high_dim_2
            self.tau = self._tau_high_dim_2
        return None

    def sample(self, n, random_state=None):
        rng_sample = np.random.default_rng(random_state)

        if self.scenario == "poly":
            X = rng_sample.uniform(-1, 1, n)
        elif (self.scenario == "high_dim") or (self.scenario == "high_dim_2"):
            X = rng_sample.multivariate_normal(self.d * [0], np.eye(self.d), n)

        propensity = self.propensity(X)
        a = rng_sample.binomial(1, propensity)
        mu_0 = self.response(X)
        mu_1 = self.response(X)
        cate = mu_1 - mu_0
        _mean = np.zeros(n)
        # _var = np.square(np.diag(0.2 - 0.1 * np.cos(2 * np.pi * X)))
        _var = np.square(np.diag(np.ones(n) * 0.1))

        if self.scenario == "poly":
            Y = (
                mu_0
                + a * cate
                + rng_sample.multivariate_normal(_mean, _var, 1).flatten()
            )
            df = pd.DataFrame(
                {
                    "x_1": X,
                    "a": a,
                    "Y": Y,
                    "mu_0": mu_0,
                    "mu_1": mu_1,
                    "cate": cate,
                }
            )
        elif self.scenario == "high_dim":
            Y = rng_sample.binomial(1, mu_0)

            values = np.concatenate(
                [
                    X,
                    a[:, None],
                    Y[:, None],
                    mu_0[:, None],
                    mu_1[:, None],
                    cate[:, None],
                ],
                axis=1,
            )
            columns = [f"x_{i}" for i in range(self.d)] + [
                "a",
                "Y",
                "mu_0",
                "mu_1",
                "cate",
            ]
            df = pd.DataFrame(values, columns=columns)

        elif self.scenario == "high_dim_2":
            tau = self.tau(X)
            mu_0 = self.response(X, a=0)
            mu_1 = mu_0 + tau
            _mean = np.zeros(n)
            # _var = np.square(np.diag(0.2 - 0.1 * np.cos(2 * np.pi * X)))
            _var = np.square(np.diag(np.ones(n) * 2))

            Y = mu_0 + a * tau + rng_sample.multivariate_normal(_mean, _var).flatten()
            values = np.concatenate(
                [
                    X,
                    a[:, None],
                    Y[:, None],
                    tau[:, None],
                    mu_0[:, None],
                    mu_1[:, None],
                    propensity[:, None],
                ],
                axis=1,
            )
            columns = [f"x_{i}" for i in range(self.d)] + [
                "a",
                "y",
                "tau",
                "mu_0",
                "mu_1",
                "pi",
            ]
            df = pd.DataFrame(values, columns=columns)
        return df

    def _propensity_poly(self, X):
        return 0.5 + 0.4 * np.sign(X)

    def _response_poly(self, x, a=None):
        if -1 <= x <= -0.5:
            return (x + 2) ** 2 / 2
        elif -0.5 < x <= 0:
            return x / 2 + 0.875
        elif 0 < x <= 0.5:
            return -5 * (x - 0.2) ** 2 + 1.075
        elif 0.5 < x <= 1:
            return x + 0.125

    def _propensity_high_dim(self, x, a=None):
        x_a = x[:, self.important_features_a]
        return expit(np.sum(x_a, axis=1) / (2 * np.sqrt(self.alpha)))

    def _response_high_dim(self, x, a=None):
        x_mu = x[:, self.important_features_mu]
        return expit(np.sum(x_mu, axis=1) / np.sqrt(self.beta))

    def _propensity_high_dim_2(self, x, a=None):
        x_a = x[:, self.important_features_a] * self.coef_a
        prop_tmp = np.sum(x_a, axis=1) / 2 / np.sqrt(np.linalg.norm(self.coef_a, ord=2))
        return expit(prop_tmp)

    def _response_high_dim_2(self, x, a=None):
        x_mu = x[:, self.important_features_mu] * self.coef_mu
        return np.sum(x_mu, axis=1) / np.sqrt(np.linalg.norm(self.coef_mu))

    def _tau_high_dim_2(self, x, a=None):
        x_tau = x[:, self.imp_var_tau] * self.coef_tau
        return np.sum(x_tau, axis=1) / np.sqrt(np.linalg.norm(self.coef_tau))
