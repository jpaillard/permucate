import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid
from scipy.stats import bernoulli, norm


class DGPHines:
    def __init__(
        self,
        dgp_id: int = 3,
        random_state: int = None,
    ):
        """
        Class to generate data from the Hines DGP

        Parameters
        ----------
        dgp_id : int, optional
            ID of the DGP, by default 3
        random_state : int, optional
            Random state, by default None
        """

        self.dgp_id = dgp_id
        self.random_state = random_state

    def sample(self, n, random_state=None):

        rng = np.random.default_rng(random_state)
        if self.dgp_id == 1:
            x_1, x_2 = rng.uniform(-1, 1, size=(2, n))
            pi = expit(-0.4 * x_1 + 0.1 * x_1 * x_2)
            a = bernoulli.rvs(pi, size=n, random_state=rng)
            tau = x_1**3 + 1.4 * x_1**2 + 25 * x_2**2 / 9
            mu_0 = x_1 * x_2 + 2 * x_2**2 - x_1
            mu_1 = mu_0 + tau
            self.imp_var_tau = [0, 1]

            # N(X11X2 + 2X22 − X1 + Aτ (1)(X), 1)
            y = rng.normal(loc=x_1 * x_2 + 2 * x_2**2 - x_1 + a * tau, scale=1)
            X = np.column_stack((x_1, x_2))
            df = pd.DataFrame(
                np.column_stack((X, a, y, tau, mu_0, mu_1, pi)),
                columns=["x1", "x2", "a", "y", "tau", "mu_0", "mu_1", "pi"],
            )
            return df
        if self.dgp_id == 2:
            x_1, x_2 = rng.uniform(-1, 1, size=(2, n))
            pi = expit(-0.4 * x_1 + 0.1 * x_1 * x_2)
            a = bernoulli.rvs(pi, size=n, random_state=rng)
            # DGP 2 is the same as DGP 1 with a 10x decrease in effect size
            tau = x_1**3 + 1.4 * x_1**2 + 25 * x_2**2 / 9 / 10
            mu_0 = x_1 * x_2 + 2 * x_2**2 - x_1
            mu_1 = mu_0 + tau
            self.imp_var_tau = [0, 1]

            # N(X11X2 + 2X22 − X1 + Aτ (1)(X), 1)
            y = norm(loc=x_1 * x_2 + 2 * x_2**2 - x_1 + a * tau, scale=1).rvs(
                size=n, random_state=rng
            )
            X = np.column_stack((x_1, x_2))
            df = pd.DataFrame(
                np.column_stack((X, a, y, tau, mu_0, mu_1, pi)),
                columns=["x1", "x2", "a", "y", "tau", "mu_0", "mu_1", "pi"],
            )
            return df
        if self.dgp_id == 3:
            corr = 0.5
            x_temp = rng.multivariate_normal([0, 0], [[1, corr], [corr, 1]], size=n)
            x_1, x_2 = x_temp[:, 0], x_temp[:, 1]
            del x_temp
            x_temp = rng.multivariate_normal([0, 0], [[1, corr], [corr, 1]], size=n)
            x_3, x_4 = x_temp[:, 0], x_temp[:, 1]
            del x_temp
            x_temp = rng.multivariate_normal([0, 0], [[1, corr], [corr, 1]], size=n)
            x_5, x_6 = x_temp[:, 0], x_temp[:, 1]
            del x_temp

            pi = expit(-0.4 * x_1 + 0.1 * x_1 * x_2 + 0.2 * x_5)
            a = bernoulli.rvs(pi, size=n, random_state=rng)

            tau = x_1 + 2 * x_2 + x_3
            self.imp_var_tau = [0, 1, 2]
            mu_0 = x_3 - x_6
            mu_1 = mu_0 + tau
            # y = norm(
            #     loc=x_3 - x_6 + a * tau, scale=2
            # ).rvs(size=n, random_state=rng)
            y = rng.normal(loc=x_3 - x_6 + a * tau, scale=3, size=n)

            X = np.column_stack((x_1, x_2, x_3, x_4, x_5, x_6))
            df = pd.DataFrame(
                np.column_stack((X, a, y, tau, mu_0, mu_1, pi)),
                columns=[
                    "x1",
                    "x2",
                    "x3",
                    "x4",
                    "x5",
                    "x6",
                    "a",
                    "y",
                    "tau",
                    "mu_0",
                    "mu_1",
                    "pi",
                ],
            )
            return df
        if self.dgp_id == 4:
            corr = 0.0
            x_temp = rng.multivariate_normal([0, 0], [[1, corr], [corr, 1]], size=n)
            x_1, x_2 = x_temp[:, 0], x_temp[:, 1]
            del x_temp
            x_temp = rng.multivariate_normal([0, 0], [[1, corr], [corr, 1]], size=n)
            x_3, x_4 = x_temp[:, 0], x_temp[:, 1]
            del x_temp
            x_temp = rng.multivariate_normal([0, 0], [[1, corr], [corr, 1]], size=n)
            x_5, x_6 = x_temp[:, 0], x_temp[:, 1]
            del x_temp

            pi = expit(-0.4 * x_1 + 0.1 * x_1 * x_2 + 0.2 * x_5)
            a = bernoulli.rvs(pi, size=n, random_state=rng)

            tau = x_1 + 2 * x_2 + x_3
            self.imp_var_tau = [0, 1, 2]
            mu_0 = x_3 - x_6
            mu_1 = mu_0 + tau
            # y = norm(
            #     loc=x_3 - x_6 + a * tau, scale=2
            # ).rvs(size=n, random_state=rng)
            nois_scale = 3.0
            y = rng.normal(loc=x_3 - x_6 + a * tau, scale=nois_scale, size=n)

            X = np.column_stack((x_1, x_2, x_3, x_4, x_5, x_6))
            df = pd.DataFrame(
                np.column_stack((X, a, y, tau, mu_0, mu_1, pi)),
                columns=[
                    "x1",
                    "x2",
                    "x3",
                    "x4",
                    "x5",
                    "x6",
                    "a",
                    "y",
                    "tau",
                    "mu_0",
                    "mu_1",
                    "pi",
                ],
            )
            return df
