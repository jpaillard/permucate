import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid
from scipy.stats import bernoulli, norm
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class HighDimCaussim:
    def __init__(
        self,
        d: int,
        d_imp_a: int,
        d_imp_tau: int,
        d_imp_mu: int,
        effect_size: float = 0.5,
        treatment_ratio: float = 0.5,
        link_fn_name: str = "nystroem",
        link_fn_kwargs: dict = dict(n_components=500),
        random_state: int = None,
        var_corr: float = 0,
        y_noise: float = 1e-3,
        same_var=True,
        normalize=False,
    ) -> None:
        """
        Class to generate high dimensional data with causal structure

        Parameters
        ----------
        d : int
            Dimension of the data
        d_imp_a : int
            Number of important variables for the treatment assignment
        d_imp_tau : int
            Number of important variables for the treatment effect
        d_imp_mu : int
            Number of important variables for the outcome mean
        effect_size : float, optional
            Effect size of the treatment effect, by default .5
        treatment_ratio : float, optional
            Ratio treated / non-treated, by default .5
        link_fn_name : str, optionala
            Function used to create the feature map that links covariates to
            outcome, treatment assignment ... , by default 'nystroem'
        link_fn_kwargs : dict, optional
            Arguments for the link function, by default dict(n_components=500)
        random_state : int, optional
            Random state, by default None
        var_corr : float, optional
            Correlation between the variables of the simulation, by default 0
        y_noise : float, optional
            Strength of the noise affecting the outcome, by default 1
        same_var : bool, optional
            If True, the treatment assignment and the responses are linked to
            the same variables, by default True
        """

        self.d = d
        self.d_imp_a = d_imp_a
        self.d_imp_tau = d_imp_tau
        self.d_imp_mu = d_imp_mu
        self.effect_size = effect_size
        self.treatment_ratio = treatment_ratio
        self.link_fn_name = link_fn_name
        self.link_fn_kwargs = link_fn_kwargs
        self.random_state = random_state
        self.var_corr = var_corr
        self.y_noise = y_noise
        self.same_var = same_var
        self.normalize = normalize

        self.cov_mat = np.ones((d, d)) * var_corr
        np.fill_diagonal(self.cov_mat, 1)

        self.rng = np.random.default_rng(random_state)
        if self.link_fn_name == "nystroem":
            self.link_fn = Nystroem(**self.link_fn_kwargs, random_state=random_state)
            self.latent_dim = self.link_fn_kwargs["n_components"]
        if self.link_fn_name == "poly":
            self.link_fn = Pipeline(
                [
                    (
                        "poly features",
                        PolynomialFeatures(
                            **self.link_fn_kwargs,
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            self.latent_dim = self.link_fn[0]._num_combinations(
                n_features=self.d_imp_a,
                min_degree=0,
                max_degree=self.link_fn_kwargs["degree"],
                interaction_only=False,
                include_bias=True,
            )
        if self.link_fn_name == "rbf":
            self.link_fn = RBFSampler(**self.link_fn_kwargs, random_state=random_state)
            self.latent_dim = self.link_fn_kwargs["n_components"]
        self.init_scenario()

    def init_scenario(self):
        self.imp_var_a = self.rng.choice(self.d, self.d_imp_a, replace=False)
        self.coef_a = self.rng.choice([-1, 1], size=self.latent_dim)
        if self.same_var:
            self.imp_var_tau = self.imp_var_a
            self.imp_var_mu = self.imp_var_a
        else:
            self.imp_var_tau = self.rng.choice(self.d, self.d_imp_tau, replace=False)
            self.imp_var_mu = self.rng.choice(self.d, self.d_imp_mu, replace=False)

        self.coef_tau = self.rng.choice([-1, 1], size=self.latent_dim)
        self.coef_mu = self.rng.choice([-1, 1], size=self.latent_dim)
        return None

    def sample(self, n, random_state=None):
        rng_sample = np.random.default_rng(random_state)

        X = rng_sample.multivariate_normal(
            np.zeros(self.d),
            self.cov_mat,
            size=n,
        )

        X_imp_mu = X[:, self.imp_var_mu].copy()

        # Treatment assignment:
        #  1. Compute feature map
        # 2. Apply threshold to assign `self.treatment_ratio` of the samples to
        # the treated group
        X_imp_a = X[:, self.imp_var_a].copy()
        features_a = self.link_fn.fit_transform(X_imp_a)
        feat_map_a = features_a @ self.coef_a
        threshold = np.quantile(feat_map_a, 1 - self.treatment_ratio)

        propensity = expit(feat_map_a - threshold)
        a = bernoulli.rvs(propensity, random_state=rng_sample).astype(int)

        # Treatment effect
        X_imp_tau = X[:, self.imp_var_tau].copy()
        features_tau = self.link_fn.fit_transform(X_imp_tau)
        tau = features_tau @ self.coef_tau
        if self.normalize:
            tau = StandardScaler().fit_transform(tau.reshape(-1, 1)).ravel()
        tau *= self.effect_size

        # Response functions
        features_mu = self.link_fn.fit_transform(X_imp_mu)
        mu_0 = features_mu @ self.coef_mu
        if self.normalize:
            mu_0 = StandardScaler().fit_transform(mu_0.reshape(-1, 1)).ravel()
        mu_0 *= 1 - self.effect_size
        mu_1 = mu_0 + tau

        # Outcome
        y = norm(loc=mu_0 + a * tau * self.effect_size, scale=self.y_noise).rvs(
            size=n, random_state=rng_sample.integers(1e3)
        )

        df = pd.DataFrame(
            np.column_stack((X, a, y, tau * self.effect_size, mu_0, mu_1, propensity)),
            columns=[f"x{i}" for i in range(self.d)]
            + ["a", "y", "tau", "mu_0", "mu_1", "pi"],
        )
        return df
