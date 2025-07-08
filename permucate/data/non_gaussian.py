# %%
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import PolynomialFeatures

# %%


def sample_multimodal(
    n_samples, n_features, n_modes, rng, mean_range=(-1, 1), std_range=(0.1, 1)
):
    """
    Sample from a multimodal distribution
    """
    X_list = []
    for _ in range(n_features):
        modes_prop = rng.uniform(size=n_modes)
        modes_prop /= modes_prop.sum()
        n_samples_per_mode = np.array(n_samples * modes_prop, dtype=int)
        n_samples_per_mode[-1] += n_samples - n_samples_per_mode.sum()

        means = rng.uniform(*mean_range, size=n_modes)
        stds = rng.uniform(*std_range, size=n_modes)
        X = np.hstack(
            [
                rng.normal(loc=mean, scale=std, size=n_sample)
                for mean, std, n_sample in zip(means, stds, n_samples_per_mode)
            ]
        )
        X_list.append(X)
    X = np.stack(X_list, axis=1)
    return X


def link_function(latent_vars, n_features, rng, sparsity=0.1, degree=None):
    """
    Generate features through polynomial link function"
    """
    if degree is not None:
        link = PolynomialFeatures(degree=degree, include_bias=False)
    else:
        link = Nystroem(n_components=200, random_state=rng.integers(0, 1000))
    X_out_list = []
    X_linked = link.fit_transform(latent_vars)

    for _ in range(n_features):
        suppprt = rng.choice(
            np.arange(X_linked.shape[1]),
            int((1 - sparsity) * X_linked.shape[1]),
            replace=False,
        )
        X_linked_sub = X_linked[:, suppprt]
        beta_tmp = rng.choice([-2, -1, 0, 1, 2], X_linked_sub.shape[1])
        X_out_list.append(np.dot(X_linked_sub, beta_tmp))
    X_out = np.stack(X_out_list, axis=1)
    return X_out


# %%
import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid
from scipy.stats import bernoulli, norm
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class NonGaussianX:
    def __init__(
        self,
        d_latent: int,
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
        d_latent : int
            Dimension of the latent variables
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

        latent_vars = rng_sample.multivariate_normal(
            np.zeros(self.d),
            self.cov_mat,
            size=n,
        )
        X = self.link_function(latent_vars, self.d, sparsity=0.95, degree=None)

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

    def link_function(self, latent_vars, n_features, sparsity=0.1, degree=None):
        """
        Generate features through polynomial link function"
        """
        if degree is not None:
            link = PolynomialFeatures(degree=degree, include_bias=False)
        else:
            link = Nystroem(n_components=200, random_state=self.rng.integers(0, 1000))
        X_out_list = []
        X_linked = link.fit_transform(latent_vars)

        for _ in range(n_features):
            suppprt = self.rng.choice(
                np.arange(X_linked.shape[1]),
                int((1 - sparsity) * X_linked.shape[1]),
                replace=False,
            )
            X_linked_sub = X_linked[:, suppprt]
            beta_tmp = self.rng.choice([-2, -1, 0, 1, 2], X_linked_sub.shape[1])
            X_out_list.append(np.dot(X_linked_sub, beta_tmp))
        X_out = np.stack(X_out_list, axis=1)
        return X_out


class MultiModalX:
    def __init__(
        self,
        n_modes: int,
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
        n_modes : int
            Number of modes in the data
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
        self.n_modes = n_modes

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
        self.means = self.rng.uniform(-10, 10, size=(self.d, self.n_modes))
        self.stds = self.rng.uniform(0.1, 1, size=(self.d, self.n_modes))
        self.prop_modes = self.rng.uniform(size=(self.d, self.n_modes))
        self.prop_modes /= self.prop_modes.sum(axis=1)[:, None]
        return None

    def sample_x(self, n, random_state=None):
        X_list = []
        for i in range(self.d):
            n_samples_per_mode = np.array(n * self.prop_modes[i], dtype=int)
            n_samples_per_mode[-1] += n - n_samples_per_mode.sum()

            X = np.hstack(
                [
                    self.rng.normal(
                        loc=mean,
                        scale=std,
                        size=n_sample,
                    )
                    for mean, std, n_sample in zip(
                        self.means[i], self.stds[i], n_samples_per_mode
                    )
                ]
            )
            X_list.append(X)
        X = np.stack(X_list, axis=1)
        return X

    def sample(self, n, random_state=None):
        rng_sample = np.random.default_rng(random_state)
        X = self.link_function(
            self.sample_x(n, random_state=random_state),
            self.d,
            sparsity=0.9,
            degree=2,
        )
        X = StandardScaler().fit_transform(X)

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

    def link_function(self, latent_vars, n_features, sparsity=0.1, degree=None):
        """
        Generate features through polynomial link function"
        """
        if degree is not None:
            link = PolynomialFeatures(degree=degree, include_bias=False)
        else:
            link = Nystroem(n_components=200, random_state=self.rng.integers(0, 1000))
        X_out_list = []
        X_linked = link.fit_transform(latent_vars)

        for _ in range(n_features):
            suppprt = self.rng.choice(
                np.arange(X_linked.shape[1]),
                int((1 - sparsity) * X_linked.shape[1]),
                replace=False,
            )
            X_linked_sub = X_linked[:, suppprt]
            beta_tmp = self.rng.choice([-2, -1, 0, 1, 2], X_linked_sub.shape[1])
            X_out_list.append(np.dot(X_linked_sub, beta_tmp))
        X_out = np.stack(X_out_list, axis=1)
        return X_out
