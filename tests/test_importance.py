import numpy as np
from sklearn.linear_model import RidgeCV

from permucate.data.hines_vim import DGPHines
from permucate.importance import cross_val_vim


def test_loco():
    simulator = DGPHines(dgp_id=4, random_state=0)
    df = simulator.sample(50)
    x_cols = [x for x in df.columns if x.startswith("x")]

    output = cross_val_vim(
        df=df,
        importance_estimator=RidgeCV(alphas=np.logspace(-3, 3, 10), cv=2),
        cv=2,
        model="linear",
        meta_learner="DR",
        learner_cv=2,
        x_cols=x_cols,
        scoring="pseudo_outcome_risk",
        n_perm=3,
        method="loco",
        random_state=0,
        n_jobs=2,
        return_coefs=True,
    )

    # D x 1 x K
    # D: number of features
    # 1: number of permutation for PermuCATE
    # K: number of CV splits
    assert output["vim"].shape == (len(x_cols), 2)

    assert output["coefs"].shape == (len(x_cols), len(x_cols), 2)
    assert output["coefs_j"].shape == (len(x_cols), len(x_cols) - 1, 2)


def test_permucate():
    simulator = DGPHines(dgp_id=4, random_state=0)
    df = simulator.sample(50)
    x_cols = [x for x in df.columns if x.startswith("x")]

    output = cross_val_vim(
        df=df,
        importance_estimator=RidgeCV(alphas=np.logspace(-3, 3, 10), cv=2),
        cv=2,
        model="linear",
        meta_learner="T",
        model_nuisances="linear",
        learner_cv=2,
        x_cols=x_cols,
        scoring="pseudo_outcome_risk",
        n_perm=3,
        method="permucate",
        random_state=0,
        n_jobs=2,
        return_coefs=True,
    )

    # D x P x K
    # D: number of features
    # P: number of permutation
    # K: number of CV splits
    assert output["vim"].shape == (len(x_cols), 3, 2)

    # D x N x K
    # D: number of features
    # N: number of test samples per CV split
    # K: number of CV splits
    assert output["nu_j"].shape == (len(x_cols), 25, 2)
