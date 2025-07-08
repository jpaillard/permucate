import numpy as np

from permucate.data.dr_learner import DGPRLearner
from permucate.scoring import (
    compute_pseudo_outcome_risk,
    compute_r_risk,
    compute_tau_risk,
)
from permucate.utils import get_learner


def test_dr_learners():
    simulator = DGPRLearner(scenario="poly", random_state=0)
    df = simulator.sample(50)

    dr_learner = get_learner(
        model="linear", meta_learner="DR", cv=2, n_iter=1, n_jobs=1
    )
    dr_learner.fit(
        X=df["x_1"].values.reshape(-1, 1), Y=df["Y"].values, T=df["a"].values
    )
    dr_effect = dr_learner.effect(df["x_1"].values.reshape(-1, 1))

    risk = compute_pseudo_outcome_risk(
        df_test=df,
        x_cols=["x_1"],
        cate_estimator=dr_learner,
        pseudo_outcomes=dr_learner.compute_pseudo_outcomes(
            X=df["x_1"].values.reshape(-1, 1), Y=df["Y"].values, T=df["a"].values
        ),
    )

    mu_0 = dr_learner.mu_0.predict(df["x_1"].values.reshape(-1, 1))
    mu_1 = dr_learner.mu_1.predict(df["x_1"].values.reshape(-1, 1))
    pi_hat = dr_learner.model_propensity.predict_proba(df["x_1"].values.reshape(-1, 1))[
        :, 1
    ]
    m_hat = mu_0 * (1 - pi_hat) + mu_1 * pi_hat

    r_score = compute_r_risk(
        df_test=df.rename(columns={"Y": "y"}),  # rename for consistency
        x_cols=["x_1"],
        cate_estimator=dr_learner,
        pi_hat=pi_hat,
        m_hat=m_hat,
    )

    assert dr_effect.shape[0] == df.shape[0]
    assert risk.shape[0] == df.shape[0]
    assert r_score.shape[0] == df.shape[0]


def test_t_learners():
    simulator = DGPRLearner(scenario="poly", random_state=0)
    df = simulator.sample(50)

    t_learner = get_learner(
        model="super_learner",
        meta_learner="T",
        n_iter=1,
        cv=2,
        n_jobs=2,
        random_search_dict_reg={
            "lr__alpha": np.logspace(-3, 3, 10),
            "dt__learning_rate": np.logspace(-3, 0, 5),
            "dt__max_leaf_nodes": np.arange(10, 100, 5),
            "dt__max_iter": [5],
        },
        random_search_clf={
            "lr__C": np.logspace(-3, 3, 10),
            "dt__learning_rate": np.logspace(-3, 0, 5),
            "dt__max_leaf_nodes": np.arange(10, 100, 5),
            "dt__max_iter": [5],
        },
    )
    t_learner.fit(X=df["x_1"].values.reshape(-1, 1), Y=df["Y"].values, T=df["a"].values)
    t_effect = t_learner.effect(df["x_1"].values.reshape(-1, 1))

    tau_score = compute_tau_risk(
        df_test=df,
        x_cols=["x_1"],
        cate_estimator=t_learner,
        tau_true=[0.5] * df.shape[0],
    )

    assert t_effect.shape[0] == df.shape[0]
    assert tau_score.shape[0] == df.shape[0]
