import numpy as np

from permucate.data.dr_learner import DGPRLearner
from permucate.data.high_dim import HighDimCaussim
from permucate.data.hines_vim import DGPHines


def test_reproducibility_high_dim():
    simulator = HighDimCaussim(
        d=100,
        d_imp_a=5,
        d_imp_tau=5,
        d_imp_mu=5,
        random_state=0,
        var_corr=0.2,
        treatment_ratio=0.5,
        link_fn_name="nystroem",
        link_fn_kwargs={"n_components": 10},
    )
    df_1 = simulator.sample(n=1000, random_state=0)
    df_2 = simulator.sample(n=1000, random_state=0)

    assert np.array_equal(df_1.values, df_2.values)
    assert df_1.a.mean() - 0.5 < 1e-1

    simulator = HighDimCaussim(
        d=100,
        d_imp_a=5,
        d_imp_tau=5,
        d_imp_mu=5,
        random_state=0,
        var_corr=0.2,
        treatment_ratio=0.5,
        same_var=False,
        link_fn_name="poly",
        link_fn_kwargs={"degree": 2},
    )
    simulator_2 = HighDimCaussim(
        d=100,
        d_imp_a=5,
        d_imp_tau=5,
        d_imp_mu=5,
        random_state=0,
        var_corr=0.2,
        treatment_ratio=0.5,
        same_var=False,
        link_fn_name="poly",
        link_fn_kwargs={"degree": 2},
    )
    df_3 = simulator.sample(n=1000, random_state=0)
    df_4 = simulator_2.sample(n=1000, random_state=0)

    assert np.array_equal(df_3.values, df_4.values)
    assert df_3.a.mean() - 0.5 < 1e-1


def test_reproducibility_dr_learner():
    simulator = DGPRLearner(
        d=10,
        alpha=5,
        beta=5,
        scenario="high_dim",
        random_state=0,
    )
    df_1 = simulator.sample(n=100, random_state=0)
    df_2 = simulator.sample(n=100, random_state=0)

    assert np.array_equal(df_1.values, df_2.values)
    assert df_1.a.mean() - 0.5 < 1e-1

    simulator = DGPRLearner(
        d=10,
        alpha=5,
        beta=5,
        gamma=2,
        scenario="high_dim_2",
        random_state=0,
    )
    df_1 = simulator.sample(n=100, random_state=0)
    df_2 = simulator.sample(n=100, random_state=0)

    assert np.array_equal(df_1.values, df_2.values)

    simulator = DGPRLearner(
        d=10,
        scenario="poly",
        random_state=0,
    )
    simulator_2 = DGPRLearner(
        d=10,
        scenario="poly",
        random_state=0,
    )
    df_3 = simulator.sample(n=100, random_state=0)
    df_4 = simulator_2.sample(n=100, random_state=0)

    assert np.array_equal(df_3.values, df_4.values)
    assert df_3.a.mean() - 0.5 < 1e-1


def test_reproducibility_hines_vim():
    simulator = DGPHines(dgp_id=1, random_state=0)
    df_1 = simulator.sample(n=100, random_state=0)
    df_2 = simulator.sample(n=100, random_state=0)

    assert np.array_equal(df_1.values, df_2.values)
    assert df_1.a.mean() - 0.5 < 1e-1

    simulator = DGPHines(dgp_id=2, random_state=0)
    simulator_2 = DGPHines(dgp_id=2, random_state=0)
    df_3 = simulator.sample(n=100, random_state=0)
    df_4 = simulator_2.sample(n=100, random_state=0)

    assert np.array_equal(df_3.values, df_4.values)
    assert df_3.a.mean() - 0.5 < 1e-1

    simulator = DGPHines(dgp_id=3, random_state=0)
    df_5 = simulator.sample(n=100, random_state=0)
    df_6 = simulator.sample(n=100, random_state=0)

    assert np.array_equal(df_5.values, df_6.values)
