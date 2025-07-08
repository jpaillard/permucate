# %%
import argparse
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tqdm import tqdm

from permucate.data.non_gaussian import MultiModalX
from permucate.importance import joblib_compute_conditional_one, joblib_compute_loco_one
from permucate.learners import CateNet, CausalForest, DRLearner
from permucate.scoring import compute_tau_risk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="linear")
    parser.add_argument("--model_name", type=str, default="TabPFN")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Offset for the random seeds. Used to parallelize the experiments using "
        "SLURM job array.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_multimodal",
    )
    return parser.parse_args()


# %%
def get_learner(model_name):
    if model_name == "linear":
        return DRLearner(
            model_final=RidgeCV(alphas=np.logspace(-3, 3, 50)),
            model_propensity=LogisticRegressionCV(Cs=np.logspace(-3, 3, 50)),
            model_regression=RidgeCV(alphas=np.logspace(-3, 3, 50)),
            cv=5,
            random_state=0,
        )
    elif model_name == "TabPFN":
        return DRLearner(
            model_final=TabPFNRegressor(),
            model_propensity=TabPFNClassifier(),
            model_regression=TabPFNRegressor(),
            cv=5,
            random_state=0,
        )
    elif model_name == "CateNet":
        return CateNet(
            transformation="RA",
            first_stage_strategy="S2",
            n_units_r=100,
            n_layers_out=2,
            n_layers_r=3,
            penalty_l2_t=0.01 / 100,
            penalty_l2=0.01 / 100,
            n_layers_out_t=2,
            n_layers_r_t=3,
            cross_fit=True,
        )
    elif model_name == "CausalForest":
        return CausalForest()
    else:
        raise ValueError(f"Unknown model name {model_name}")


def vim_one(simulator, n_sample, s, model_name, j_imp, output_dir):

    df = simulator.sample(n=n_sample, random_state=s)
    if (model_name == "TabPFN") and (n_sample > 10000):
        df = df.sample(n=10000, random_state=s)

    x_cols = df.columns[df.columns.str.startswith("x")]
    if model_name == "TabPFN":
        df = df.astype(np.float32)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=s)

    learner = get_learner(model_name)
    learner.fit(
        X=df_train[x_cols].values,
        Y=df_train["y"].values,
        T=df_train["a"].values,
    )

    imputation_model = RidgeCV(alphas=np.logspace(-3, 3, 50))

    score_ref = compute_tau_risk(df_test=df_test, x_cols=x_cols, cate_estimator=learner)
    output = joblib_compute_conditional_one(
        df_train=df_train,
        df_test=df_test,
        col_idx=j_imp,
        fitted_learner=learner,
        importance_estimator=[imputation_model] * len(x_cols),
        n_perm=100,
        random_state=s,
        x_cols=x_cols,
        scoring_params={"tau_true": df_test["tau"].values},
        score_fn=compute_tau_risk,
        score_ref=score_ref,
    )
    vim = output["vim"]
    res_df_cpi = pd.DataFrame(
        {
            "model": model_name,
            "n": n_sample,
            "seed": s,
            "score_ref": score_ref.mean(),
            "vim": vim.mean(),
            "method": "PermuCATE",
        },
        index=[0],
    )

    output_loco = joblib_compute_loco_one(
        df_train=df_train,
        df_test=df_test,
        col_idx=j_imp,
        fitted_learner=learner,
        importance_estimator=clone(learner),
        x_cols=x_cols,
        random_state=s,
        scoring_params={"tau_true": df_test["tau"].values},
        score_fn=compute_tau_risk,
        score_ref=score_ref,
    )
    vim_loco = output_loco["vim"]
    res_df_loco = pd.DataFrame(
        {
            "model": model_name,
            "n": n_sample,
            "seed": s,
            "score_ref": score_ref.mean(),
            "vim": vim_loco.mean(),
            "method": "LOCO",
        },
        index=[0],
    )
    np.save(
        output_dir / f"imp_var_{model_name}_{n_sample}_{s}_{j_imp}.npy",
        simulator.imp_var_tau,
    )
    np.save(
        output_dir / f"imp_var_loco_{model_name}_{n_sample}_{s}_{j_imp}.npy",
        simulator.imp_var_tau,
    )
    np.save(
        output_dir / f"imp_var_cpi_{model_name}_{n_sample}_{s}_{j_imp}.npy",
        simulator.imp_var_tau,
    )

    return pd.concat([res_df_cpi, res_df_loco])


# %%
def main(args):
    d_latent = 25
    d = 50
    d_imp = 10
    scenario = args.scenario
    n_jobs = args.n_jobs
    model_name = args.model_name

    if scenario == "linear":
        degree = 1
    elif scenario == "nonlinear":
        degree = 3

    simulator = MultiModalX(
        n_modes=3,
        d=d,
        d_imp_a=d_imp,
        d_imp_tau=d_imp,
        d_imp_mu=d_imp,
        effect_size=0.5,
        random_state=0,
        link_fn_name="poly",
        link_fn_kwargs={"degree": degree},
        var_corr=0.5,
        y_noise=0.1,
    )

    df = simulator.sample(n=100000, random_state=0)
    df_x = df.drop(columns=["y", "a", "tau", "mu_1", "mu_0", "pi"])

    # consider the first important variable, reduces computation time
    j_imp = simulator.imp_var_tau[0]

    df_list = []
    n_list = [
        200,
        400,
        800,
        1600,
        3200,
        6400,
        12800,
    ]
    seed_offset = args.seed_offset
    n_seeds = args.n_seeds
    seeds = np.arange(seed_offset * 10, seed_offset * 10 + n_seeds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    df_list = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(vim_one)(simulator, n_sample, s, model_name, j_imp, output_dir)
        for n_sample, s in tqdm(product(n_list, seeds), total=len(n_list) * len(seeds))
    )

    df_res = pd.concat(df_list)

    df_res.to_csv(
        output_dir
        / f"variance_comparison_{model_name}_{d}_{scenario}_{seed_offset}.csv",
        index=False,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
