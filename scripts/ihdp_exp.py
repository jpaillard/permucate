import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catenets.datasets.dataset_ihdp import get_one_data_set, load_raw, prepare_ihdp_data
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tqdm import tqdm

from permucate.importance import compute_variable_importance
from permucate.learners import CateNet, CausalForest, DRLearner
from permucate.scoring import compute_tau_risk

DATA_DIR = Path("./data")
CATEGORICAL_COL_IDX = np.arange(6, 25)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute variable importance")
    parser.add_argument(
        "--outdir",
        type=str,
        default="./results_ihdp",
        help="Dataset",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument(
        "--model_name",
        type=str,
        default="cate_nets",
        help="Model name. Should be in ['CateNet', 'CF', 'TabPFN']",
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Offset for the random seeds. Used to parallelize the experiments using "
        "SLURM job array.",
    )
    args = parser.parse_args()
    return args


def transform_dict(data_exp, beta, rng, exp_offset=0.5, sigma_y=1, tau=4):
    X = data_exp["X"]
    a = data_exp["w"].ravel()
    y = data_exp["y"].ravel()

    W = np.ones_like(X) * exp_offset
    W[:, 0] = 0
    mu_0 = np.exp(np.dot(X + W, beta))
    mu_1 = np.dot(X, beta)
    omega = np.mean(mu_1[a == 1] - mu_0[a == 1]) - tau
    mu_1 -= omega

    y_0 = rng.normal(mu_0, sigma_y)
    y_1 = rng.normal(mu_1, sigma_y)
    y = y_0 * (1 - a) + y_1 * a
    ycf = y_0 * a + y_1 * (1 - a)

    output_dict = {
        "X": X,
        "w": a.reshape(-1, 1),
        "y": y.reshape(-1, 1),
        "ycf": ycf.reshape(-1, 1),
        "mu0": mu_0.reshape(-1, 1),
        "mu1": mu_1.reshape(-1, 1),
    }
    return output_dict


def get_ihdp_data(dataset_idx, random_state=0):
    loadings = [0, 0.1, 0.2, 0.3, 0.4]
    probas = [0.6, 0.1, 0.1, 0.1, 0.1]

    rng = np.random.default_rng(random_state)
    data_exp = get_one_data_set(data_train, i_exp=dataset_idx, get_po=True)
    data_exp_test = get_one_data_set(data_test, i_exp=dataset_idx, get_po=True)
    beta = rng.choice(loadings, size=data_exp["X"].shape[1], p=probas)
    data_exp_out = transform_dict(data_exp, beta, rng)
    data_exp_test_out = transform_dict(data_exp_test, beta, rng)
    return data_exp_out, data_exp_test_out, beta


def get_model(model_name):
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


def vim_one_cpi(data_idx, model_name, random_state=0):
    data_exp, data_exp_test, beta = get_ihdp_data(data_idx, random_state=random_state)
    X, y, w, cate_true_in, X_t, cate_true_out = prepare_ihdp_data(
        data_exp,
        data_exp_test,
        rescale=True,
    )
    cate_true_out = cate_true_out.ravel()
    x_cols = [f"x_{i}" for i in range(X.shape[1])]
    df_train = pd.DataFrame(
        np.hstack(
            [X, y.reshape(-1, 1), w.reshape(-1, 1), data_exp["mu0"], data_exp["mu1"]]
        ),
        columns=x_cols + ["y", "a", "mu0", "mu1"],
    )
    for idx in CATEGORICAL_COL_IDX:
        df_train[x_cols[idx]] = df_train[x_cols[idx]].astype(float)
    df_train["x_13"] -= 1
    df_test = pd.DataFrame(
        np.hstack(
            [
                X_t,
                data_exp_test["y"].reshape(-1, 1),
                data_exp_test["w"].reshape(-1, 1),
                data_exp_test["mu0"],
                data_exp_test["mu1"],
            ]
        ),
        columns=x_cols + ["y", "a", "mu0", "mu1"],
    )
    for idx in CATEGORICAL_COL_IDX:
        df_test[x_cols[idx]] = df_test[x_cols[idx]].astype(float)
    df_test["x_13"] -= 1
    importance_estimator = [RidgeCV(alphas=np.logspace(-3, 3, 50))] * (
        len(x_cols) - len(CATEGORICAL_COL_IDX)
    ) + [LogisticRegressionCV(Cs=np.logspace(-3, 3, 50))] * len(CATEGORICAL_COL_IDX)
    learner = get_model(model_name)
    learner.fit(
        X=df_train[x_cols].values,
        Y=df_train["y"].values,
        T=df_train["a"].values,
    )
    score_learner = compute_tau_risk(
        df_test=df_test, x_cols=x_cols, cate_estimator=learner, tau_true=cate_true_out
    )
    vim = compute_variable_importance(
        df_train=df_train,
        df_test=df_test,
        importance_estimator=importance_estimator,
        fitted_learner=learner,
        scoring="tau_risk",
        x_cols=x_cols,
        n_perm=100,
        method="permucate",
        random_state=0,
        n_jobs=1,
        scoring_params={"tau_true": cate_true_out},
    )
    return vim, score_learner


def vim_one_loco(data_idx, model_name, random_state=0):
    data_exp, data_exp_test, beta = get_ihdp_data(data_idx, random_state=random_state)
    X, y, w, cate_true_in, X_t, cate_true_out = prepare_ihdp_data(
        data_exp,
        data_exp_test,
        rescale=True,
    )
    cate_true_out = cate_true_out.ravel()
    x_cols = [f"x_{i}" for i in range(X.shape[1])]
    df_train = pd.DataFrame(
        np.hstack(
            [X, y.reshape(-1, 1), w.reshape(-1, 1), data_exp["mu0"], data_exp["mu1"]]
        ),
        columns=x_cols + ["y", "a", "mu0", "mu1"],
    )
    df_test = pd.DataFrame(
        np.hstack(
            [
                X_t,
                data_exp_test["y"].reshape(-1, 1),
                data_exp_test["w"].reshape(-1, 1),
                data_exp_test["mu0"],
                data_exp_test["mu1"],
            ]
        ),
        columns=x_cols + ["y", "a", "mu0", "mu1"],
    )

    learner = get_model(model_name)
    importance_estimator = clone(learner)

    learner.fit(
        X=df_train[x_cols].values,
        Y=df_train["y"].values,
        T=df_train["a"].values,
    )

    vim = compute_variable_importance(
        df_train=df_train,
        df_test=df_test,
        importance_estimator=importance_estimator,
        fitted_learner=learner,
        scoring="tau_risk",
        x_cols=x_cols,
        method="loco",
        random_state=0,
        n_jobs=1,
        scoring_params={"tau_true": cate_true_out},
    )
    return vim


def main(args):
    _, _, beta = get_ihdp_data(1, random_state=0)
    results_dir = Path(args.outdir)
    results_dir.mkdir(parents=True, exist_ok=True)

    seed_offset_10 = args.seed_offset

    print("Computing VIM with CPI...")
    output = Parallel(n_jobs=args.n_jobs)(
        delayed(vim_one_cpi)(
            i + 1 + seed_offset_10 * 10, args.model_name, random_state=0
        )
        for i in tqdm(range(args.n_seeds))
    )
    vim_list_cpi, score_learner = zip(*output)
    vim_arr = np.stack(vim_list_cpi)
    score_arr = np.stack(score_learner)
    np.save(results_dir / f"vim_cpi_{args.model_name}{seed_offset_10}", vim_arr)
    np.save(results_dir / f"score_learner_{args.model_name}{seed_offset_10}", score_arr)

    print("Computing VIM with LOCO...")
    vim_list_loco = Parallel(n_jobs=args.n_jobs)(
        delayed(vim_one_loco)(
            i + 1 + seed_offset_10 * 10, args.model_name, random_state=0
        )
        for i in tqdm(range(args.n_seeds))
    )
    vim_arr = np.stack(vim_list_loco)
    np.save(results_dir / f"vim_loco_{args.model_name}{seed_offset_10}", vim_arr)


if __name__ == "__main__":
    args = parse_args()
    data_train, data_test = load_raw(DATA_DIR)

    main(args)
