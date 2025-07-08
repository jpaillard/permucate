import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold

from permucate.data.high_dim import HighDimCaussim
from permucate.data.hines_vim import DGPHines
from permucate.importance import cross_val_vim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def joblib_one_cv_vim(n, vim_method, seed, model, meta_learner, config, output_dir):
    if config["dgp"] == "hines":
        simulator = DGPHines(random_state=seed, **config["dgp_params"])
    elif config["dgp"] == "high_dim":
        simulator = HighDimCaussim(
            random_state=0,
            # This seed controls the important variable selection, not the
            # sample generation. We keep it fixed but change the seed in
            # `simulator.sample`
            **config["dgp_params"],
        )
        np.save(
            output_dir
            / f"{config['dgp']}_{vim_method}_{n}_{seed}_{meta_learner}_{model}_imp_var.npy",  # noqa
            simulator.imp_var_tau,
        )
    else:
        raise ValueError(f"Unknown DGP: {config['dgp']}")

    df = simulator.sample(n, random_state=seed)
    x_cols = [x for x in df.columns if x.startswith("x")]

    cv = StratifiedKFold(
        n_splits=config["n_splits_simul"], shuffle=True, random_state=seed
    )

    output = cross_val_vim(
        df=df,
        importance_estimator=RidgeCV(alphas=np.logspace(-3, 3, 10)),
        cv=cv,
        model=model,
        meta_learner=meta_learner,
        learner_cv=config["n_splits_learner"],
        x_cols=x_cols,
        scoring=config["scoring"],
        n_perm=config["n_perm"],
        method=vim_method,
        random_state=seed,
        n_jobs=config["n_jobs_inner"],
        return_coefs=config["return_coefs"],
    )

    vi_list = output["vim"]
    if config["return_coefs"] and (vim_method == "loco"):
        coefs_list = output["coefs"]
        coefs_j_list = output["coefs_j"]

        vi_arr = np.array(vi_list)
        coefs_arr = np.array(coefs_list)
        coefs_j_arr = np.array(coefs_j_list)
        np.save(
            output_dir
            / f"{config['dgp']}_{vim_method}_{n}_{seed}_{meta_learner}_{model}_coefs.npy",  # noqa
            coefs_arr,
        )
        np.save(
            output_dir
            / f"{config['dgp']}_{vim_method}_{n}_{seed}_{meta_learner}_{model}_coefs_j.npy",  # noqa
            coefs_j_arr,
        )
    elif config["return_coefs"] and (vim_method == "permucate"):
        vi_arr = np.array(vi_list)
        nu_j = output["nu_j"]
        np.save(
            output_dir
            / f"{config['dgp']}_{vim_method}_{n}_{seed}_{meta_learner}_{model}_nu_j.npy",  # noqa
            nu_j,
        )

    else:
        vi_arr = np.array(vi_list)

    if vim_method == "permucate":
        vi_arr /= 2
    np.save(
        output_dir
        / f"{config['dgp']}_{vim_method}_{n}_{seed}_{meta_learner}_{model}.npy",
        vi_arr,
    )

    return None


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir)

    _ = Parallel(n_jobs=config["n_jobs"])(
        delayed(joblib_one_cv_vim)(
            n, vim_method, seed, model, meta_learner, config, output_dir
        )
        for n in config["n_list"]
        for vim_method in config["vim_methods"]
        for seed in reversed(range(config["n_seeds"]))
        for model in config["models"]
        for meta_learner in config["meta_learners"]
    )


if __name__ == "__main__":
    main()
