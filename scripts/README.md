# Experiment scripts
This directory contains Python scripts to reproduce the experiments presented in the paper. 

## Variable Importance on Causal Simulations

| Script Name         | `cross_val_vim_exp.py`                |
|---------------------|---------------------------------------|
| Related Experiments | Figure 1, 3, S1, S2                   |
| Short Description   | Computes variable importance on simulated CATE |

### Usage 
```bash
python ./cross_val_vim.py --config ./config.yaml
```

The `config.yaml` file contains the parameters for the experiment. The following parameters can be set:
| Parameter           | Description                                                                                                   |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| `output_dir`        | Directory where the results will be saved                                                                     |
| `vim_methods`       | List of variable importance methods to be used (`['cpi', 'loco']`). Use `'cpi'` for the proposed PermuCATE method. |
| `n_list`            | List of sample sizes to be used                                                                               |
| `meta_learners`     | List of meta-learners to be used (`['T', 'DR']`)                                                             |
| `models`            | List of models to be used (`['linear', 'rf', 'super_learner']`)                                              |
| `n_perm`            | Number of permutations to be used for the PermuCATE method                                                   |
| `n_seeds`           | Number of random samples drawn for each sample size                                                          |
| `n_splits_learner`  | Number of splits for the cross-validation of the DR-learner                                                  |
| `n_splits_simul`    | Number of splits for the variable importance estimation                                                      |
| `scoring`           | Scoring method to be used for the DR-learner (`'pseudo_outcome_risk'`, `'r_risk'`)                           |
| `dgp`               | Data generating process (`'hines'` for LD, `'high_dim'` for HL and HP depending on `dgp_params`)             |
| `dgp_params`        | Dictionary with the parameters of the data generating process. For `high_dim`, specify `degree: 1` for HL and `degree: 3` for HP |
| `return_coefs`      | Boolean to return the coefficients of the models for evaluating the terms delta beta and variance(nu)         |
| `n_jobs`            | Number of jobs to be used for the parallelization of the experiments                                          |
| `n_jobs_inner`      | Number of jobs to be used for the parallelization of the inner cross-validation of the DR-learner            |

### Output
The output is a folder containing the files `<dgp>_<vim_method>_<n>_<seed>_<meta_learner>_<model>.npy` with seed ranging from `0` to `n_seeds` in the format:
where the different placeholders (`<dgp>`, `<vim_method>`, `<n>`, `<seed>`, `<meta_learner>`, `<model>`) depend on the parameters.

The array is of shape $D\times P\times K$ where,
 - $D$ is the number of covariates
  - $P$ is the number of permutations (set to 1 for LOCO)
  - $K$ is the number of splits for the variable importance estimation.
#### LOCO Method

For LOCO, if the model is `linear` and `return_coefs=True`:
- `<dgp>_<vim_method>_<n>_<seed>_<meta_learner>_<model>_coefs.npy` contains the weights of the model trained on the full set of covariates.
The array is of shape $D\times D \times K$ where,
 - $D$ is the number of covariates
 - $K$ is the number of splits for the variable importance estimation.
 The array is redundant in the first dimension. 
- `<dgp>_<vim_method>_<n>_<seed>_<meta_learner>_<model>_coefs_j.npy` contains the weights of the model trained on the subset excluding covariate `j`.
The array is of shape $D\times D-1 \times K$ where,
 - $D$ is the number of covariates
 - $K$ is the number of splits for the variable importance estimation.
 Along the first dimension, for $j=1, \cdots, D$ , the array contains the weights of the model trained on the subset of $D-1$ covariates excluding the covariate `j`.

#### PermuCATE Method
For PermuCATE, if `return_coefs=True`:
- `<dgp>_<vim_method>_<n>_<seed>_<meta_learner>_<model>_nu_j.npy` contains the estimation of covariate `j` from the `j` other covariates.
The array is of shape $D\times N\times K$ where,
 - $D$ is the number of covariates
 - $N$ is number of samples in the test set (per split)
 - $K$ is the number of splits for the variable importance estimation.

 Two example files are provided, `config.yaml` allows to reproduce the Figure 1 and `config_highdim.yaml` allows to reproduce the Figure 3.

## Variance Comparison
| Script Name         | `variance_exp.py`                |
|---------------------|----------------------------------|
| Related Experiments | Figure 2                         | 
| Short Description   | Compute the variance of the importance scores |

### Usage 
```bash
python ./variance_exp.py \
        --scenario linear               # --scenario: Type of scenario ['linear', 'nonlinear']
        --model_name TabPFN             # --model_name: Model to use ['TabPFN', 'CateNet', 'linear']
        --n_jobs 1                      # --n_jobs: Number of parallel jobs
        --n_seeds 10                    # --n_seeds: Number of random seeds
        --seed_offset 0                 # --seed_offset: Offset for the random seed. Used to parallelize the experiments using SLURM job array.
        --output_dir ./results_variance # --output_dir: Directory for output results
```

## Variance Comparison non Gaussian
| Script Name         | `variance_exp_non_gaussian.py`                |
|---------------------|----------------------------------|
| Related Experiments | Figure S5, S6                         |
| Short Description   | Compute the variance of the importance scores for non Gaussian data |

### Usage 
Same as `variance_exp.py`. 

## IHDP experiment
| Script Name         | `ihdp_exp.py`                |
|---------------------|------------------------------|
| Related Experiments | Figure 4 |
| Short Description   | Compute the variable importance on the IHDP dataset |

### Usage 
```bash
python ./ihdp_exp.py \
    --outdir ./results_ihdp         # Directory for output results
    --n_jobs 1                      # Number of parallel jobs
    --n_seeds 10                    # Number of random seeds
    --model_name CateNet            # Model to use ['CateNet', 'CF', 'TabPFN']
    --seed_offset 0                 # Offset for the random seed. Used to parallelize the experiments using SLURM job array.
```
