# MCMC Methods for Stochastic Volatility Models

This repository contains the Python implementation of several pseudo-marginal Markov chain Monte Carlo (MCMC) algorithms applied to a standard stochastic volatility (SV) model. The project compares the efficiency of different sampling strategies, including Particle Markov Chain Monte Carlo (PMMH) and MCMC with importance sampling (PM-IS), each both with and without enhancements like correlated proposals and adaptive jumps.

## Table of Contents

  * [Overview](#-Overview)
  * [How to Use](#-How-to-Use)
  * [Technical File Descriptions](#-Technical -File-Descriptions)
  * [Dependencies](#-Dependencies)

## Overview

The core objective of this project is to estimate the parameters of a stochastic volatility model, which is crucial for understanding and forecasting risk in financial time series. Standard MCMC methods are often inefficient for such models due to the intractable likelihood function. This repository explores advanced pseudo-marginal methods that overcome this challenge by approximating the likelihood.

The project evaluates four variations of the two main pseudo-marginal MCMC algorithms (PM-IS and PMMH) based on two key dimensions:

1.  **Proposal Mechanism**: Standard diagonal Gaussian proposal vs. an adaptive proposal that learns the covariance structure of the posterior.
2.  **Randomness**: Standard (independent) proposals vs. correlated proposals that reduce the variance of the likelihood estimator.

## How to Use

This guide explains how to run the MCMC simulations and analyze the results.

### 1\. Configure and Run the Simulations

All simulations are run from the `main.py` script. To configure a run, open `main.py` and modify the parameters within the `if __name__ == "__main__":` block at the bottom of the file.

  * **Choose the method**: Set the `method` variable to run the desired algorithms. Options are:
      * `"PMMH"`: Runs all four PMMH variations (standard/adaptive, correlated/non-correlated).
      * `"PM_IS"`: Runs all four PM-IS variations.
      * `"PMMH_adapt"`: Runs only the adaptive PMMH variations.
      * `"both"`: Runs all eight algorithm variations.
  * **Set simulation parameters**: Adjust `T_obs` (number of observations), `N_mcmc` (number of MCMC iterations), `burnin`, and other model-specific hyperparameters in the same section.
  * **Execute the script**: Run `python main.py` from your terminal.

### 2\. What to Expect

When you run `main.py`:

1.  A synthetic dataset is generated based on the `real_pars` and `T_obs` settings.
2.  The selected MCMC algorithms run sequentially. The script will print the progress and time taken for each chain.
3.  The output chains are saved as individual `.csv` files inside a new folder named `MCMC_chains`. Each filename includes the algorithm type, number of observations, and a unique seed, for example: `PMMH_adapt_correl_T_700_3141592653_12345.csv`.

### 3\. Analyze the Results

The `extract_stats.py` script is used to process the output chains for a **single model configuration** (e.g., all runs for PMMH with diagonal jumps and no correlation for T=700).

1.  **Configure the script**: Open `extract_stats.py` and modify the variables in the **choose model specifications** section to match the chains you want to analyze. For example:
    ```python
    T = 700
    MCMC_type = "PMMH"
    jumps = "diag"
    corr = "nocorrel"
    ```
2.  **Run the analysis**: Execute `python extract_stats.py`.
3.  **Check the output**: The script will:
      * Load all matching chain files from the `MCMC_chains` folder.
      * Calculate convergence diagnostics (like ESS and R-hat), bias, and average runtime.
      * Save these statistics to a new CSV file, e.g., `diagnostics_PMMH_T_700_nocorrel_diag.csv`.
      * Generate and save rank plots (`.png` and `.pdf`) to visually assess chain convergence.

## Technical File Descriptions

This section provides a more detailed, technical breakdown of each Python script in the repository.

#### `main.py`

This is the main executable script for running the simulations. It orchestrates the entire process by generating data, calling the appropriate MCMC algorithm functions based on user configuration, timing the execution, and saving the resulting posterior draws for each chain into a separate CSV file in the `MCMC_chains` directory.

#### `extract_stats.py`

This script is used for post-simulation analysis. It loads a group of saved MCMC chains corresponding to a specific algorithm configuration (e.g., PMMH, adaptive, correlated). It then computes key performance and convergence metrics, including Effective Sample Size (ESS), potential scale reduction factor (R-hat), parameter bias, and acceptance ratios. It saves these diagnostics in a summary CSV file and also generates rank plots to visually inspect the convergence of the chains.

#### `MCMC_functions.py`

This is a utility script containing core components of the stochastic volatility model.

  * **Parameter Transformations**: Includes functions `x_to_theta` and `theta_to_x` to transform parameters between the natural space (e.g., `phi`) and the estimation space (e.g., $log[(1+\\phi)/(1-\\phi)]$) for unconstrained optimization.
  * **Log Prior**: The `log_prior` function calculates the log-prior probability of the parameters based on standard priors: a Normal prior for `mu`, a Beta prior for `phi`, and a Gamma prior for `sigma2_eta`.
  * **Data Generation**: The `stochvol` class contains a `generate` method to produce synthetic time-series data (`ys`) and the corresponding latent log-volatility states (`hs`).

#### `particle_filter.py`

This file implements a particle filter using Sequential Monte Carlo (SMC) to estimate the log-likelihood of the SV model, which is required by the PMMH algorithms.

  * **`SMC` function**: The main function that iterates through time, calling the particle propagation and resampling steps to compute the total log-likelihood.
  * **`systematic_resample`**: An efficient resampling function that helps reduce the Monte Carlo variation.
  * **`stable_log_pdf`**: A numerically stable function to calculate the log probability density of the observations, avoiding underflow issues.

#### `PMMH.py` and `PMMH_adaptive.py`

These scripts implement the Particle Markov Chain Monte Carlo (PMMH) algorithm. The likelihood is estimated using the particle filter from `particle_filter.py`.

  * **`PMMH.py`**: Implements PMMH with a standard diagonal Gaussian proposal for the parameters.
  * **`PMMH_adaptive.py`**: Implements PMMH with an adaptive proposal. It learns the empirical covariance of the posterior samples and uses this to propose more efficient jumps.
  * **Correlated Proposals**: Both scripts can implement correlated PMMH by setting the `rho` parameter close to 1. Setting `rho = 0` yields the standard, non-correlated version.

#### `PM_IS.py` and `PM_IS_adaptive.py`

These scripts implement a pseudo-marginal MCMC algorithm where the likelihood is estimated using Importance Sampling (IS) instead of a particle filter.

  * **`latent` function**: Generates draws of the entire latent volatility path `h`.
  * **`log_lik` function**: Calculates the log-likelihood estimate from the generated latent paths using the log-sum-exp trick for numerical stability.
  * Like their PMMH counterparts, these files implement both a standard diagonal proposal (`PM_IS.py`) and an adaptive proposal (`PM_IS_adaptive.py`), with the option for correlated randomness via the `rho` parameter.

#### `m_latent_check.py`

A diagnostic script used to calibrate the number of particles (`m_latent`) needed for the likelihood estimators. It implements the methodology proposed in the literature: for standard pseudo-marginal methods, the variance of the log-likelihood estimate should be approximately 1. For correlated methods, the variance of the *difference* between consecutive log-likelihood estimates should be around 1.

## Dependencies

This project requires the following Python libraries:

  * NumPy
  * SciPy
  * Pandas
  * Matplotlib
  * ArviZ

You can install them via pip:
`pip install numpy scipy pandas matplotlib arviz`
