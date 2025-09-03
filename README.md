# MCMC Methods for Stochastic Volatility Models

This repository contains the Python implementation of several pseudo-marginal Markov chain Monte Carlo (MCMC) algorithms applied to a standard stochastic volatility (SV) model. The project compares the efficiency of different sampling strategies, including Particle Markov Chain Monte Carlo (PMMH) and MCMC with importance sampling (PM-IS), each both with and without enhancements like correlated proposals and adaptive jumps.

## Table of Contents
* [Overview](#-overview)
* [How to Use](#-how-to-use)
* [Technical File Descriptions](#-technical-file-descriptions)
* [Dependencies](#-dependencies)

## Overview

The core objective of this project is to estimate the parameters of a stochastic volatility model, which is crucial for understanding and forecasting risk in financial time series. Standard MCMC methods are often inefficient for such models due to the intractable likelihood function. This repository explores advanced pseudo-marginal methods that overcome this challenge by approximating the likelihood.

The project evaluates four variations of the two main pseudo-marginal MCMC algorithms (PM-IS and PMMH) based on two key dimensions:
1.  **Proposal Mechanism**: Standard diagonal Gaussian proposal vs. an adaptive proposal that learns the covariance structure of the posterior.
2.  **Randomness**: Standard (independent) proposals vs. correlated proposals that reduce the variance of the likelihood estimator.

## How to Use

This guide explains how to run a full comparison of the different MCMC algorithms and generate results.

### 1. Generate Synthetic Data and Run a Simulation

The main scripts are designed to be run directly to compare the different methods.

* To compare the **Particle MCMC (PMMH)** methods, open and run the `PMMH_main_full.py` file.
* To compare the **Particle MCMC with Importance Sampling (PM-IS)** methods, open and run `PM_IS_main_full.py`.

### 2. What to Expect

When you run one of the main files (e.g., `PMMH_main_full.py`):
1.  A synthetic dataset of observations will be generated using the stochastic volatility model.
2.  Four different MCMC chains will run sequentially to estimate the model parameters:
    * Standard proposal, no correlation.
    * Standard proposal, with correlation.
    * Adaptive proposal, no correlation.
    * Adaptive proposal, with correlation.
3.  The script will print the progress and total time taken for each chain.
4.  Once completed, a plot will be displayed showing the trace plots for the parameters (`mu`, `sigma2_eta`, `phi`) from all four chains, allowing for a visual comparison of their mixing properties.
5.  A CSV file named `seed_[...]_T_[...]_results.csv` will be saved in your directory. This file contains a detailed performance summary, including acceptance ratios, effective sample size (ESS), bias, and total runtime for each algorithm.

### 3. Analyzing Results from Multiple Runs

If you run the main scripts multiple times (which generates multiple CSV files), you can use the `extract_stats.py` script to aggregate the results.
1.  Place all your result CSV files into a single folder.
2.  Update the `folder` path in `extract_stats.py` to point to that folder.
3.  Run the script. It will calculate the average of all statistics across your simulation runs and save them into a new file named `full_T_[...].csv`.

## Technical File Descriptions

This section provides a more detailed, technical breakdown of each Python script in the repository.

#### `MCMC_functions.py`
This is a script containing core components of the stochastic volatility model.
* **Parameter Transformations**: Includes functions `x_to_theta` and `theta_to_x` to transform parameters between the natural space (e.g., `phi`) and the estimation space (e.g., `log[(1+phi)/(1-phi)]`) for unconstrained optimization.
* **Log Prior**: The `log_prior` function calculates the log-prior probability of the parameters based on standard priors: a Normal prior for `mu`, a Beta prior for `phi`, and a Gamma prior for `sigma2_eta`.
* **Data Generation**: The `stochvol` class contains a `generate` method to produce synthetic time-series data (`ys`) and the corresponding latent log-volatility states (`hs`) for the specified parameter values ('mu, sigma2_eta, phi`) and sample length (`t`).

#### `particle_filter.py`
This file implements a particle filter using Sequential Monte Carlo (SMC) to estimate the log-likelihood of the SV model, which is required by the PMMH algorithms.
* **`SMC` function**: The main function that iterates through time, calling the particle propagation and resampling steps to compute the total log-likelihood for a given set of parameters (`x`) and random numbers (`U`).
* **`systematic_resample`**: An efficient resampling function to resample ancestor indices, using only one uniform random number in process.
* **`stable_log_pdf`**: A numerically stable function to calculate the log probability density function of the observations, avoiding underflow issues with `np.exp()`.

#### `PMMH.py` and `PMMH_adaptive.py`
These scripts implement the Particle Markov Chain Monte Carlo (PMMH) algorithm. The likelihood is estimated using the particle filter from `particle_filter.py`.
* **`PMMH.py`**: Implements PMMH with a standard diagonal Gaussian proposal for the parameters.
* **`PMMH_adaptive.py`**: Implements PMMH with an adaptive proposal mechanism. It learns the empirical covariance of the posterior samples during the chain and uses this information to propose more efficient jumps.
* **Correlated Proposals**: Both scripts can implement correlated PMMH by setting the `rho` parameter close to 1. This uses correlated random numbers in the particle filter at each MCMC step to reduce the variance of the likelihood estimate. Setting `rho = 0` yields the standard, non-correlated version.

#### `PM_IS.py` and `PM_IS_adaptive.py`
These scripts implement a pseudo-marginal MCMC algorithm where the likelihood is estimated using Importance Sampling (IS) instead of a particle filter.
* **`latent` function**: Generates draws of the entire latent volatility path `h` of length `m_latent`  based on the model parameters.
* **`log_lik` function**: Calculates the log-likelihood estimate from the generated latent paths using the log-sum-exp trick for numerical stability.
* Like their PMMH counterparts, these files implement both a standard diagonal proposal (`PM_IS.py`) and an adaptive proposal (`PM_IS_adaptive.py`), with the option for correlated randomness via the `rho` parameter.

#### `m_latent_check.py`
A diagnostic script used to calibrate the number of particles (`m_latent`) needed for the likelihood estimators.
* It implements the methodology proposed in the literature: for standard pseudo-marginal methods, the variance of the log-likelihood estimate should be approximately 1. For correlated methods, the variance of the *difference* between consecutive log-likelihood estimates should be around 1. This script runs simulations to find the `m_latent` that satisfies these conditions.

## Dependencies
This project requires the following Python libraries:
* NumPy
* SciPy
* Pandas
* Matplotlib
* ArviZ

You can install them via pip:
`pip install numpy scipy pandas matplotlib arviz`

