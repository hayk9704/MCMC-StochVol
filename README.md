# MCMC Methods for Stochastic Volatility Models

This repository contains the Python implementation of several pseudo-marginal Markov chain Monte Carlo (MCMC) algorithms applied to a standard stochastic volatility (SV) model. The project compares the efficiency of different sampling strategies, including Particle Markov Chain Monte Carlo (PMMH) and MCMC with importance sampling (PM-IS), each with enhancements like correlated proposals and adaptive jumps.

## Table of Contents
* [Overview](#-overview)
* [Features](#-features)
* [How to Use](#-how-to-use)
* [Technical File Descriptions](#-technical-file-descriptions)
* [Dependencies](#-dependencies)

## Overview

The core objective of this project is to estimate the parameters of a stochastic volatility model, which is crucial for understanding and forecasting risk in financial time series. Standard MCMC methods are often inefficient for such models due to the intractable likelihood function. This repository explores advanced pseudo-marginal methods that overcome this challenge by approximating the likelihood.

The project evaluates four main PMMH algorithms based on two key dimensions:
1.  **Proposal Mechanism**: Standard diagonal Gaussian proposal vs. an adaptive proposal that learns the covariance structure of the posterior.
2.  **Randomness**: Standard (independent) proposals vs. correlated proposals that reduce the variance of the likelihood estimator.

## Features

* **Stochastic Volatility Model**: Generation of synthetic financial data based on a standard SV model.
* **Particle Filter**: An efficient particle filter with systematic resampling to estimate the log-likelihood of the SV model.
* **PMMH Implementation**: Particle MCMC with standard and adaptive proposals.
* **PM-IS Implementation**: An alternative pseudo-marginal approach using importance sampling to estimate the likelihood.
* **Correlated Proposals**: Implementation of correlated auxiliary random variables to improve sampler efficiency.
* **Performance Comparison**: Main scripts (`PMMH_main_full.py`, `PM_IS_main_full.py`) to run all algorithm variants, compare their performance, and save results to a CSV file.

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
3.  Run the script. It will calculate the average of all statistics across your simulation runs and save them into a new file named `full_T200.csv`.

## Technical File Descriptions

This section provides a more detailed, technical breakdown of each Python script in the repository.

#### `MCMC_functions.py`
This is a utility script containing core components of the stochastic volatility model.
* **Parameter Transformations**: Includes functions `x_to_theta` and `theta_to_x` to transform parameters between the natural space (e.g., `phi`) and the estimation space (e.g., `log[(1+phi)/(1-phi)]`) for unconstrained optimization.
* **Log Prior**: The `log_prior` function calculates the log-prior probability of the parameters based on standard priors: a Normal prior for `mu`, a Beta prior for `phi`, and a Gamma prior for `sigma2_eta`.
* **Data Generation**: The `stochvol` class contains a `generate` method to produce synthetic time-series data (`ys`) and the corresponding latent log-volatility states (`hs`).

#### `particle_filter.py`
This file implements a particle filter using Sequential Monte Carlo (SMC) to estimate the log-likelihood of the SV model, which is required by the PMMH algorithms.
* **`SMC` function**: The main function that iterates through time, calling the particle propagation and resampling steps to compute the total log-likelihood for a given set of parameters (`x`) and random numbers (`U`).
* **`systematic_resample`**: An efficient resampling function that helps mitigate particle degeneracy.
* **`stable_log_pdf`**: A numerically stable function to calculate the log probability density function of the observations, avoiding underflow issues with `np.exp()`.

#### `PMMH.py` and `PMMH_adaptive.py`
These scripts implement the Particle Markov Chain Monte Carlo (PMMH) algorithm. The likelihood is estimated using the particle filter from `particle_filter.py`.
* **`PMMH.py`**: Implements PMMH with a standard diagonal Gaussian proposal for the parameters.
* **`PMMH_adaptive.py`**: Implements PMMH with an adaptive proposal mechanism. It learns the empirical covariance of the posterior samples during the chain and uses this information to propose more efficient jumps.
* **Correlated Proposals**: Both scripts can implement correlated PMMH by setting the `rho` parameter close to 1. This uses correlated random numbers in the particle filter at each MCMC step to reduce the variance of the likelihood estimate. Setting `rho = 0` yields the standard, non-correlated version.

#### `PM_IS.py` and `PM_IS_adaptive.py`
These scripts implement a pseudo-marginal MCMC algorithm where the likelihood is estimated using Importance Sampling (IS) instead of a particle filter.
* **`latent` function**: Generates `m_latent` draws of the entire latent volatility path `h` based on the model parameters.
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


---------------------------------------------------------------
# MCMC for Stochastic Volatility Models

This repository holds the code for my MSc dissertation project. I wanted to explore how different pseudo-marginal MCMC methods perform when applied to a standard stochastic volatility (SV) model.

The main challenge with SV models is that their likelihood is tough to calculate directly, which makes standard MCMC tricky. The algorithms here get around that by using particle methods to approximate the likelihood. I've implemented a few variations to see which ones are most efficient.

## 📝 Table of Contents
* [What's Inside](#-whats-inside)
* [Getting Started](#-getting-started)
* A Look at the Code
* [Dependencies](#-dependencies)

## ✨ What's Inside

This project compares a few different samplers, mainly focusing on two flavors of Particle MCMC (PMMH and PM-IS). I've tinkered with them in two main ways:
1.  **Proposals**: Using a simple, standard random-walk proposal vs. an adaptive one that learns the shape of the posterior as it runs.
2.  **Randomness**: Using fresh random numbers for each step vs. using correlated random numbers, which can help reduce the noise in the likelihood estimate.

This gives a total of four main algorithms for each "flavor" to compare.

## 🚀 Getting Started

Here’s how you can run the simulations yourself.

### 1. Run a Full Simulation

Your best starting point is one of the `_main_full.py` scripts.

* To check out the **Particle MCMC (PMMH)** methods, just run `PMMH_main_full.py`.
* To see the **Importance Sampling (PM-IS)** versions, run `PM_IS_main_full.py`.

### 2. What to Expect

When you run one of these main scripts, here's what it'll do:
1.  It'll cook up some synthetic financial data using the SV model defined in `MCMC_functions.py`.
2.  It will then run the four different MCMC chains, one after the other. You'll see progress updates printed in the console.
3.  Once it's done, a Matplotlib window will pop up showing the trace plots for the model parameters. This is a great way to visually check how well each sampler is mixing.
4.  Finally, it will spit out a detailed CSV file named something like `seed_[...]_T_[...]_results.csv`. This file has all the juicy stats: acceptance ratios, Effective Sample Size (ESS), bias, runtime, etc., for each method.

### 3. Combining Results

If you run the simulation multiple times, you'll end up with a bunch of CSV files. I wrote `extract_stats.py` to help with that.
1.  Toss all your result CSVs into one folder.
2.  Open `extract_stats.py` and change the `folder` variable to point to your folder's path.
3.  Run the script. It will average the results from all your runs and save them in a new `full_T200.csv` file.

## 🛠️ A Look at the Code

For anyone who wants to dig into the details, here’s a breakdown of what each script does.

#### `MCMC_functions.py`
This is the utility belt for the project. It has all the basics for the SV model:
* Functions to transform parameters back and forth (`x_to_theta`, `theta_to_x`) for easier sampling.
* The `log_prior` function, which defines the priors for the model parameters.
* A `stochvol` class with a `generate` method to create the synthetic data for the simulations.

#### `particle_filter.py`
This is the engine for the PMMH methods. It implements a Sequential Monte Carlo (SMC) algorithm, aka a particle filter, to estimate the log-likelihood.
* It uses systematic resampling to avoid the issue where only a few particles have all the weight.
* I included a `stable_log_pdf` function to prevent numerical errors when calculating densities with very small variances.

#### `PMMH.py` & `PMMH_adaptive.py`
These are the PMMH samplers. They use the particle filter to get the likelihood estimate at each step.
* `PMMH.py` uses a simple diagonal Gaussian proposal.
* `PMMH_adaptive.py` uses a smarter proposal that adapts based on the empirical covariance of the samples it has already drawn.
* Both can be switched to their "correlated" versions by setting the `rho` parameter to a value close to 1.

#### `PM_IS.py` & `PM_IS_adaptive.py`
These are similar to the PMMH scripts but use a simpler Importance Sampling (IS) approach to estimate the likelihood instead of a full particle filter.
* The `latent` function generates batches of the entire volatility path, and the `log_lik` function averages them to get the estimate.
* Just like the PMMH scripts, this pair comes in a standard (`PM_IS.py`) and an adaptive (`PM_IS_adaptive.py`) version, both with the `rho` parameter for correlation.

#### `m_latent_check.py`
This is a diagnostic tool I used to figure out how many particles (`m_latent`) were "enough". The goal is to get the variance of the log-likelihood estimate to a target level (around 1.0 for the standard methods). This script runs a test to check that variance for a given number of particles.

## 📦 Dependencies
To get this running, you'll need the usual suspects from the Python data science world:
* NumPy
* SciPy
* Pandas
* Matplotlib
* ArviZ

You can install them all with pip:
`pip install numpy scipy pandas matplotlib arviz`
