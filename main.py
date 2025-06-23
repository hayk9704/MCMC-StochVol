import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from stoch_gen import stochvol
from SV_MALA import MALA_MCMC
from SV_plain_MCMC import plain_MCMC
from SV_corel_pseudo import correl_pseudo_MCMC, xstart

"""
all MCMC functions return dicts:
            "mu_draws": mu_draws,
            "sigma2_draws": sigma2_draws,
            "phi_draws":phi_draws,
            "acc_ratio": acc_ratio}
"""

real_pars = {"mu": -0.86, "sigma2_eta": 0.025, "phi": 0.98}
priors = {"mu_mean": -0.86, "mu_var" : 0.01,"sigma2_mean" : 0.025, "sigma2_var" : 0.01, "phi_mean" : 0.98, "phi_var" : 0.01}

# number of MCMC iterations
N= 5000

# the starting parameter values for the chain
x_0 = xstart(mu = -0.2, phi = 0.6, sigma2_eta = 0.02)

stochvol.generate(mu = real_pars["mu"], phi = real_pars["sigma2_eta"], sigma2_eta = real_pars["phi"], T = 100)
y_gen = stochvol.ys
h_gen = stochvol.hs

# plain MH chain - x_std - std for proposal
plain_chain = plain_MCMC(prior_pars = priors, ys = y_gen, hs = h_gen, N_mcmc = N, x_first = x_0,  x_std = 0.01)

# pseudo marginal MCMC - standard chain
pseudo_std_chain = correl_pseudo_MCMC(prior_pars = priors, ys = y_gen, correl = 0.8, N_mcmc = N, x_first = x_0, x_std = 0.01, m_latent = 5)

# pseudo marginal MCMC - correlated chain (correl = 0)
pseudo_correl_chain = correl_pseudo_MCMC(prior_pars = priors, ys = y_gen, correl = 0.8, N_mcmc = N, x_first = x_0, x_std = 0.01, m_latent = 50)

# The returned fig is your container—you can use it to adjust things globally 
# (titles that span multiple subplots, overall background color, saving the figure to disk, etc.).
fig, axes = plt.subplots(nrows=3,
                         ncols=1,
                         figsize=(8, 10),
                         sharex=True)

# setting the background color and axes color
fig.patch.set_facecolor('#f0f0f0')
axes[0].set_facecolor('white')
axes[1].set_facecolor('white')
axes[2].set_facecolor('white')


# chain for the mu parameters
axes[0].set_xlabel("mu")
axes[0].set_ylabel("Parameter value")
axes[0].plot(plain_chain["mu_draws"], color = 'tab:blue', label = "plain MCMC")
axes[0].plot(pseudo_std_chain["mu_draws"], color = 'tab:orange', label = "pseudo MCMC")
axes[0].plot(pseudo_correl_chain["mu_draws"], color = 'tab:green', label = "pseudo correl MCMC")
axes[0].legend()

axes[1].set_xlabel("sigma2_eta")
axes[1].set_ylabel("Parameter value")
axes[1].plot(plain_chain["sigma2_draws"], color = 'tab:blue', label = "plain MCMC")
axes[1].plot(pseudo_std_chain["sigma2_draws"], color = 'tab:orange', label = "pseudo MCMC")
axes[1].plot(pseudo_correl_chain["sigma2_draws"], color = 'tab:green', label = "pseudo correl MCMC")
axes[1].legend()


axes[2].set_xlabel("phi")
axes[2].set_ylabel("Parameter value")
axes[2].plot(plain_chain["phi_draws"], color = 'tab:blue', label = "plain_MCMC")
axes[2].plot(pseudo_std_chain["phi_draws"], color = 'tab:orange', label = "pseudo MCMC")
axes[2].plot(pseudo_correl_chain["phi_draws"], color = 'tab:green', label = "pseudo correl MCMC")
axes[2].legend()

plt.show()

# make sure to start over