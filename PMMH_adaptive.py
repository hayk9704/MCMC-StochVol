import os
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from particle_filter import SMC
from MCMC_functions import log_prior, theta_to_x as xstart, stochvol



"""
This code applies the correlated PMMH algorithm to the SV model - proposal with adaptive jumps
set rho = 0 to get the standard (non-correlated) PMMH with adaptive jumps
"""


def PMCMC(ys, N_mcmc = 20000, x_first = xstart(), s = 2.38**2/3, m_latent = 50, burnin = 2000, rho = 0.9):
    U_old = np.random.normal(size = len(ys)*m_latent + len(ys)).reshape(len(ys), m_latent+1)        # first standard normal vector dims (T, m+1)
    xold = x_first

    # specific to adaptive jumps
    mean = xold                                                             # initializing the mean
    CC = np.zeros((3,3))                                                    # initializing empirical covariance matrix
    eps = 1e-6                                                              # the regularization term to make sure cov is positive definite
    covlist = []
    chol = np.linalg.cholesky(s*np.eye(3))

    
    draws = np.full((N_mcmc,3), np.nan) 
    
    loglik0 = SMC(y = ys, x = x_first, m_latent = m_latent, U = U_old)
    draws[0] = xold
    
    oldloglik = loglik0 + log_prior(xold)
    acc_all = 0                                                             # to count the number of acceptances

    for i in range(N_mcmc-1):
        covlist.append(CC)
        xnew = xold + chol@np.random.normal(size=3)
        U_new = rho*U_old + np.sqrt(1-rho**2)*np.random.normal(size=U_old.shape)
        
        loglik = SMC(y = ys, x = xnew, m_latent = m_latent, U = U_new)
        newloglik = loglik + log_prior(xnew)
        
        acc = newloglik - oldloglik                                         # the acceptance ratio
        
        #uniform draw to approve acceptance
        u = np.random.uniform()
        if np.log(u) < acc:
            xold = xnew
            oldloglik = newloglik
            U_old = U_new
            acc_all = acc_all + 1
        CC = CC + (1/(i+1))*(np.outer(xold - mean, xold - mean) - CC)       # updating empirical covariance
        mean = mean + (1/(i+1))*(xold - mean)                               # updating empirical mean
        cov = s*CC + eps*np.eye(3)                                          # using empirical covariance to get a cov. for the jump
        chol = np.linalg.cholesky(cov)
        draws[i+1] = xold
        if i%1000 == 0:
            print(f"MCMC iteration {i}")

    mu_draws = draws[:,0]
    mu_burnin_draws = mu_draws[burnin:]
    sigma2_draws = np.exp(draws[:,1])
    sigma2_burnin_draws = sigma2_draws[burnin:]
    phi_draws = (np.exp(draws[:,2])-1)/(1+np.exp(draws[:,2]))
    phi_burnin_draws = phi_draws[burnin:]
    acc_ratio = acc_all/N_mcmc

    # credible intervals
    lo_mu, hi_mu = np.quantile(mu_burnin_draws, [0.025, 0.975])
    lo_sigma2, hi_sigma2 = np.quantile(sigma2_burnin_draws, [0.025, 0.975])
    lo_phi, hi_phi = np.quantile(phi_burnin_draws, [0.025, 0.975])
    pars_95 = {"mu": (lo_mu, hi_mu),
               "sigma2_eta": (lo_sigma2, hi_sigma2),
               "phi": (lo_phi, hi_phi)}
    return {"mu_draws": mu_draws,
            "sigma2_draws": sigma2_draws,
            "phi_draws":phi_draws,
            "acc_ratio": acc_ratio,
            "mu_burnin_draws": mu_burnin_draws,
            "sigma2_burnin_draws": sigma2_burnin_draws,
            "phi_burnin_draws": phi_burnin_draws,
            "covlist": covlist,
            "pars_95": pars_95}



    # run this only if the code is run directly
if __name__ == "__main__":

    theseed = 121

    m_latent = 25
    s = 2.38**2/3

    rho = 0.99
    T = 700

    real_pars = {"mu": -0.86, "sigma2_eta": 0.0225, "phi": 0.98}

    stochvol.generate(mu = real_pars["mu"], phi = real_pars["phi"], sigma2_eta = real_pars["sigma2_eta"], T = T, seed = theseed)
    y_gen = stochvol.ys
    h_gen = stochvol.hs
    values = PMCMC(ys = y_gen, N_mcmc = 8000, x_first = xstart(), s = s, m_latent = m_latent, burnin = 2000, rho = rho)
    mu_mean = np.mean(values["mu_burnin_draws"])
    sigma2_mean = np.mean(values["sigma2_burnin_draws"])
    phi_mean = np.mean(values["phi_burnin_draws"])
    acc_ratio = values["acc_ratio"]
    print(f" for m = {m_latent}, s = {s}, and seed = {theseed}: \nmu_mean: {mu_mean} \nsigma2_mean: {sigma2_mean} \nphi_mean: {phi_mean} \nacc ratio: {acc_ratio}")


    fig, axes = plt.subplots(2,3, figsize = (16, 10))
    axes = axes.ravel()
    fig.suptitle(f"m = {m_latent}, s = {s}, rho = {rho}, T = {T}")

    az.plot_autocorr(values["mu_burnin_draws"], ax=axes[0])
    axes[0].set_title("Autocorrelation of mu")
    az.plot_autocorr(values["sigma2_burnin_draws"], ax=axes[1])
    axes[1].set_title("Autocorrelation of sigma2")
    az.plot_autocorr(values["phi_burnin_draws"], ax=axes[2])
    axes[2].set_title("Autocorrelation of phi")
    axes[3].plot(values["mu_draws"], linewidth = 0.75)
    axes[3].set_title("mu plot")
    axes[4].plot(values["sigma2_draws"], linewidth = 0.75)
    axes[4].set_title("sigma2 plot")
    axes[5].plot(values["phi_draws"], linewidth = 0.75)
    axes[5].set_title("phi plot")
    plt.show()

# set up the directory

"""


code to check the convergence of the var covar:

covlist = values["covlist"]
muchain = np.full(len(covlist),np.nan)
for i in range(len(covlist)):
    mu = covlist[i][0,0]
    muchain[i] = mu
plt.plot(muchain)

"""
