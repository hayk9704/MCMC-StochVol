import os
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from MCMC_functions import theta_to_x as xstart, stochvol, log_prior
from scipy.special import logsumexp
import arviz as az



"""
This code applies the correlated PM-IS algorithm to the SV model - diagonal gaussian proposal
set rho = 0 to get the standard (non-correlated) PM-IS
"""


# Input - T x m matrix of auxiliary RV - U and pars x ----- Output - T x m matrix H of latent variables
def latent(x, U):
                                                                    # x is 3 dimensional: mu, log(sigma2_eta) and log[(1+phi)/(1-phi)]
    mu = x[0]
    sigma2_eta = np.exp(x[1])
    phi = (np.exp(x[2])-1)/(1+np.exp(x[2]))
    

    H = np.full((U.shape), np.nan)                                  # matrix H - Txm, each row i has m draws of latent variable h_i/each clumn is a new draw vector h_1..T 
    H[0] = mu + np.sqrt(sigma2_eta/(1-phi**2))*U[0]
    for i in range(H.shape[0]-1):
        H[i+1] = mu + phi*(H[i] - mu) + np.sqrt(sigma2_eta)*U[i+1]
    return H


# given matrix H containing diff draws of latent variables as columns, calculate the loglik
def log_lik(H, y): 
    m = H.shape[1]
    Y = np.tile(y[:, None], (1, m))                                 # matrix Y, each column j is the same vector of obs. y1..T

    Y_pdf = st.norm.logpdf(Y, 0, np.exp(H/2))                       # calculate the individual pdfs of y_tj conditional on h_tj
    pdfs = np.sum(Y_pdf, axis = 0)
   
    loglik = logsumexp(pdfs) - np.log(m)                            # doing the log sum exp trick to avoid numerical underflow
    return loglik


#log_prior + log_lik = log_post
def log_post(x, latent, y):
    return log_prior(x) + log_lik(H = latent, y =y)



# main correlated MCMC chain
def PM_IS(ys, N_mcmc = 20000, x_first = xstart(), s = 2.38**2/3, m_latent = 50, burnin = 2000, rho = 0.9):
    Uold = st.norm.rvs(size = (ys.size, m_latent))                  # the first seed of standard normals
    draws = np.full((N_mcmc,3), np.nan) 
    draws[0] = x_first
    xold = x_first

    Hold = latent(xold, Uold)
    oldlik = log_post(xold, latent = Hold, y=ys)
    acc_all = 0                                                     # to count the number of acceptances

    for i in range(N_mcmc-1):
        xnew = xold + np.sqrt(s)*st.norm.rvs(size = 3)
        Unew = rho*Uold + np.sqrt(1 - rho**2)*st.norm.rvs(size = Uold.shape)
        Hnew = latent(xnew, Unew)
        newlik = log_post(xnew, latent = Hnew, y=ys)
        # let's define the acceptance ratio
        acc = newlik - oldlik 
        
        # uniform draw to approve acceptance
        u = st.uniform.rvs()
        if np.log(u) < acc:
            xold = xnew
            oldlik = newlik
            Uold = Unew
            acc_all = acc_all + 1
        draws[i+1] = xold

        if i%1000 == 0:
            print(f"iteration {i}")

    mu_draws = draws[:,0]
    mu_burnin_draws = mu_draws[burnin:]
    sigma2_draws = np.exp(draws[:,1])
    sigma2_burnin_draws = sigma2_draws[burnin:]
    phi_draws = (np.exp(draws[:,2])-1)/(1+np.exp(draws[:,2]))
    phi_burnin_draws = phi_draws[burnin:]
    acc_ratio = acc_all/N_mcmc

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
            "pars_95": pars_95}




    # run this only if the code is run directly
if __name__ == "__main__":

    theseed = 101
    
    m_latent = 1000
    s = 0.25
    rho = 0
    T = 200

    real_pars = {"mu": -0.86, "sigma2_eta": 0.0225, "phi": 0.98}

    stochvol.generate(mu = real_pars["mu"], phi = real_pars["phi"], sigma2_eta = real_pars["sigma2_eta"], T = T, seed = theseed)
    y_gen = stochvol.ys
    h_gen = stochvol.hs
    values = PM_IS(ys = y_gen, N_mcmc = 8000, x_first = xstart(), s = s, m_latent = m_latent, burnin = 2000, rho = rho)
    mu_mean = np.mean(values["mu_burnin_draws"])
    sigma2_mean = np.mean(values["sigma2_burnin_draws"])
    phi_mean = np.mean(values["phi_burnin_draws"])
    acc_ratio = values["acc_ratio"]
    print(f"for m = {m_latent}, s = {s}, and seed = {theseed}: \nmu_mean: {mu_mean} \nsigma2_mean: {sigma2_mean} \nphi_mean: {phi_mean} \nacc ratio: {acc_ratio}")


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


