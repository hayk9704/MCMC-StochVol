import numpy as np
import matplotlib.pyplot as plt
from particle_filter import SMC
from MCMC_functions import theta_to_x as xstart, stochvol
from PM_IS import latent, log_lik

"""
This code is to verify the number of particles of particles for PMMH and PM-IS algorithms

------------------------------for high posterior density areas----------------------------------
For standard - non-correlated versions the variance of the log likelihood must be appx 1
For correlated versions the variance of consecutive log likelihood differences must be appx 1
"""



# the function to verify the number of particles for particle filter loglik estimator (applied to PMMH)
def PMMH_m_check(ys, N_mcmc = 20000, x = xstart(), m_latent = 50, burnin = 2000, rho = 0.9):
    likdiffs = np.full((N_mcmc-1,), np.nan)                                 # to record the likelihood differences
    Lold = np.empty(N_mcmc-1, float)
    Lnew = np.empty(N_mcmc-1, float)

    U_old = np.random.normal(size = len(ys)*m_latent + len(ys)).reshape(len(ys), m_latent+1)
    oldloglik = SMC(y = ys, x = x, m_latent = m_latent, U = U_old)
    for i in range(N_mcmc -1):
        U_new = rho*U_old + np.sqrt(1-rho**2)*np.random.normal(size=U_old.shape)
        newloglik = SMC(y = ys, x = x, m_latent = m_latent, U = U_new)
        likdiffs[i] = newloglik - oldloglik
        Lold[i] = oldloglik
        Lnew[i] = newloglik

        oldloglik = newloglik
        U_old = U_new
        if i%100 == 0:
            print(f"iteration {i}")
    ld = likdiffs[burnin:]
    L1 = Lold[burnin:]
    L2 = Lnew[burnin:]
    var_dZ = np.var(ld, ddof=1)
    rho_L = np.corrcoef(L1, L2)[0,1]
    sigma2_hat = np.var(L1, ddof=1)

    print(f"Var(ΔZ): {var_dZ:.3f} | realized ρ_L: {rho_L:.3f} | "
          f"2σ²(1-ρ_L): {2*sigma2_hat*(1-rho_L):.3f} | "
          f"sigma2 (for standard PMMH): {sigma2_hat}")                      # 2σ²(1-ρ_L) and Var(ΔZ) must be close

    return ld, rho_L, sigma2_hat


# the function to verify the number of particles for importance sampling loglik estimator 
def IS_m_check(ys, N_mcmc = 20000, x = xstart(), m_latent = 50, burnin = 2000, rho = 0.9):
    likdiffs = np.full((N_mcmc-1,), np.nan)                                 # to record the likelihood differences
    Lold = np.empty(N_mcmc-1, float)
    Lnew = np.empty(N_mcmc-1, float)

    U_old = np.random.normal(size = len(ys)*m_latent).reshape(len(ys), m_latent)
    oldloglik = log_lik(latent(x, U_old), ys)
    for i in range(N_mcmc -1):
        U_new = rho*U_old + np.sqrt(1-rho**2)*np.random.normal(size=U_old.shape)
        newloglik = log_lik(latent(x, U_new), ys)
        likdiffs[i] = newloglik - oldloglik
        Lold[i] = oldloglik
        Lnew[i] = newloglik

        oldloglik = newloglik
        U_old = U_new
        if i%100 == 0:
            print(f"iteration {i}")
    ld = likdiffs[burnin:]
    L1 = Lold[burnin:]
    L2 = Lnew[burnin:]
    var_dZ = np.var(ld, ddof=1)
    rho_L = np.corrcoef(L1, L2)[0,1]
    sigma2_hat = np.var(L1, ddof=1)

    print(f"Var(ΔZ): {var_dZ:.3f} | realized ρ_L: {rho_L:.3f} | "
          f"2σ²(1-ρ_L): {2*sigma2_hat*(1-rho_L):.3f} | "
          f"sigma2 (for standard PM-IS): {sigma2_hat}")                      # 2σ²(1-ρ_L) and Var(ΔZ) must be close

    return ld, rho_L, sigma2_hat


if __name__ =="__main__":
    theseed = 111
    m_particles = 1000                                                       # number of particles to check
    T_obs = 200                                                              # number of obs.
    real_pars = {"mu": -0.86, "sigma2_eta": 0.0225, "phi": 0.98}
    stochvol.generate(mu = real_pars["mu"], phi = real_pars["phi"], sigma2_eta = real_pars["sigma2_eta"], T = T_obs, seed = theseed)
    y_gen = stochvol.ys
    par_vector = xstart(mu = -0.86, phi = 0.98, sigma2_eta = 0.0225)  


    # PMMH_m_check(ys = y_gen, N_mcmc = 500, x = par_vector, m_latent = m_particles, burnin = 100, rho = 0.99)

    IS_m_check(ys = y_gen, N_mcmc = 500, x = par_vector, m_latent = m_particles, burnin = 100, rho = 0.99)

    """
    ---------------T_obs = 200----------------
    for standard PM-IS - no. particles = 900
    for correlated PM-IS - no. particles = 300


    """
