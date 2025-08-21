import numpy as np
import matplotlib.pyplot as plt
from SV_SMC_system_resample import SMC
from MCMC_functions import log_prior, theta_to_x as xstart, stochvol




def PMCMC(ys, N_mcmc = 20000, x_first = xstart(), s = 2.38**2/3, m_latent = 50, burnin = 2000):
    
    draws = np.full((N_mcmc,3), np.nan) 
    
    loglik0 = SMC(y = ys, x = x_first, m_latent = m_latent)
    draws[0] = x_first
    
    xold = x_first
    oldloglik = loglik0 + log_prior(xold)
    acc_all = 0 # to count the number of acceptances

    for i in range(N_mcmc-1):
        xnew = xold +np.sqrt(s)*np.random.normal(size = 3)
        loglik = SMC(y = ys, x = xnew, m_latent = m_latent)
        newloglik = loglik + log_prior(xnew)
        # let's define the acceptance ratio
        acc = newloglik - oldloglik
        
        #uniform draw to approve acceptance
        u = np.random.uniform()
        if np.log(u) < acc:
            xold = xnew
            oldloglik = newloglik
            acc_all = acc_all + 1
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
    return {"mu_draws": mu_draws,
            "sigma2_draws": sigma2_draws,
            "phi_draws":phi_draws,
            "acc_ratio": acc_ratio,
            "mu_burnin_draws": mu_burnin_draws,
            "sigma2_burnin_draws": sigma2_burnin_draws,
            "phi_burnin_draws": phi_burnin_draws}




    # run this only if the code is run directly
if __name__ == "__main__":
    real_pars = {"mu": -0.86, "sigma2_eta": 0.0225, "phi": 0.86}

    stochvol.generate(mu = real_pars["mu"], phi = real_pars["phi"], sigma2_eta = real_pars["sigma2_eta"], T = 500, seed = 103)
    y_gen = stochvol.ys
    h_gen = stochvol.hs
    values = PMCMC(ys = y_gen, N_mcmc = 10000, x_first = xstart(), s = 0.4, m_latent = 120, burnin = 2000)
    mu_mean = np.mean(values["mu_burnin_draws"])
    sigma2_mean = np.mean(values["sigma2_burnin_draws"])
    phi_mean = np.mean(values["phi_burnin_draws"])
    acc_ratio = values["acc_ratio"]
    print(f"mu_mean: {mu_mean} \nsigma2_mean: {sigma2_mean} \nphi_mean: {phi_mean} \nacc ratio: {acc_ratio}")
    plt.subplot(2,2,1) # plotting mu
    plt.plot(values["mu_draws"])

    plt.subplot(2,2,2) # plotting 
    plt.plot(values["sigma2_draws"])

    plt.subplot(2,2,3) # plotting phi
    plt.plot(values["phi_draws"])
    plt.tight_layout()
    plt.show()

# set up the directory

