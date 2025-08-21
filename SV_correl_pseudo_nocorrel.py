import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from MCMC_functions import theta_to_x, log_prior, stochvol
# load the generated returns and log volatility



# the actual parameters (to be changed)
# mu = -0.86
# phi = 0.98
# sigma_eta = 0.15 -> sigma2_eta = 0.225

# the joint likelihood funtion. x is the vector of parameters.

# a 3 dimensional vector x to start the chain:

# xstart - to transform mu -> mu, phi -> log[(1+phi)/(1-phi)], sigma2_eta -> log(sigma2_eta) and store in x:
# latent inputs the seed vector U(size =(y.size, m_latent)), U~N(0,1) and makes latent variables.
def latent(x, U):
    # x is 3 dimensional: mu, log(sigma2_eta) and log[(1+phi)/(1-phi)]
    mu = x[0]
    sigma2_eta = np.exp(x[1])
    phi = (np.exp(x[2])-1)/(1+np.exp(x[2]))
    # matrix H - Txm, each row i has m draws of latent variable h_i
    # initiazize matrix H - T by m - each clumn is a new draw vector h_1..T 
    H = np.full((U.shape), np.nan)
    H[0] = mu + np.sqrt(sigma2_eta/(1-phi**2))*U[0]
    for i in range(H.shape[0]-1):
        H[i+1] = mu + phi*(H[i] - mu) + np.sqrt(sigma2_eta)*U[i+1]
    return H


# given matrix H containing diff draws of latent variables as columns, calculate the loglik
def log_lik(H, y): 
    m = H.shape[1]
    Y = np.tile(y[:, None], (1, m)) # matrix Y, each column j is the same vector of obs. y1..T

    Y_pdf = st.norm.logpdf(Y, 0, np.exp(H/2)) # calculate the individual pdfs of y_tj conditional on h_tj
    pdfs = np.sum(Y_pdf, axis = 0)
    # doing the log sum exp trick to avoid numerical underflow
    max_pdf = np.max(pdfs)
    loglik = np.log(np.sum(np.exp(pdfs - max_pdf))) + max_pdf + np.log(1/H.shape[1])
    return loglik


#log_prior + log_lik = log_post
def log_post(x, latent, y):
    return log_prior(x) + log_lik(H = latent, y =y)



# main correlated MCMC chain

def correl_pseudo_MCMC(ys, rho = 0.8, N_mcmc = 20000, x_first = theta_to_x(), x_std = 0.01, m_latent = 50, burnin = 2000):
    Uold = st.norm.rvs(size = (ys.size, m_latent)) # the first seed of standard normals
    draws = np.full((N_mcmc,3), np.nan) 
    draws[0] = x_first
    xold = x_first
    L = latent(xold, Uold)
    oldlik = log_post(xold, latent = L, y=ys)
    acc_all = 0 # to count the number of acceptances

    for i in range(N_mcmc-1):
        xnew = xold + x_std*st.norm.rvs(size = 3)
        Unew = rho*Uold + np.sqrt(1 - rho**2)*st.norm.rvs(size = Uold.shape)
        Lnew = latent(xnew, Unew)
        newlik = log_post(xnew, latent = Lnew, y=ys)
        # let's define the acceptance ratio
        acc = newlik - oldlik 
        
        #uniform draw to approve acceptance
        u = st.uniform.rvs()
        if np.log(u) < acc:
            xold = xnew
            oldlik = newlik
            Uold = Unew
            acc_all = acc_all + 1
        draws[i+1] = xold
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
    real_pars = {"mu": -0.86, "sigma2_eta": 0.025, "phi": 0.98}
    prior_pars = {"mu_mean": -0.86, "mu_var" : 0.01,"sigma2_mean" : 0.025, "sigma2_var" : 0.01, "phi_mean" : 0.98, "phi_var" : 0.01}

    stochvol.generate(mu = real_pars["mu"], phi = real_pars["phi"], sigma2_eta = real_pars["sigma2_eta"], T = 10000, seed = 101)
    y_gen = stochvol.ys[-150:]
    values = correl_pseudo_MCMC(ys = y_gen, N_mcmc = 40000, m_latent = 500, burnin = 10000, x_std = 1.4, rho = 0) 
    mu_mean = np.mean(values["mu_draws"][1000:])
    sigma2_mean = np.mean(values["sigma2_draws"][1000:])
    phi_mean = np.mean(values["phi_draws"][1000:])
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


