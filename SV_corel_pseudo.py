import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt

# load the generated returns and log volatility
from stoch_gen import stochvol
stochvol.generate(mu = -0.86, phi = 0.98, sigma2_eta = 0.0225, T = 100)


# the actual parameters (to be changed)
# mu = -0.86
# phi = 0.98
# sigma_eta = 0.15 -> sigma2_eta = 0.225

# the joint likelihood funtion. x is the vector of parameters.

# a 3 dimensional vector x to start the chain:

# xstart - to transform mu -> mu, phi -> log[(1+phi)/(1-phi)], sigma2_eta -> log(sigma2_eta) and store in x:
def xstart(mu = -0.86, phi = 0.98, sigma2_eta = 0.0225):
    xstart = np.full(3,np.nan)
    xstart[0] = mu 
    xstart[1] = np.log(sigma2_eta) 
    xstart[2] = np.log((1+phi)/(1-phi)) 
    return xstart

# firstlatent - to draw the first latent variables matrix (each col is a sep. draw) 
# T - number of y observations. m - number of latent variables
def firstlatent(x, T, m):
    # x is 3 dimensional: mu, log[(1+phi)/(1-phi)] and log(sigma2_eta)
    mu = x[0]
    sigma2_eta = np.exp(x[1])
    phi = (np.exp(x[2])-1)/(1+np.exp(x[2]))
    # matrix H - Txm, each row i has m draws of latent variable h_i
    # initiazize matrix H - T by m - each clumn is a new draw vector h_1..T 
    H = np.full((T,m), np.nan)
    H[0] = st.norm.rvs(mu, np.sqrt(sigma2_eta/(1-phi**2)), m)
    for i in range(T-1):
        H[i+1] = st.norm.rvs(mu + phi*(H[i] - mu), np.sqrt(sigma2_eta), m)
    return H

# to draw the latent matrices after first draw. rho is correlation coefficient
def newlatent(H, rho): 
    x = st.multivariate_normal.rvs(size = H.size)
    x = x.reshape(H.shape)
    Hnew = rho*H + np.sqrt(1 - rho**2)*x
    return Hnew

# given matrix H containing diff draws of latent variables as columns, calculate the loglik
def log_lik(H, y): 
    m = H.shape[1]
    Y = np.tile(y[:, None], (1, m)) # matrix Y, each column j is the same vector of obs. y1..T

    Y_pdf = st.norm.pdf(Y, 0, np.exp(H/2)) # calculate the individual pdfs of y_ij conditional on h_ij
    pdfs = np.mean(Y_pdf, axis = 1) # vector pdfs, each element is the likelihood estimate for y_i: (1/m)*sumj{p(y_i|h_j)}
    loglik = np.sum(np.log(pdfs)) # full log likelihood estimator

    return loglik


# defining the prior pdfs to return log_prior
def log_prior(x, mu_mean, mu_var, sigma2_mean, \
               sigma2_var, phi_mean, phi_var):
    logpdf_mu = st.norm.logpdf(x[0],mu_mean,np.sqrt(mu_var))
    # for sigma_eta and phi, we work with the unconstrained parameters x[1] = log(sigma2_eta), x[2] = log[(1+phi)/(1-phi)]
    # we form the posterior with these variables. After MCMC chain is ready, we will transform back
    logpdf_sigma = st.norm.logpdf(x[1],np.log(sigma2_mean),np.sqrt(sigma2_var))
    logpdf_phi = st.norm.logpdf(x[2],np.log((1+phi_mean)/(1-phi_mean)),np.sqrt(phi_var))
    log_prior = logpdf_mu + logpdf_sigma + logpdf_phi
    return log_prior

#log_prior + log_lik = log_post
def log_post(x, latent, y, prior_pars):
    return log_prior(x, mu_mean = prior_pars["mu_mean"], mu_var = prior_pars["mu_var"], \
                     sigma2_mean = prior_pars["sigma2_mean"], sigma2_var = prior_pars["sigma2_var"],\
                     phi_mean = prior_pars["phi_mean"], phi_var = prior_pars["phi_var"]) + log_lik(H = latent, y =y)



# initialize a matrix to store the MCMC draws. Store each new vector as a row of a matrix.

def correl_pseudo_MCMC(prior_pars, ys, correl = 0.8, N_mcmc = 20000, x_first = xstart(), x_std = 0.01, m_latent = 50):
    TT = len(ys)
    cov = x_std**2*np.eye(3)
    draws = np.full((N_mcmc,3), np.nan) 
    draws[0] = x_first
    xold = x_first
    L = firstlatent(xold, T = TT, m = m_latent)
    oldlik = log_post(xold, latent = L, y=ys, prior_pars = prior_pars)
    acc_all = 0 # to count the number of acceptances

    for i in range(N_mcmc-1):
        xnew = xold + x_std*st.norm.rvs(size = 3)
        Lnew = newlatent(L, rho = correl)
        newlik = log_post(xnew, latent = Lnew, y=ys, prior_pars = prior_pars)
        # let's define the acceptance ratio
        acc = newlik + st.multivariate_normal.logpdf(xold, xnew, cov) - oldlik - st.multivariate_normal.logpdf(xnew, xold, cov)
        
        #uniform draw to approve acceptance
        u = st.uniform.rvs()
        if np.log(u) < acc:
            xold = xnew
            oldlik = newlik
            L = Lnew
            acc_all = acc_all + 1
        draws[i+1] = xold
    mu_draws = draws[:,0]
    sigma2_draws = np.exp(draws[:,1])
    phi_draws = (np.exp(draws[:,2])-1)/(1+np.exp(draws[:,2]))
    acc_ratio = acc_all/N_mcmc
    return {"mu_draws": mu_draws,
            "sigma2_draws": sigma2_draws,
            "phi_draws":phi_draws,
            "acc_ratio": acc_ratio}




    # load the generated returns and log volatility
if __name__ == "__main__":
    real_pars = {"mu": -0.86, "sigma2_eta": 0.025, "phi": 0.98}
    prior_pars = {"mu_mean": -0.86, "mu_var" : 0.01,"sigma2_mean" : 0.025, "sigma2_var" : 0.01, "phi_mean" : 0.98, "phi_var" : 0.01}

    stochvol.generate(mu = real_pars["mu"], phi = real_pars["sigma2_eta"], sigma2_eta = real_pars["phi"], T = 100)
    y_gen = stochvol.ys
    h_gen = stochvol.hs
    values = correl_pseudo_MCMC(prior_pars = prior_pars)
    mu_mean = np.mean(values["mu_draws"][1000:])
    sigma2_mean = np.mean(values["sigma2_draws"][1000:])
    phi_mean = np.mean(values["phi_draws"][1000:])
    print(f"mu_mean: {mu_mean} \nsigma2_mean: {sigma2_mean} \nphi_mean: {phi_mean}")
    plt.subplot(2,2,1) # plotting mu
    plt.plot(values["mu_draws"])

    plt.subplot(2,2,2) # plotting 
    plt.plot(values["sigma2_draws"])

    plt.subplot(2,2,3) # plotting phi
    plt.plot(values["phi_draws"])
    plt.tight_layout()
    plt.show()

# set up the directory


