import numpy as np
from scipy import stats as st



def xstart(mu = -0.86, phi = 0.98, sigma2_eta = 0.0225):
    xstart = np.full(3,np.nan)
    xstart[0] = mu 
    xstart[1] = np.log(sigma2_eta) 
    xstart[2] = np.log((1+phi)/(1-phi)) 
    return xstart

def log_lik(x, y): 
    # x is 3 + len(h) dimensional. 0:3 are mu, phi and sigma-eta.
    mu = x[0]
    sigma2_eta = np.exp(x[1])
    phi = (np.exp(x[2])-1)/(1+np.exp(x[2]))

    # 3: are h variables
    h = x[3:]
    
    ### p(y_t|h_t)
    vec_y = st.norm.logpdf(y, 0, np.exp(h/2)) 

    ### p(h_1|mu,phi,sigma_eta) and p(h_t|h_t-1)
    vec_h = np.zeros(len(h))
    vec_h[0] = st.norm.logpdf(h[0], mu, np.sqrt(sigma2_eta/(1-phi**2)))
    vec_h[1:] = st.norm.logpdf(h[1:],  mu + phi*(h[0:-1] - mu), np.sqrt(sigma2_eta))
    
    y_loglik = np.sum(vec_y)
    h_loglik = np.sum(vec_h)
    full_loglik = y_loglik + h_loglik
    return full_loglik


# defining the prior pdfs
def log_prior(x, mu_mean, mu_var, sigma2_mean, \
               sigma2_var, phi_mean, phi_var):
    logpdf_mu = st.norm.logpdf(x[0],mu_mean,np.sqrt(mu_var))
    # for sigma_eta and phi, we work with the unconstrained parameters x[1] = log(sigma2_eta), x[2] = log[(1+phi)/(1-phi)]
    # we form the posterior with these variables. After MCMC chain is ready, we will transform back
    logpdf_sigma = st.norm.logpdf(x[1],np.log(sigma2_mean),np.sqrt(sigma2_var))
    logpdf_phi = st.norm.logpdf(x[2],np.log((1+phi_mean)/(1-phi_mean)),np.sqrt(phi_var))
    log_prior = logpdf_mu + logpdf_sigma + logpdf_phi
    return log_prior

def log_post(x, y, prior_pars):
    return log_prior(x, mu_mean = prior_pars["mu_mean"], mu_var = prior_pars["mu_var"], \
                     sigma2_mean = prior_pars["sigma2_mean"], sigma2_var = prior_pars["sigma2_var"],\
                     phi_mean = prior_pars["phi_mean"], phi_var = prior_pars["phi_var"]) + log_lik(x, y = y)

def plain_MCMC(prior_pars, ys, hs, N_mcmc = 20000, x_first = xstart(),  x_std = 0.01): 
    # the proposal is multivariate normal with N_mcmc(x_t-1,s^2*I)
    x0 = np.concatenate((x_first, hs)) # (mu -> x[0], sigma_eta -> x[2], phi -> x[3], h1..100)
    n = len(x0)

    # initialize a matrix to store the MCMC draws. Store each new vector as a row of a matrix.
    draws = np.full((N_mcmc,n), np.nan) 
    draws[0] = x0
    
    xold = x0
    oldlik = log_post(x0, y = ys, prior_pars = prior_pars)
    acc_all = 0 #to count the number of acceptances


    for i in range(N_mcmc-1):
        xnew = xold + x_std*st.norm.rvs(size = n)
        newlik = log_post(xnew, y = ys, prior_pars = prior_pars)
        # let's define the acceptance ratio
        # below i don't include + st.multivariate_normal.logpdf(xold, xnew, cov) and - st.multivariate_normal.logpdf(xnew, xold, cov) as they're symmetric
        acc = newlik - oldlik 
        
        #uniform draw to approve acceptance
        u = st.uniform.rvs()
        if np.log(u) < acc:
            xold = xnew
            oldlik = newlik
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

