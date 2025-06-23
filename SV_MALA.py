import numpy as np
from scipy import stats as st
from scipy.optimize import minimize

# the actual parameters (to be changed)
# mu = -0.86
# phi = 0.98
# sigma_eta = 0.15 -> sigma2_eta = 0.225

# the joint likelihood funtion. x is the vector of parameters.

def xstart(mu = -0.86, sigma2_eta = 0.0225, phi = 0.98, ):
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
    assert len(y) == len(h)
    
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

# negative log post to run minimization algorithm:

def bfgs_cov(y, prior_pars, real_pars, h, maxiter = 500, disp = True):

    # 1) build x0 = [μ, log σ²_η, logit φ, h₁,…,h_T]
    x0 = np.concatenate([
        xstart(real_pars["mu"],real_pars["sigma2_eta"], real_pars["phi"], ),
        h
    ])

    # 2) objective for minimize: neg_log_post(x, y)
    def neg_log_post(x):
        return -log_post(x, y, prior_pars)   # uses your previously‐defined log_post
    
    # 3) call BFGS
    res = minimize(
        fun    = neg_log_post,
        x0     = x0,
        method = 'BFGS',
        options = {
          'maxiter': maxiter,
          'disp'   : disp
        }
    )

    # 4) extract proposal covariance (already –[H⁻¹])
    S    = res.hess_inv
    
    return S


# the function to output the gradient vector. # x is 3 + len(h) dimensional. 0:3 are mu, phi and sigma-eta.
# y is the vector of obs.
def gradient(x, y, prior_pars):
    assert len(y) == len(x[3:])

    mu_mean = prior_pars["mu_mean"]
    mu_var = prior_pars["mu_var"]
    sigma2_mean = prior_pars["sigma2_mean"]
    sigma2_var = prior_pars["sigma2_var"]
    phi_mean = prior_pars["phi_mean"]
    phi_var = prior_pars["phi_var"]

    mu = x[0]
    sigma2_eta = np.exp(x[1])
    phi = (np.exp(x[2])-1)/(1+np.exp(x[2]))
    # vector of der p(yt|ht) w.r.t. ht
    deriv_hy = -1/2 + ((y**2)/2)*np.exp(-x[3:])
    # derivative of the sum logp(h_1)+logp(h_2 ∣h_1​) w.r.t. h_1
    deriv_h1 = -(1-phi**2)*(x[3] - mu)*(1/sigma2_eta) + phi*(x[4] - mu*(1-phi) - phi*x[3])*(1/sigma2_eta)
    # Derivaive of log p(ht|ht-1) w.r.t. middle entries up to p(hT-1)
    deriv_ht = phi*(x[5:] - mu*(1-phi) - phi*x[4:-1])*(1/sigma2_eta) - (x[4:-1] - mu*(1-phi) - phi*x[3:-2])*(1/sigma2_eta)
    # Derivative of log p(hT|hT-1) w.r.t. hT - last entry
    deriv_hT = -(x[-1] - mu*(1-phi) - phi*x[-2])*(1/sigma2_eta)
    deriv_h = deriv_hy + np.concatenate((np.array([deriv_h1]), deriv_ht, np.array([deriv_hT])))

    # derivative of the log posterior w.r.t. mu
    dmu = -(x[0]- mu_mean)/mu_var + (1-phi**2)*(x[3]-x[0])*(1/sigma2_eta)
    dmu = dmu +(1-phi)*np.sum(x[4:]-mu*(1-phi)-phi*x[3:-1])*(1/sigma2_eta)

    # derivative of the log post. w.r.t. s2eta
    ds2eta = -(x[1]-np.log(sigma2_mean))/sigma2_var -0.5*len(x[3:]) + 0.5*(1-phi**2)*((x[3]-x[0])**2)*(1/sigma2_eta)
    ds2eta =ds2eta + 0.5*np.sum( (x[4:]-mu*(1-phi)-phi*x[3:-1])**2   )*(1/sigma2_eta)

    # Derivative of the likelihood w.r.t. phi
    dphi = -(x[2]-np.log((1+phi_mean)/(1-phi_mean)))/phi_var +0.5 -1/(1+np.exp(-x[2])) - 2*np.exp(2*x[2])*((x[3]-x[0])**2)/(sigma2_eta*(1+np.exp(x[2]))**3)
    dphi = dphi + 2*np.exp(x[2])*np.sum(  (x[4:]-mu*(1-phi)-phi*x[3:-1])*(x[3:-1]-mu))/(sigma2_eta*(1+np.exp(x[2]))**2)

    grad = np.concatenate((np.array([dmu]), np.array([ds2eta]), np.array([dphi]), deriv_h))
    return grad
 


def MALA_MCMC(prior_pars, real_pars, hs, ys, N_mcmc = 3000, x_first = xstart(), MALA_step = 0.8): 
    # the proposal is multivariate normal with N_mcmc(x_t-1,s^2*I)
    x0 = np.concatenate((x_first, hs)) # (mu -> x[0], sigma_eta -> x[2], phi -> x[3], h1..100)
    n = len(x0)
    S_prop = bfgs_cov(y = ys, real_pars = real_pars, prior_pars = prior_pars, h = hs) # the covar approximation
    eps2 = MALA_step**2
    

    # initialize a matrix to store the MCMC draws. Store each new vector as a row of a matrix.
    draws = np.full((N_mcmc,n), np.nan) 
    draws[0] = x0
    
    xold = x0
    oldlik = log_post(x0, y = ys, prior_pars = prior_pars)
    acc_all = 0 #to count the number of acceptances
    grad_old = gradient(xold, ys, prior_pars)


    for i in range(N_mcmc-1):
        xnew = xold + 0.5*eps2*S_prop@grad_old+ st.multivariate_normal.rvs(np.zeros(n),eps2*S_prop)
        newlik = log_post(xnew, y = ys, prior_pars = prior_pars)
        grad_new = gradient(xnew, ys, prior_pars)
        # let's define the acceptance ratio
        # below i don't include + st.multivariate_normal.logpdf(xold, xnew, cov) and - st.multivariate_normal.logpdf(xnew, xold, cov) as they're symmetric
        acc = newlik + st.multivariate_normal.logpdf(xold, xnew + (eps2/2)*S_prop@grad_new, eps2*S_prop) \
            - oldlik - st.multivariate_normal.logpdf(xnew, xold + (eps2/2)*S_prop@grad_old, eps2*S_prop)
        
        #uniform draw to approve acceptance
        u = st.uniform.rvs()
        if np.log(u) < acc:
            xold = xnew
            oldlik = newlik
            grad_old = grad_new
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

