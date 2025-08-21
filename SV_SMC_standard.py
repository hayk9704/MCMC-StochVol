import numpy as np
from MCMC_functions import stochvol, x_to_theta, theta_to_x
from scipy.special import logsumexp

"""
x is an np.array with those parameters
    x[0] = mu 
    x[1] = np.log(sigma2_eta) 
    x[2] = np.log((1+phi)/(1-phi)) 
"""

def stable_log_pdf(y, h):
    """
    Numerically stable calculation of the log PDF for y ~ N(0, exp(h)).
    
    This avoids calculating np.exp(h/2) directly, which can underflow to 0
    and cause NaN values in standard library logpdf functions.
    
    The log PDF of y ~ N(mu, sigma^2) is:
    -0.5 * log(2*pi*sigma^2) - (y-mu)^2 / (2*sigma^2)
    
    Here, mu=0 and sigma^2 = exp(h), so the formula becomes:
    -0.5 * (log(2*pi) + h) - y^2 / (2*exp(h))
    """
    return -0.5 * (np.log(2 * np.pi) + h) - 0.5 * (y**2) * np.exp(-h)


def SMC(y, x, m_latent = 50):
    theta = x_to_theta(x)
    T = len(y)
    h, loglik = particles0(y_t = y[0], theta = theta, m_latent = m_latent)
    for i in range(T-1):
        h, newloglik  = particles(y_t = y[i+1], h_anc = h, theta = theta, m_latent = m_latent)        
        loglik = loglik + newloglik
    return loglik

# function te return ancestor h and loglik for obs. y_0
def particles0(y_t, theta, m_latent):
    var = theta["sigma2_eta"]/(1 - theta["phi"]**2)
    h = np.random.normal(loc = theta["mu"], scale = np.sqrt(var), size = m_latent)
    
    logweights = stable_log_pdf(y_t, h)                                               # weights    
    log_total_weight = logsumexp(logweights)
    norm_weights = np.exp(logweights - log_total_weight)  # logsumexp for w

    loglik = log_total_weight - np.log(m_latent)                    # logsumexp for loglik
    ancestor_indices = np.random.choice(m_latent, size=m_latent, replace=True, p=norm_weights)

    ancestor_h = h[ancestor_indices]
    return ancestor_h, loglik

# function te return ancestor h and loglik for obs. y_1...T
def particles(y_t, h_anc, theta, m_latent):
    mu = theta["mu"] + theta["phi"]*(h_anc - theta["mu"])
    var = theta["sigma2_eta"]
    h = np.random.normal(loc = mu, scale = np.sqrt(var), size = m_latent)

    logweights = stable_log_pdf(y_t, h)                                               # weights    
    log_total_weight = logsumexp(logweights)
    norm_weights = np.exp(logweights - log_total_weight)  # logsumexp for w
    
    loglik = log_total_weight - np.log(m_latent)                    # logsumexp for loglik
    ancestor_indices = np.random.choice(m_latent, size=m_latent, replace=True, p=norm_weights)

    ancestor_h = h[ancestor_indices]
    return ancestor_h, loglik



if __name__ == "__main__":
    SMC_rep = 500                       # number of SMC repetitions
    x = theta_to_x(mu = -0.86, phi = 0.98, sigma2_eta = 0.0225)
    stochvol.generate(mu = -0.86, phi = 0.98, sigma2_eta = 0.0225, T = 200, seed = 101)
    y_obs = stochvol.ys
    loglik_chain = np.full(SMC_rep, np.nan)
    for i in range(SMC_rep):
        loglik = SMC(y_obs, x = x, m_latent = 100)
        loglik_chain[i] = loglik
    print(f"standard deviation: {np.std(loglik_chain)}")