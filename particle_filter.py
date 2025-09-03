import numpy as np
from MCMC_functions import stochvol, x_to_theta, theta_to_x
from scipy.special import logsumexp
from scipy.stats import norm

"""
x is an np.array with those parameters
    x[0] = mu 
    x[1] = np.log(sigma2_eta) 
    x[2] = np.log((1+phi)/(1-phi)) 
"""

"""
Note for U:
u = U[0] are all randomness for T = 0
u[0] is the uniform, u[1:] are the normals
"""


# for below u_unif is the uniform RV
def systematic_resample(weights, u_unif):
    """
    Performs systematic resampling.
    """
    N = len(weights)
    positions = (np.arange(N) + u_unif) / N
    
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    indices = np.zeros(N, 'i')
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices

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


def SMC(y, x, U, m_latent = 50):                            # U is the full matrix with randomness
    theta = x_to_theta(x)
    h, loglik = particles0(y_t = y[0], theta = theta, m_latent = m_latent, randomness = U[0])
    for i in range(len(y)-1):
        h, newloglik  = particles(y_t = y[i+1], h_anc = h, theta = theta, m_latent = m_latent, randomness = U[i+1])        
        loglik = loglik + newloglik
    return loglik

def particles0(y_t, theta, m_latent, randomness):
    unif = norm.cdf(randomness[0])                                  #transforming standard normal to uniform

    var = theta["sigma2_eta"]/(1 - theta["phi"]**2)
    h =  theta ["mu"] + np.sqrt(var)*randomness[1:]
    
    logweights = stable_log_pdf(y_t, h)                             # weights    
    log_total_weight = logsumexp(logweights)
    norm_weights = np.exp(logweights - log_total_weight)            # logsumexp for w

    loglik = log_total_weight - np.log(m_latent)                    # logsumexp for loglik
    ancestor_indices = systematic_resample(weights = norm_weights,u_unif =  unif)

    ancestor_h = h[ancestor_indices]
    return ancestor_h, loglik

def particles(y_t, h_anc, theta, m_latent, randomness):
    unif = norm.cdf(randomness[0])                                  #transforming standard normal to uniform
    mu = theta["mu"] + theta["phi"]*(h_anc - theta["mu"])
    var = theta["sigma2_eta"]
    h = mu + np.sqrt(var)*randomness[1:]

    logweights = stable_log_pdf(y_t, h)                                               # weights    
    log_total_weight = logsumexp(logweights)
    norm_weights = np.exp(logweights - log_total_weight)            # logsumexp for w
    
    loglik = log_total_weight - np.log(m_latent)                    # logsumexp for loglik
    ancestor_indices = systematic_resample(weights = norm_weights,u_unif = unif)

    ancestor_h = h[ancestor_indices]
    return ancestor_h, loglik