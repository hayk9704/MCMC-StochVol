import numpy as np
from scipy.stats import norm, beta, gamma, halfnorm
# the commonly used prior distributions:
# m ~ N(0, 10), (phi + 1)/2 ~ Beta(20, 1/5) and sigma2 Gam(1/2, 1/2).

# turns the transformed vector x back into theta
def x_to_theta(x):
    mu = x[0]
    sigma2_eta = np.exp(x[1])
    phi = (np.exp(x[2])-1)/(1+np.exp(x[2]))
    theta = {"mu": mu, "sigma2_eta": sigma2_eta, "phi": phi}
    return theta

# transforms original parameters into 
# mu -> mu, phi -> log[(1+phi)/(1-phi)], sigma2_eta -> log(sigma2_eta) and store in x:
def theta_to_x(mu = -0.86, phi = 0.98, sigma2_eta = 0.0225):
    x = np.full(3,np.nan)
    x[0] = mu 
    x[1] = np.log(sigma2_eta) 
    x[2] = np.log((1+phi)/(1-phi)) 
    return x   


def log_prior(x):
    theta = x_to_theta(x)
    mu = theta["mu"]
    sigma2 = theta["sigma2_eta"]
    phi = theta["phi"]
    """
    Compute the log prior for the parameters:
      mu       ~ N(0, 10^2)
      (phi+1)/2~ Beta(20, 1.5)
      sigma2   ~ Gamma(shape=1/2, rate=1/2)
    """
    # Log-density for mu
    lp_mu = norm.logpdf(mu, loc=0, scale=10)
    
    # Check phi bounds
    if phi < -1 or phi > 1:
        return -np.inf
    
    # Transformation for phi
    phi_star = (phi + 1) / 2
    # Beta log-density plus Jacobian term np.log(phi_star) +np.log(1 - phi_star)
    lp_phi = beta.logpdf(phi_star, a=20, b=1.5) + np.log(phi_star) +np.log(1 - phi_star)
    
    # Check sigma2 positivity
    if sigma2 <= 0:
        return -np.inf
    
    # Gamma prior on sigma2: shape=0.5, rate=0.5 => scale = 1/rate = 2
    # log(sigma2) - the jacobian term
    lp_sigma2 = gamma.logpdf(sigma2, a=0.5, scale=2) + np.log(sigma2)
    """
    a half normal prior on sigma2 instead - scale = 0.2 means sigma is centered around 0.15
    
    lp_sigma2 = halfnorm.logpdf(np.sqrt(sigma2), loc = 0, scale = 0.2) - np.log(2) - 0.5*np.log(sigma2)
    """ 
    return lp_mu + lp_phi + lp_sigma2

class stochvol:
    @classmethod
    def generate(cls, mu = -0.86, phi = 0.98, sigma2_eta = 0.025, T = 100, seed = None):
        rng = np.random.default_rng(seed)
        cls.mu = mu
        cls.sigma2_eta = sigma2_eta
        cls.phi = phi
        cls.T = T
        sd_h0 = np.sqrt(cls.sigma2_eta/(1-cls.phi**2))
        h = np.zeros(cls.T)
        h[0] = rng.normal(cls.mu, sd_h0)
        for i in range(T-1):
            h[i+1] = rng.normal(cls.mu + cls.phi*(h[i] - cls.mu), np.sqrt(cls.sigma2_eta))
        y = rng.normal(0, np.exp(h/2))
        cls.ys = y
        cls.hs = h
        cls.pars = np.array([cls.mu, cls.sigma2_eta, cls.phi])
        cls.theta = np.concatenate((cls.pars, cls.hs))

def log_uninf_prior(x):
    theta = x_to_theta(x)
    mu = theta["mu"]
    sigma2 = theta["sigma2_eta"]
    phi = theta["phi"]
    """
    Compute the log prior for the parameters:
      mu       ~ N(0, 10^2)
      (phi+1)/2~ Beta(20, 1.5)
      sigma2   ~ Gamma(shape=1/2, rate=1/2)
    """
    # Log-density for mu
    lp_mu = norm.logpdf(mu, loc=0, scale=10)
    
    # Check phi bounds
    if phi < -1 or phi > 1:
        return -np.inf
    
    # Transformation for phi
    phi_star = (phi + 1) / 2
    # Beta log-density plus Jacobian term np.log(phi_star) +np.log(1 - phi_star)
    lp_phi = beta.logpdf(phi_star, a=1, b=1) + np.log(phi_star) +np.log(1 - phi_star)
    
    # Check sigma2 positivity
    if sigma2 <= 0:
        return -np.inf
    
    # Gamma prior on sigma2: shape=0.5, rate=0.5 => scale = 1/rate = 2
    # log(sigma2) - the jacobian term
    lp_sigma2 = gamma.logpdf(sigma2, a=0.001, scale=1000) + np.log(sigma2)
    
    return lp_mu + lp_phi + lp_sigma2