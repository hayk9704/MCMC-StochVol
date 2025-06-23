import numpy as np
from scipy import stats as st

class stochvol:
    @classmethod
    def generate(cls, mu = -0.86, phi = 0.98, sigma2_eta = 0.025, T = 100):
        cls.mu = mu
        cls.sigma2_eta = sigma2_eta
        cls.phi = phi
        cls.T = T
        sd_h0 = np.sqrt(cls.sigma2_eta/(1-cls.phi**2))
        h = np.zeros(cls.T)
        h[0] = st.norm.rvs(cls.mu, sd_h0)
        for i in range(T-1):
            h[i+1] = st.norm.rvs(cls.mu + phi*(h[i] - mu), np.sqrt(cls.sigma2_eta), 1)
        y = st.norm.rvs(0, np.exp(h/2))
        cls.ys = y
        cls.hs = h
        cls.pars = np.array([cls.mu, cls.sigma2_eta, cls.phi])
        cls.theta = np.concatenate((cls.pars, cls.hs))
