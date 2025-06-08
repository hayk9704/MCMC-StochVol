import numpy as np
from scipy import stats as st

# set up the directory
direct = r"C:\Users\haykg\Documents\university files\Dissertation\MCMC-R-codes\MCMC-python"

# load the generated returns and log volatility
h_gen = np.loadtxt(f"{direct}\\h100.txt")
y_gen = np.loadtxt(f"{direct}\\y100.txt")

# total number of variables (h1..100, mu, sigma_eta, phi)
n = 103

# the parameters (to be changed)
mu = -0.86
phi = 0.98
sigma_eta = 0.15

# the joint likelihood funtion. x is the vector of parameters.
def log_lik(x): 
    # x is 103 dimensional. 0:3 are mu, phi and sigma-eta.
    mu = x[0]
    sigma_eta = np.exp(x[1])
    phi = (np.exp(x[2])-1)/(1+np.exp(x[2]))

    # 3:102 are h variables
    h = x[3:]
    
    ### p(y_t|h_t)
    vec_y = np.zeros(n-3)
    vec_y = st.norm.pdf(y_gen, 0, np.exp(h/2)) 

    ### p(h_1|mu,phi,sigma_eta) and p(h_t|h_t-1)
    vec_h = np.zeros(len(h))
    vec_h[0] = st.norm.pdf(h[0], mu, sigma_eta/np.sqrt(1-phi**2))
    vec_h[1:] = st.norm.pdf(h[1:],  mu + phi*(h[0:99] - mu), np.sqrt(sigma_eta))

    y_loglik = np.sum(np.log(vec_y))
    h_loglik = np.sum(np.log(vec_h))
    full_loglik = y_loglik + h_loglik
    return full_loglik

xstart = np.zeros(103)
xstart[0] = -0.86
xstart[1] = np.log(0.15)
xstart[2] = np.log((1+0.98)/(1-0.98))
xstart[3:] = h_gen
thelik = log_lik(xstart)

