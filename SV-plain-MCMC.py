import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt

# set up the directory
direct = r"C:\Users\haykg\Documents\university files\Dissertation\MCMC-R-codes\MCMC-python"

# load the generated returns and log volatility
h_gen = np.loadtxt(f"{direct}\\h100.txt")
y_gen = np.loadtxt(f"{direct}\\y100.txt")

# total number of variables (h1..100, mu, sigma_eta, phi)
n = 103

# The number of MCMC iterations:
N = 10000
s = 0.01 # cov of MVN s^2*I

# the actual parameters (to be changed)
# mu = -0.86
# phi = 0.98
# sigma_eta = 0.15 -> sigma2_eta = 0.225

# the joint likelihood funtion. x is the vector of parameters.

def log_lik(x): 
    # x is 103 dimensional. 0:3 are mu, phi and sigma-eta.
    mu = x[0]
    sigma2_eta = np.exp(x[1])
    phi = (np.exp(x[2])-1)/(1+np.exp(x[2]))

    # 3:102 are h variables
    h = x[3:]
    
    ### p(y_t|h_t)
    vec_y = np.zeros(n-3)
    vec_y = st.norm.pdf(y_gen, 0, np.exp(h/2)) 

    ### p(h_1|mu,phi,sigma_eta) and p(h_t|h_t-1)
    vec_h = np.zeros(len(h))
    vec_h[0] = st.norm.pdf(h[0], mu, np.sqrt(sigma2_eta/(1-phi**2)))
    vec_h[1:] = st.norm.pdf(h[1:],  mu + phi*(h[0:99] - mu), np.sqrt(sigma2_eta))
    
    y_loglik = np.sum(np.log(vec_y))
    h_loglik = np.sum(np.log(vec_h))
    full_loglik = y_loglik + h_loglik
    return full_loglik


# defining the prior pdfs
def log_prior(x):
    mu = x[0] # the 4 lines are unnecessary. Remove them later.
    sigma2_eta = np.exp(x[1])
    phi = (np.exp(x[2])-1)/(1+np.exp(x[2]))
    logpdf_mu = st.norm.logpdf(x[0],-0.86,np.sqrt(0.01))
    # for sigma_eta and phi, we work with the unconstrained parameters x[1] = log(sigma2_eta), x[2] = log[(1+phi)/(1-phi)]
    # we form the posterior with these variables. After MCMC chain is ready, we will transform back
    logpdf_sigma = st.norm.logpdf(x[1],-3.67848,np.sqrt(0.01))
    logpdf_phi = st.norm.logpdf(x[2],np.log((1+0.98)/(1-0.98)),np.sqrt(0.01))
    log_prior = logpdf_mu + logpdf_sigma + logpdf_phi
    return log_prior

def log_post(x):
    return log_prior(x) + log_lik(x)

xstart = np.zeros(103)
xstart[0] = -0.2 # mu
xstart[1] = np.log(0.1**2) # log(sigma2_eta)
xstart[2] = np.log((1+0.9)/(1-0.9)) # log[(1+phi)/(1-phi)]
xstart[3:] = h_gen 
post = log_post(xstart)

print(post)

# initialize a matrix to store the MCMC draws. Store each new vector as a row of a matrix.
draws = np.full((N,n), np.nan) 
draws[0] = xstart
# the proposal is multivariate normal with N(x_t-1,s^2*I)
cov = s**2*np.eye(n) # s^2*I
oldlik = log_post(xstart)
xold = draws[0]
for i in range(N-1):
    xnew = st.multivariate_normal.rvs(xold, cov, 1)
    newlik = log_post(xnew)
    # let's define the acceptance ratio
    acc = newlik + st.multivariate_normal.logpdf(xold, xnew, cov) - oldlik - st.multivariate_normal.logpdf(xnew, xold, cov)
    
    #uniform draw to approve acceptance
    u = st.uniform.rvs()
    if np.log(u) < acc:
        xold = xnew
        oldlik = newlik
    draws[i+1] = xold
print(np.mean(draws, axis = 0))

mu_draws = draws[:,0]
sigma2_draws = np.exp(draws[:,1])
phi_draws = (np.exp(draws[:,2])-1)/(1+np.exp(draws[:,2]))

plt.subplot(2,2,1) # plotting mu
plt.plot(mu_draws)

plt.subplot(2,2,2) # plotting 
plt.plot(sigma2_draws)

plt.subplot(2,2,3) # plotting phi
plt.plot(phi_draws)
plt.tight_layout
plt.show()