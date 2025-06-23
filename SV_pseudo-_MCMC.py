import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt

# set up the directory
direct = r"C:\Users\haykg\Documents\university files\Dissertation\MCMC-R-codes\MCMC-python"

# load the generated returns and log volatility
y_gen = np.loadtxt(f"{direct}\\y100.txt")


n = 103 # total number of variables (h1..100, mu, sigma_eta, phi)

T = n - 3 # number of observations y_t. also the same as number of h_t

m = 50 # number of latent draws

N = 10000 # The number of MCMC iterations:
s = 0.01 # cov of MVN s^2*I

# the actual parameters (to be changed)
# mu = -0.86
# phi = 0.98
# sigma_eta = 0.15 -> sigma2_eta = 0.225

# the joint likelihood funtion. x is the vector of parameters.

def log_lik(x): 
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


    Y = np.tile(y_gen[:, None], (1, m)) # matrix Y, each column j is the same vector of obs. y1..T

    Y_pdf = st.norm.pdf(Y, 0, np.exp(H/2)) # calculate the individual pdfs of y_ij conditional on h_ij
    pdfs = np.sum(Y_pdf, axis = 1)/N # vector pdfs, each element is the likelihood estimate for y_i: 1/N*sumj{p(y_i|h_j)}
    loglik = np.sum(np.log(pdfs)) # full log likelihood estimator

    return loglik


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

xstart = np.full(3,np.nan)
xstart[0] = -0.2 # mu
xstart[1] = np.log(0.1**2) # log(sigma2_eta)
xstart[2] = np.log((1+0.9)/(1-0.9)) # log[(1+phi)/(1-phi)]
post = log_post(xstart)

print(post)

# initialize a matrix to store the MCMC draws. Store each new vector as a row of a matrix.
draws = np.full((N,3), np.nan) 
draws[0] = xstart
# the proposal is multivariate normal with N(x_t-1,s^2*I)
cov = s**2*np.eye(3) # s^2*I
oldlik = log_post(xstart)
xold = draws[0]
acc = 0
for i in range(N-1):
    xnew = st.multivariate_normal.rvs(xold, cov)
    newlik = log_post(xnew)
    # let's define the acceptance ratio
    acc = newlik + st.multivariate_normal.logpdf(xold, xnew, cov) - oldlik - st.multivariate_normal.logpdf(xnew, xold, cov)
    
    #uniform draw to approve acceptance
    u = st.uniform.rvs()
    if np.log(u) < acc:
        xold = xnew
        oldlik = newlik
        acc = acc + 1
    draws[i+1] = xold


mu_draws = draws[:,0]
sigma2_draws = np.exp(draws[:,1])
phi_draws = (np.exp(draws[:,2])-1)/(1+np.exp(draws[:,2]))
mu_mean = np.mean(mu_draws[1000:])
sigma2_mean = np.mean(sigma2_draws[1000:])
phi_mean = np.mean(phi_draws[1000:])
print(f"mu_mean: {mu_mean} \nsigma2_mean: {sigma2_mean} \nphi_mean: {phi_mean}")
plt.subplot(2,2,1) # plotting mu
plt.plot(mu_draws)

plt.subplot(2,2,2) # plotting 
plt.plot(sigma2_draws)

plt.subplot(2,2,3) # plotting phi
plt.plot(phi_draws)
plt.tight_layout
plt.show()