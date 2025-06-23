import pandas as pd
import numpy as np
from scipy import stats as st

output_path = r"C:\Users\haykg\Documents\university files\Dissertation\MCMC-R-codes\MCMC-python"

T = 100
mu = -0.86
phi = 0.98
sigma_eta = 0.15
sd_h0 = sigma_eta/np.sqrt(1-phi**2)

y = np.zeros(T)
h = np.zeros(T)
print(y)

h[0] = st.norm.rvs(mu, sd_h0)

for i in range(T-1):
    h[i+1] = st.norm.rvs(mu + phi*(h[i] - mu), sigma_eta, 1)

y = st.norm.rvs(0, np.exp(h/2))

np.savetxt(f"{output_path}\\h100.txt", h)
np.savetxt(f"{output_path}\\y100.txt", y)