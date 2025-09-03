import os
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"


import time
import numpy as np
import pandas as pd
import arviz as az 
from scipy import stats as st
import matplotlib.pyplot as plt
from MCMC_functions import stochvol, theta_to_x as xstart
from PM_IS_adaptive import PM_IS_adaptive
from PM_IS import PM_IS
"""
all MCMC functions return dicts:
            {"mu_draws": mu_draws,
            "sigma2_draws": sigma2_draws,
            "phi_draws":phi_draws,
            "acc_ratio": acc_ratio,
            "mu_burnin_draws": mu_burnin_draws,
            "sigma2_burnin_draws": sigma2_burnin_draws,
            "phi_burnin_draws": phi_burnin_draws}

I will adjust the x_std values for all chains so that the acc ratio stays at:
plain MCMC - 23%
pseudo-marginal MCMC - 7%
correlated-pseudo-marginal MCMC - 23%
"""

run_seed = 4247505863
# run_seed = int(np.random.default_rng().integers(0, 2**32, dtype=np.uint32))
real_pars = {"mu": -0.86, "sigma2_eta": 0.0225, "phi": 0.97}
T_obs = 200


N= 20000 # number of MCMC iterations
x_0 = xstart(mu = -0.6, phi = 0.8, sigma2_eta = 0.02) # the starting parameter values for the chain


# generating the data
stochvol.generate(mu = real_pars["mu"], phi = real_pars["phi"], sigma2_eta = real_pars["sigma2_eta"], T = T_obs, seed = run_seed)
y_gen = stochvol.ys
h_gen = stochvol.hs


#------------------------------------PMCMC standard no correlation---------------------
t0 = time.perf_counter()

print("starting PM_IS_std_nocorrel")
PM_IS_std_nocorrel = PM_IS(ys = y_gen, N_mcmc = N, x_first = x_0, s = 0.3, m_latent = 900, burnin = 5000, rho = 0)


t1 = time.perf_counter()
t_PM_IS_std_nocorrel = t1 - t0           
print(f"PM_IS_std_nocorrel - completed. time: {t_PM_IS_std_nocorrel} seconds\n")                       
#--------------------------------------------------------------------------------------


#------------------------------------PMCMC standard with correlation-------------------
print("starting PM_IS_std_correl \n")
PM_IS_std_correl = PM_IS(ys = y_gen, N_mcmc = N, x_first = x_0, s = 0.3, m_latent = 300, burnin = 5000, rho = 0.99)


t2 = time.perf_counter()
t_PM_IS_std_correl = t2 - t1         
print(f"PM_IS_std_correl - completed. time: {t_PM_IS_std_correl} seconds\n")               
#--------------------------------------------------------------------------------------


#------------------------------------PMCMC adaptive no correlation---------------------
print("starting PM_IS_adapt_nocorrel")
PM_IS_adapt_nocorrel = PM_IS_adaptive(ys = y_gen, N_mcmc = N, x_first = x_0, s = 1.88, m_latent = 900, burnin = 5000, rho = 0)


t3 = time.perf_counter()
t_PM_IS_adapt_nocorrel = t3 - t2
print(f"PM_IS_adapt_nocorrel - completed. time: {t_PM_IS_adapt_nocorrel} seconds\n")                           
#--------------------------------------------------------------------------------------


#-------------------------------------PMCMC adaptive with correlation------------------
print("starting PM_IS_adapt_correl")
PM_IS_adapt_correl = PM_IS_adaptive(ys = y_gen, N_mcmc = N, x_first = x_0, s = 1.5, m_latent = 300, burnin = 5000, rho = 0.99)


t4 = time.perf_counter()
t_PM_IS_adapt_correl = t4 - t3        
print(f"PM_IS_adapt_correl - completed. time {t_PM_IS_adapt_correl} seconds\n")                
#---------------------------------------------------------------------------------------

# to plot the MCMC chains
fig, axes = plt.subplots(nrows=2,
                         ncols=2,
                         figsize=(16, 12),
                         sharex=True)

# setting the background color and axes color
fig.patch.set_facecolor('#f0f0f0')
ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]

ax1.set_facecolor('white')
ax2.set_facecolor('white')
ax3.set_facecolor('white')

# chain for the mu parameters
ax1.set_xlabel("iteration")
ax1.set_ylabel("mu")
ax1.plot(PM_IS_std_nocorrel["mu_draws"], color = 'tab:blue', label = "standard PMCMC - no correl.")
ax1.plot(PM_IS_std_correl["mu_draws"], color = 'tab:orange', label = "standard PMCMC - with correl.")
ax1.plot(PM_IS_adapt_nocorrel["mu_draws"], color = 'tab:green', label = "PMCMC with adaptive jumps - no correl.")
ax1.plot(PM_IS_adapt_correl["mu_draws"], color = 'tab:olive', label = "PMCMC with adaptive jumps - with correl.")
ax1.axhline(y = real_pars["mu"], linestyle = "--", color = "gray", label = "true mu")
ax1.legend()

# chain for the sigma2_eta parameters
ax2.set_xlabel("iteration")
ax2.set_ylabel("sigma2_eta")
ax2.plot(PM_IS_std_nocorrel["sigma2_draws"], color = 'tab:blue', label = "standard PMCMC - no correl.")
ax2.plot(PM_IS_std_correl["sigma2_draws"], color = 'tab:orange', label = "standard PMCMC - with correl.")
ax2.plot(PM_IS_adapt_nocorrel["sigma2_draws"], color = 'tab:green', label = "PMCMC with adaptive jumps - no correl.")
ax2.plot(PM_IS_adapt_correl["sigma2_draws"], color = 'tab:olive', label = "PMCMC with adaptive jumps - with correl.")
ax2.axhline(y = real_pars["sigma2_eta"], linestyle = "--", color = "gray", label = "true sigma2_eta")
ax2.legend()

# chain for the phi parameters
ax3.set_xlabel("iteration")
ax3.set_ylabel("phi")
ax3.plot(PM_IS_std_nocorrel["phi_draws"], color = 'tab:blue', label = "standard PMCMC - no correl.")
ax3.plot(PM_IS_std_correl["phi_draws"], color = 'tab:orange', label = "standard PMCMC - with correl.")
ax3.plot(PM_IS_adapt_nocorrel["phi_draws"], color = 'tab:green', label = "PMCMC with adaptive jumps - no correl.")
ax3.plot(PM_IS_adapt_correl["phi_draws"], color = 'tab:olive', label = "PMCMC with adaptive jumps - with correl.")
ax3.axhline(y = real_pars["phi"], linestyle = "--", color = "gray", label = "true phi")
ax3.legend()




plt.show()


# ------- below we create InferenceData objects to get the effective sample size
idata_PM_IS_std_nocorrel  = az.from_dict(
        posterior={
            "mu":   PM_IS_std_nocorrel["mu_burnin_draws"][None, :],       # (1, N) shape needed for az
            "sigma2_eta": PM_IS_std_nocorrel["sigma2_burnin_draws"][None, :],
            "phi":  PM_IS_std_nocorrel["phi_burnin_draws"][None, :]
        } 
    )       
# pseudo-marginal
idata_PM_IS_std_correl = az.from_dict(
        posterior={
            "mu":   PM_IS_std_correl["mu_burnin_draws"][None, :],       
            "sigma2_eta": PM_IS_std_correl["sigma2_burnin_draws"][None, :],
            "phi":  PM_IS_std_correl["phi_burnin_draws"][None, :]
        } 
    )        
# correlated pseudo-marginal    
idata_PM_IS_adapt_nocorrel = az.from_dict(
        posterior={
            "mu":   PM_IS_adapt_nocorrel["mu_burnin_draws"][None, :],       
            "sigma2_eta": PM_IS_adapt_nocorrel["sigma2_burnin_draws"][None, :],
            "phi":  PM_IS_adapt_nocorrel["phi_burnin_draws"][None, :]
        } 
    )   

idata_PM_IS_adapt_correl = az.from_dict(
        posterior={
            "mu":   PM_IS_adapt_correl["mu_burnin_draws"][None, :],       
            "sigma2_eta": PM_IS_adapt_correl["sigma2_burnin_draws"][None, :],
            "phi":  PM_IS_adapt_correl["phi_burnin_draws"][None, :]
        } 
    )   



# effective sample sizes
ess_PM_IS_std_nocorrel = az.ess(idata_PM_IS_std_nocorrel)
ess_PM_IS_std_correl = az.ess(idata_PM_IS_std_correl)
ess_PM_IS_adapt_nocorrel = az.ess(idata_PM_IS_adapt_nocorrel)
ess_PM_IS_adapt_correl = az.ess(idata_PM_IS_adapt_correl)

# per second effective sample sizes for each method
PM_IS_std_nocorrel_persec = ess_PM_IS_std_nocorrel/t_PM_IS_std_nocorrel
PM_IS_std_correl_persec = ess_PM_IS_std_correl/t_PM_IS_std_correl
PM_IS_adapt_nocorrel_persec = ess_PM_IS_adapt_nocorrel/t_PM_IS_adapt_nocorrel
PM_IS_adapt_correl_persec = ess_PM_IS_adapt_correl/t_PM_IS_adapt_correl

# the 95 percent conf. interval. =1 if included, =0 if not included
"""
values["pars_95"] = {"mu": (lo_mu, hi_mu),
        "sigma2_eta": (lo_sigma2, hi_sigma2),
        "phi": (lo_phi, hi_phi)}
"""
##---------------------------95 percent confidence intervals------------------

# The dicts to store the 1 or 0 values indicating belonging to 95 percent conf. interval.
PM_IS_std_nocorrel_95 = {"mu": float(0),"sigma2_eta": float(0),"phi": float(0)}
PM_IS_std_correl_95 = PM_IS_std_nocorrel_95.copy()
PM_IS_adapt_nocorrel_95 = PM_IS_std_nocorrel_95.copy()
PM_IS_adapt_correl_95 = PM_IS_std_nocorrel_95.copy()



for p in ("mu", "sigma2_eta", "phi"):
    if PM_IS_std_nocorrel["pars_95"][p][0] <= real_pars[p] <= PM_IS_std_nocorrel["pars_95"][p][1]:
        PM_IS_std_nocorrel_95[p] = 1
    else:
        PM_IS_std_nocorrel_95[p] = 0

    if PM_IS_std_correl["pars_95"][p][0] <= real_pars[p] <= PM_IS_std_correl["pars_95"][p][1]:
        PM_IS_std_correl_95[p] = 1
    else:
        PM_IS_std_correl_95[p] = 0

    if PM_IS_adapt_nocorrel["pars_95"][p][0] <= real_pars[p] <= PM_IS_adapt_nocorrel["pars_95"][p][1]:
        PM_IS_adapt_nocorrel_95[p] = 1
    else:
        PM_IS_adapt_nocorrel_95[p] = 0

    if PM_IS_adapt_correl["pars_95"][p][0] <= real_pars[p] <= PM_IS_adapt_correl["pars_95"][p][1]:
        PM_IS_adapt_correl_95[p] = 1
    else:
        PM_IS_adapt_correl_95[p] = 0



##---------------------------BIAS and estimators------------------------------

##--PM_IS_std_nocorrel----------------------------
PM_IS_std_nocorrel_pars_est = {
"MCMC_method": "PM_IS standard - no correlation",
"acc. ratio": PM_IS_std_nocorrel["acc_ratio"],
"total time": t_PM_IS_std_nocorrel,
"mu - ESS": float(ess_PM_IS_std_nocorrel["mu"]),
"sigma2 - ESS": float(ess_PM_IS_std_nocorrel["sigma2_eta"]),
"phi - ESS": float(ess_PM_IS_std_nocorrel["phi"]),
"mu - ESS per sec": float(PM_IS_std_nocorrel_persec["mu"]),
"sigma2 - ESS per sec": float(PM_IS_std_nocorrel_persec["sigma2_eta"]),
"phi - ESS per sec": float(PM_IS_std_nocorrel_persec["phi"]),
"mu - ESS per iter.": float(ess_PM_IS_std_nocorrel["mu"])/N,
"sigma2 - ESS per iter.": float(ess_PM_IS_std_nocorrel["sigma2_eta"])/N,
"phi - ESS per iter.": float(ess_PM_IS_std_nocorrel["phi"])/N,
"est_mu": np.mean(PM_IS_std_nocorrel["mu_burnin_draws"]),
"real_mu": real_pars["mu"],
"est_sigma2": np.mean(PM_IS_std_nocorrel["sigma2_burnin_draws"]),
"real_sigma2": real_pars["sigma2_eta"],
"est_phi": np.mean(PM_IS_std_nocorrel["phi_burnin_draws"]),
"real_phi": real_pars["phi"],
"bias_mu": np.mean(PM_IS_std_nocorrel["mu_burnin_draws"]) - real_pars["mu"],
"bias_sigma2": np.mean(PM_IS_std_nocorrel["sigma2_burnin_draws"]) - real_pars["sigma2_eta"],
"bias_phi": np.mean(PM_IS_std_nocorrel["phi_burnin_draws"]) - real_pars["phi"],
"mu_in95": PM_IS_std_nocorrel_95["mu"],
"sigma2_in95":PM_IS_std_nocorrel_95["sigma2_eta"],
"phi_in95":PM_IS_std_nocorrel_95["phi"],
}

##--PM_IS_std_correl------------------------------
PM_IS_std_correl_pars_est = {
"MCMC_method": "PM_IS standard - with correlation",
"acc. ratio": PM_IS_std_correl["acc_ratio"],
"total time": t_PM_IS_std_correl,
"mu - ESS": float(ess_PM_IS_std_correl["mu"]),
"sigma2 - ESS": float(ess_PM_IS_std_correl["sigma2_eta"]),
"phi - ESS": float(ess_PM_IS_std_correl["phi"]),
"mu - ESS per sec": float(PM_IS_std_correl_persec["mu"]),
"sigma2 - ESS per sec": float(PM_IS_std_correl_persec["sigma2_eta"]),
"phi - ESS per sec": float(PM_IS_std_correl_persec["phi"]),
"mu - ESS per iter.": float(ess_PM_IS_std_correl["mu"])/N,
"sigma2 - ESS per iter.": float(ess_PM_IS_std_correl["sigma2_eta"])/N,
"phi - ESS per iter.": float(ess_PM_IS_std_correl["phi"])/N,
"est_mu": np.mean(PM_IS_std_correl["mu_burnin_draws"]),
"real_mu": real_pars["mu"],
"est_sigma2": np.mean(PM_IS_std_correl["sigma2_burnin_draws"]),
"real_sigma2": real_pars["sigma2_eta"],
"est_phi": np.mean(PM_IS_std_correl["phi_burnin_draws"]),
"real_phi": real_pars["phi"],
"bias_mu": np.mean(PM_IS_std_correl["mu_burnin_draws"]) - real_pars["mu"],
"bias_sigma2": np.mean(PM_IS_std_correl["sigma2_burnin_draws"]) - real_pars["sigma2_eta"],
"bias_phi": np.mean(PM_IS_std_correl["phi_burnin_draws"]) - real_pars["phi"],
"mu_in95": PM_IS_std_correl_95["mu"],
"sigma2_in95":PM_IS_std_correl_95["sigma2_eta"],
"phi_in95":PM_IS_std_correl_95["phi"],
}

##--PM_IS_adapt_nocorrel------------------------------
PM_IS_adapt_nocorrel_pars_est = {
"MCMC_method": "PM_IS adapt. - no correlation",
"acc. ratio": PM_IS_adapt_nocorrel["acc_ratio"],
"total time": t_PM_IS_adapt_nocorrel,
"mu - ESS": float(ess_PM_IS_adapt_nocorrel["mu"]),
"sigma2 - ESS": float(ess_PM_IS_adapt_nocorrel["sigma2_eta"]),
"phi - ESS": float(ess_PM_IS_adapt_nocorrel["phi"]),
"mu - ESS per sec": float(PM_IS_adapt_nocorrel_persec["mu"]),
"sigma2 - ESS per sec": float(PM_IS_adapt_nocorrel_persec["sigma2_eta"]),
"phi - ESS per sec": float(PM_IS_adapt_nocorrel_persec["phi"]),
"mu - ESS per iter.": float(ess_PM_IS_adapt_nocorrel["mu"])/N,
"sigma2 - ESS per iter.": float(ess_PM_IS_adapt_nocorrel["sigma2_eta"])/N,
"phi - ESS per iter.": float(ess_PM_IS_adapt_nocorrel["phi"])/N,
"est_mu": np.mean(PM_IS_adapt_nocorrel["mu_burnin_draws"]),
"real_mu": real_pars["mu"],
"est_sigma2": np.mean(PM_IS_adapt_nocorrel["sigma2_burnin_draws"]),
"real_sigma2": real_pars["sigma2_eta"],
"est_phi": np.mean(PM_IS_adapt_nocorrel["phi_burnin_draws"]),
"real_phi": real_pars["phi"],
"bias_mu": np.mean(PM_IS_adapt_nocorrel["mu_burnin_draws"]) - real_pars["mu"],
"bias_sigma2": np.mean(PM_IS_adapt_nocorrel["sigma2_burnin_draws"]) - real_pars["sigma2_eta"],
"bias_phi": np.mean(PM_IS_adapt_nocorrel["phi_burnin_draws"]) - real_pars["phi"],
"mu_in95": PM_IS_adapt_nocorrel_95["mu"],
"sigma2_in95":PM_IS_adapt_nocorrel_95["sigma2_eta"],
"phi_in95":PM_IS_adapt_nocorrel_95["phi"],
}

##--PM_IS_adapt_correl--------------------------------
PM_IS_adapt_correl_pars_est = {
"MCMC_method": "PM_IS adapt. - with correlation",
"acc. ratio": PM_IS_adapt_correl["acc_ratio"],
"total time": t_PM_IS_adapt_correl,
"mu - ESS": float(ess_PM_IS_adapt_correl["mu"]),
"sigma2 - ESS": float(ess_PM_IS_adapt_correl["sigma2_eta"]),
"phi - ESS": float(ess_PM_IS_adapt_correl["phi"]),
"mu - ESS per sec": float(PM_IS_adapt_correl_persec["mu"]),
"sigma2 - ESS per sec": float(PM_IS_adapt_correl_persec["sigma2_eta"]),
"phi - ESS per sec": float(PM_IS_adapt_correl_persec["phi"]),
"mu - ESS per iter.": float(ess_PM_IS_adapt_correl["mu"])/N,
"sigma2 - ESS per iter.": float(ess_PM_IS_adapt_correl["sigma2_eta"])/N,
"phi - ESS per iter.": float(ess_PM_IS_adapt_correl["phi"])/N,
"est_mu": np.mean(PM_IS_adapt_correl["mu_burnin_draws"]),
"real_mu": real_pars["mu"],
"est_sigma2": np.mean(PM_IS_adapt_correl["sigma2_burnin_draws"]),
"real_sigma2": real_pars["sigma2_eta"],
"est_phi": np.mean(PM_IS_adapt_correl["phi_burnin_draws"]),
"real_phi": real_pars["phi"],
"bias_mu": np.mean(PM_IS_adapt_correl["mu_burnin_draws"]) - real_pars["mu"],
"bias_sigma2": np.mean(PM_IS_adapt_correl["sigma2_burnin_draws"]) - real_pars["sigma2_eta"],
"bias_phi": np.mean(PM_IS_adapt_correl["phi_burnin_draws"]) - real_pars["phi"],
"mu_in95": PM_IS_adapt_correl_95["mu"],
"sigma2_in95":PM_IS_adapt_correl_95["sigma2_eta"],
"phi_in95":PM_IS_adapt_correl_95["phi"],
}

full_info = [PM_IS_std_nocorrel_pars_est, PM_IS_std_correl_pars_est, PM_IS_adapt_nocorrel_pars_est, PM_IS_adapt_correl_pars_est]
df1 = pd.DataFrame(full_info)

fname = f"seed_{run_seed}_T_{T_obs}_results.csv"
df1.to_csv(fname, index=False)
print("Saved:", fname)
"""
To calculate the bias - run experiment 30 times with different DGP (different seeds).
Then get the average of the individual biases to calculate the full bias.

For root mean squared error - square of the bias - (estimator["mu] - real_pars["mu"])**2
and get the average across the chains
"""