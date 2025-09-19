import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import arviz as az


script_dir = Path(__file__).parent
'''
This file saves the rank plots, convergence diagnostics and mixing stats in the same folder where the csv files are
the figures are saved as csv and pdf, the stats are saved as csv.
Specify the model details below and run the code

'''



#--------------------------- (I) choose model specifications ------------------------------------
'''
For the below specs - options are:

T = 200, 700
MCMC = "PMMH", "PM_IS"
jumps = "diag", "adapt"
corr = "correl", "nocorrel"
'''

real_pars = {"mu": -0.86, "sigma2": 0.0225, "phi": 0.98}
T = 200
MCMC_type = "PMMH"
jumps = "diag"
corr = "nocorrel"

burnin = 4000


#-----------------------(II) upload the files and convert to pandas DataFrame ---------------

model_folder = Path(f"{script_dir}/MCMC_chains")    # point to the folder where the files are stored in



dfs = []                                                                                        # a lits to store all dataframes
for fp in model_folder.glob(f"*{MCMC_type}*{jumps}*_{corr}*{T}*.csv"):
    df = pd.read_csv(fp)
    dfs.append(df)

N_chain = len(dfs[0]) - burnin                                                                  # the length of a single chain
m = len(dfs)                                                                                    # number of chains


#-----------------------(III) pool the per parameter draws from all chains ---------------

all_mu = np.full((m*N_chain), np.nan)
all_sigma2 = np.full((m*N_chain), np.nan)
all_phi = np.full((m*N_chain), np.nan)
acc_ratio_sum = 0
time_sum = 0

for i in range(m):
    mu_draws = np.array(dfs[i]["mu_draws"][burnin:])
    all_mu[i*N_chain:(i+1)*N_chain] = mu_draws

    sigma2_draws = np.array(dfs[i]["sigma2_draws"][burnin:])
    all_sigma2[i*N_chain:(i+1)*N_chain] = sigma2_draws

    phi_draws = np.array(dfs[i]["phi_draws"][burnin:])
    all_phi[i*N_chain:(i+1)*N_chain] = phi_draws

    time_sum = time_sum + dfs[i]["time"][0]                                             # average time across chains
    acc_ratio_sum = acc_ratio_sum + dfs[i]["acc_ratio"][0]

time = time_sum/m
acc_ratio = acc_ratio_sum/m


if np.nan in all_mu or np.nan in all_sigma2 or np.nan in all_phi:                                # Raise  VaueError if NANs found
    raise ValueError("NANs in an array")

#-----------------------(IV) get the sorted indices for full chains - rank of each entry ---

mu_indices = np.argsort(all_mu)
sigma2_indices = np.argsort(all_sigma2)
phi_indices = np.argsort(all_phi)

full_indices = {"mu": mu_indices,
                "sigma2": sigma2_indices,
                "phi": phi_indices}

# get the perchain sorted indices
perchain_indices = {"mu": [],
                "sigma2": [],
                "phi": []}

for j in perchain_indices:
    for i in range(m):
        perchain_indices[j].append(full_indices[j][i*N_chain:(i+1)*N_chain])

#-----------------------(V) Normalize the indices to (0,1) and extract the corresp. per chains -

normalized = {"mu": (mu_indices - 0.5)/(N_chain*m),
              "sigma2": (sigma2_indices - 0.5)/(N_chain*m),
              "phi": (phi_indices - 0.5)/(N_chain*m)

}


norm_perchain = {"mu": [],                                                                      # get the per chain indices
                "sigma2": [],
                "phi": []}

for j in norm_perchain:
    for i in range(m):
        norm_perchain[j].append(normalized[j][i*N_chain:(i+1)*N_chain])


# -----------------------(VI) Get the ECDFs ------------------------------------------------------
ECDFs = {"mu": [],
         "sigma2": [],
         "phi": []}

for j in ECDFs:
    for i in range(m):
        ECDF = np.arange(1, N_chain + 1)/N_chain
        ranks = np.argsort(np.argsort(norm_perchain[j][i]))
        ECDF = ECDF[ranks]
        ECDFs[j].append(ECDF)


# -----------------------(VII) Get the diagnostics for parameters ----------------------

az_arrays = {"mu": np.reshape(all_mu, (m, N_chain)),
             "sigma2": np.reshape(all_sigma2, (m, N_chain)),
             "phi": np.reshape(all_phi, (m, N_chain))}

idata = az.from_dict(                                                                           # this is the idata format for arviz
    posterior = az_arrays
)


ESS_avg = {"mu":np.full((az_arrays["mu"].shape[0],), np.nan),                                      # the average ESS (if we have different obs. sets)
            "sigma2":np.full((az_arrays["sigma2"].shape[0],), np.nan),                                 # note that this is only used for the within chain case of phase2
            "phi":np.full((az_arrays["phi"].shape[0],), np.nan)}                                    # for phase1 use bulk ess and tail ess from summary

for key in ESS_avg:    
    for i in range(az_arrays[key].shape[0]):
        ESS_avg[key][i] = az.ess(az_arrays[key][i])
    ESS_avg[key] = np.mean(ESS_avg[key])


df_summary = az.summary(idata, var_names=["mu","sigma2","phi"],                                 # summary dataframe to export
                 round_to=5)
df_summary["time"] = time
df_summary["acc. ratio"] = acc_ratio
df_summary["bias"] = np.array([np.mean(all_mu - real_pars["mu"]), np.mean(all_sigma2 - real_pars["sigma2"]), np.mean(all_phi - real_pars["phi"])])
df_summary["avg_ESS"] = np.array([ESS_avg["mu"], ESS_avg["sigma2"], ESS_avg["phi"]])




# -----------------------(VIII) Make the rank plots ---------------------------------------------

colors = plt.cm.viridis(np.linspace(0, 1, m))

fig, axes = plt.subplots(1,3, figsize = (35,10))
fig.supylabel('ECDF', fontsize = 20, fontweight = 'bold', x = 0.1)
fig.supxlabel('Uniform(0,1) Quantiles', fontsize = 20, fontweight = 'bold')

for i in range(3):
    axes[i].axline((0,0),slope = 1, color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0, label='_nolegend_')
axes[0].set_title("mu", fontsize = 16)
axes[1].set_title("sigma2", fontsize = 16)
axes[2].set_title("phi", fontsize = 16)
for i in range(m):
    axes[0].plot(norm_perchain["mu"][i][np.argsort(norm_perchain["mu"][i])],ECDFs["mu"][i][np.argsort(norm_perchain["mu"][i])], color=colors[i])
    axes[1].plot(norm_perchain["sigma2"][i][np.argsort(norm_perchain["sigma2"][i])],ECDFs["sigma2"][i][np.argsort(norm_perchain["sigma2"][i])], color=colors[i])
    axes[2].plot(norm_perchain["phi"][i][np.argsort(norm_perchain["phi"][i])],ECDFs["phi"][i][np.argsort(norm_perchain["phi"][i])], color=colors[i],)


# -----------------------(VIII) Save the plots and report ---------------------------------------------


save_path_pdf = f"{model_folder}/{MCMC_type}_T_{T}_{corr}_{jumps}.pdf"
save_path_png = f"{model_folder}/{MCMC_type}_T_{T}_{corr}_{jumps}.png"


df_summary.to_csv(f"{model_folder}/diagnostics_{MCMC_type}_T_{T}_{corr}_{jumps}.csv")             # save the df summary
print("summary saved as csv")

fig.savefig(save_path_png, dpi=300, bbox_inches='tight')
fig.savefig(save_path_pdf, bbox_inches='tight', transparent=False)
print(f"figures saved at {model_folder}")
plt.show()

