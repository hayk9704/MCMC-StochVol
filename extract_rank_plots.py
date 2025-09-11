import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

'''
My steps:

(i)  Get the rank plots here as a visual inspection - this will sho you if the chains actually
     target the same posterior distribution. Use 10 chains for each sampler

(ii) Next make sure to look into some numeric reports to understand if they target the same d.

'''
#listing all models here
model_T700_nocorrel_std = "T = 700, non-correlated PMMH - gaussian jumps"
model_T700_correl_std = "T = 700, correlated PMMH - gaussian jumps"
model_T700_nocorrel_adapt = "T = 700, non-correlated PMMH - adaptive jumps"
model_T700_correl_adapt = "T = 700, correlated PMMH - adaptive jumps"

model_T200_nocorrel_std_PMMH = "T = 200, non-correlated PMMH - gaussian jumps"
model_T200_correl_std_PMMH = "T = 200, correlated PMMH - gaussian jumps"
model_T200_nocorrel_adapt_PMMH = "T = 200, non-correlated PMMH - adaptive jumps"
model_T200_correl_adapt_PMMH = "T = 200, non-correlated PMMH - adaptive jumps"

model_T200_nocorrel_std_PM_IS = "T = 200, non-correlated PM-IS - gaussian jumps"
model_T200_correl_std_PM_IS = "T = 200, correlated PM-IS - gaussian jumps"
model_T200_nocorrel_adapt_PM_IS = "T = 200, non-correlated PM-IS - adaptive jumps"
model_T200_correl_adapt_PM_IS = "T = 200, non-correlated PM-IS - adaptive jumps"

# choose the model for the header of the graph
chosen_model = model_T700_correl_std
# choose the folder where the chains of a specific model are stored
model_folder = Path(r"C:\Users\haykg\Documents\chains_PMMH_t700\t700_std_correl")

burnin = 2500

dfs = []                                                                        # a lits to store all dataframes
for fp in model_folder.glob("*.csv"):
    df = pd.read_csv(fp)
    dfs.append(df)

N_chain = len(dfs[0]) - burnin # the length of a single chain
m = len(dfs) # number of chains


all_mu = np.full((m*N_chain), np.nan)
all_sigma2 = np.full((m*N_chain), np.nan)
all_phi = np.full((m*N_chain), np.nan)

for i in range(m):
    mu_draws = np.array(dfs[i]["mu_draws"][burnin:])
    all_mu[i*N_chain:(i+1)*N_chain] = mu_draws

    sigma2_draws = np.array(dfs[i]["sigma2_draws"][burnin:])
    all_sigma2[i*N_chain:(i+1)*N_chain] = sigma2_draws

    phi_draws = np.array(dfs[i]["phi_draws"][burnin:])
    all_phi[i*N_chain:(i+1)*N_chain] = phi_draws

if np.nan in all_mu or np.nan in all_sigma2 or np.nan in all_phi:
    raise ValueError("NANs in an array")

# get the sorted indices for the full chains- shows which rank each entry has
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

# Normalize the indices to (0,1)
normalized = {"mu": (mu_indices - 0.5)/(N_chain*m),
              "sigma2": (sigma2_indices - 0.5)/(N_chain*m),
              "phi": (phi_indices - 0.5)/(N_chain*m)

}


# get the per chain indices

norm_perchain = {"mu": [],
                "sigma2": [],
                "phi": []}

for j in norm_perchain:
    for i in range(m):
        norm_perchain[j].append(normalized[j][i*N_chain:(i+1)*N_chain])


# making the ECDFs

ECDFs = {"mu": [],
         "sigma2": [],
         "phi": []}

for j in ECDFs:
    for i in range(m):
        ECDF = np.arange(1, N_chain + 1)/N_chain
        ranks = np.argsort(np.argsort(norm_perchain[j][i]))
        ECDF = ECDF[ranks]
        ECDFs[j].append(ECDF)

colors = plt.cm.viridis(np.linspace(0, 1, m))

fig, axes = plt.subplots(1,3, figsize = (35,10))
fig.suptitle(f'{chosen_model}', fontsize = 40, fontweight = 'bold')
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


# save the figure both as pdf and png
save_path_pdf = f"{model_folder}/{chosen_model}.pdf"
save_path_png = f"{model_folder}/{chosen_model}.png"

fig.savefig(save_path_png, dpi=300, bbox_inches='tight')
fig.savefig(save_path_pdf, bbox_inches='tight', transparent=False)
print(f"figure samed as{model_folder}//{chosen_model}")
plt.show()

