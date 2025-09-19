import time
import numpy as np
import pandas as pd
from MCMC_functions import stochvol, theta_to_x as xstart
from PMMH_adaptive import PMCMC as PMCMC_correl_adapt
from PMMH import PMCMC as PMCMC_correl_std
from PM_IS_adaptive import PM_IS_adaptive
from PM_IS import PM_IS
import os

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()



"""
This file runs the MCMC algorithms and saves each chain as a (N)x5 CSV file
The 4_th column is the time of the chain, 5th column is the acc. ratio 
Methods - either PMMH or PM_IS
"""


#------------------------------------Part 2 - Running the simulations-------------------------------
#---------------------------------------------------------------------------------------------------

def main(s_PMMH_diag_nocorrel = 1.88, s_PMMH_diag_correl = 1.88, \
        s_PMMH_adapt_nocorrel = 1.88, s_PMMH_adapt_correl = 1.88, s_PM_IS_diag_nocorrel = 1.88, s_PM_IS_diag_correl = 1.88, s_PM_IS_adapt_nocorrel = 1.88,\
        s_PM_IS_adapt_correl = 1.88, m_PMMH_diag_nocorrel = 50, m_PMMH_diag_correl = 50, m_PMMH_adapt_nocorrel = 50, m_PMMH_adapt_correl = 50, \
        m_PM_IS_diag_nocorrel = 50, m_PM_IS_diag_correl = 50, m_PM_IS_adapt_nocorrel = 50, m_PM_IS_adapt_correl = 50, rho_PMMH_diag_correl = 0.99,\
        rho_PMMH_adapt_correl = 0.99, rho_PM_IS_diag_correl = 0.99, rho_PM_IS_adapt_correl = 0.99,\
        method = "PMMH", T_obs = 700, run_seed = int(np.random.default_rng().integers(0, 2**32, \
        dtype=np.uint32)), real_pars = {"mu": -0.86, "sigma2_eta": 0.0225, "phi": 0.97},\
        N_mcmc = 15000, burnin = 5000, x_0 = xstart(mu = -0.9, phi = 0.3, sigma2_eta = 0.7), ):
    
    output_dir = f"{script_dir}/MCMC_chains"
    os.makedirs(output_dir, exist_ok=True)

    stochvol.generate(mu = real_pars["mu"], phi = real_pars["phi"], sigma2_eta = real_pars["sigma2_eta"], T = T_obs, seed = run_seed)
    y_gen = stochvol.ys

    if method == "PMMH":
        PMMH_results = run_PMMH(y_gen = y_gen, N_mcmc = N_mcmc, burnin = burnin, x_0 = x_0, s_diag_nocorrel = s_PMMH_diag_nocorrel, s_diag_correl = s_PMMH_diag_correl, \
                 s_adapt_nocorrel = s_PMMH_adapt_nocorrel, s_adapt_correl = s_PMMH_adapt_correl, m_diag_nocorrel = m_PMMH_diag_nocorrel, 
             m_diag_correl = m_PMMH_diag_correl, m_adapt_nocorrel = m_PMMH_adapt_nocorrel, m_adapt_correl = m_PMMH_adapt_correl, \
                rho_diag_correl = rho_PMMH_diag_correl, rho_adapt_correl = rho_PMMH_adapt_correl)
        for key in PMMH_results:
            PMMH_results[key].to_csv(f"{output_dir}/{key}_T_{T_obs}_{run_seed}_{np.random.choice(10**5)}.csv", index = False)
        
        print("All PMMH chains saved")


    elif method == "PM_IS":
        PM_IS_results = run_PM_IS(y_gen = y_gen, N_mcmc = N_mcmc, burnin = burnin, x_0 = x_0, s_diag_nocorrel = s_PM_IS_diag_nocorrel, s_diag_correl = s_PM_IS_diag_correl, \
                  s_adapt_nocorrel = s_PM_IS_adapt_nocorrel, s_adapt_correl = s_PM_IS_adapt_correl, m_diag_nocorrel = m_PM_IS_diag_nocorrel, 
             m_diag_correl = m_PM_IS_diag_correl, m_adapt_nocorrel = m_PM_IS_adapt_nocorrel, m_adapt_correl = m_PM_IS_adapt_correl, \
                rho_diag_correl = rho_PM_IS_diag_correl, rho_adapt_correl = rho_PM_IS_adapt_correl)
        for key in PM_IS_results:
            PM_IS_results[key].to_csv(f"{output_dir}/{key}_T_{T_obs}_{run_seed}_{np.random.choice(10**5)}.csv", index = False)
        
        print("All PM-IS chains saved")
    
    elif method == "PMMH_adapt":
        PMMH_adapt_results = run_PMMH_adapt(y_gen = y_gen, N_mcmc = N_mcmc, burnin = burnin, x_0 = x_0, s_adapt_nocorrel = s_PMMH_adapt_nocorrel, \
                                            s_adapt_correl = s_PMMH_adapt_correl, m_adapt_nocorrel = m_PMMH_adapt_nocorrel, \
                                            m_adapt_correl = m_PMMH_adapt_correl, rho_adapt_correl = rho_PMMH_adapt_correl)
        for key in PMMH_adapt_results:
            PMMH_adapt_results[key].to_csv(f"{output_dir}/{key}_T_{T_obs}_{run_seed}_{np.random.choice(10**5)}.csv", index = False)
        
        print("All adaptive PMMH chains saved")
        
    elif method =="both":

        PMMH_results = run_PMMH(y_gen = y_gen, N_mcmc = N_mcmc, burnin = burnin, x_0 = x_0, s_diag_nocorrel = s_PMMH_diag_nocorrel, s_diag_correl = s_PMMH_diag_correl, \
                 s_adapt_nocorrel = s_PMMH_adapt_nocorrel, s_adapt_correl = s_PMMH_adapt_correl, m_diag_nocorrel = m_PMMH_diag_nocorrel, 
             m_diag_correl = m_PMMH_diag_correl, m_adapt_nocorrel = m_PMMH_adapt_nocorrel, m_adapt_correl = m_PMMH_adapt_correl, \
                rho_diag_correl = rho_PMMH_diag_correl, rho_adapt_correl = rho_PMMH_adapt_correl)
        
        for key in PMMH_results:
            PMMH_results[key].to_csv(f"{output_dir}/{key}_T_{T_obs}_{run_seed}_{np.random.choice(10**5)}.csv", index = False)
        
        print("All PMMH chains saved")

        
        PM_IS_results = run_PM_IS(y_gen = y_gen, N_mcmc = N_mcmc, burnin = burnin, x_0 = x_0, s_diag_nocorrel = s_PM_IS_diag_nocorrel, \
                                s_diag_correl = s_PM_IS_diag_correl, s_adapt_nocorrel = s_PM_IS_adapt_nocorrel, s_adapt_correl = s_PM_IS_adapt_correl, \
                                m_diag_nocorrel = m_PM_IS_diag_nocorrel, m_diag_correl = m_PM_IS_diag_correl, m_adapt_nocorrel = m_PM_IS_adapt_nocorrel,\
                                m_adapt_correl = m_PM_IS_adapt_correl, rho_diag_correl = rho_PM_IS_diag_correl, rho_adapt_correl = rho_PM_IS_adapt_correl)
        
        for key in PM_IS_results:
            PM_IS_results[key].to_csv(f"{output_dir}/{key}_T_{T_obs}_{run_seed}_{np.random.choice(10**5)}.csv", index = False)        
        print("All PM-IS chains saved")

        

#---------------------------PMCMC standard no correlation---------------------
def run_PMMH_adapt(y_gen, N_mcmc, burnin, x_0, s_adapt_nocorrel, s_adapt_correl, m_adapt_nocorrel, m_adapt_correl, rho_adapt_correl):

    t2 = time.perf_counter()                     

    #---------------------------PMCMC adaptive no correlation----------------------
    print("starting PMMH_adapt_nocorrel")
    PMMH_adapt_nocorrel = PMCMC_correl_adapt(ys = y_gen, N_mcmc = N_mcmc, x_first = x_0, s = s_adapt_nocorrel, m_latent = m_adapt_nocorrel, burnin = burnin, rho = 0)


    t3 = time.perf_counter()
    t_PMMH_adapt_nocorrel = t3 - t2
    print(f"PMMH_adapt_nocorrel - completed. time: {t_PMMH_adapt_nocorrel} seconds\n")                           
    #--------------------------------------------------------------------------------------


    #-------------------------------------PMCMC adaptive with correlation------------------
    print("starting PMMH_adapt_correl")
    PMMH_adapt_correl = PMCMC_correl_adapt(ys = y_gen, N_mcmc = N_mcmc, x_first = x_0, s = s_adapt_correl, m_latent = m_adapt_correl, burnin = burnin, rho = rho_adapt_correl)


    t4 = time.perf_counter()
    t_PMMH_adapt_correl = t4 - t3        
    print(f"PMMH_adapt_correl - completed. time {t_PMMH_adapt_correl} seconds\n")            

    PMMH_adapt_nocorrel_results = pd.DataFrame({"mu_draws": PMMH_adapt_nocorrel["mu_draws"],
                          "sigma2_draws": PMMH_adapt_nocorrel["sigma2_draws"],
                          "phi_draws": PMMH_adapt_nocorrel["phi_draws"],
                          "time": t_PMMH_adapt_nocorrel,
                          "acc_ratio": PMMH_adapt_nocorrel["acc_ratio"]})
    PMMH_adapt_correl_results = pd.DataFrame({"mu_draws": PMMH_adapt_correl["mu_draws"],
                          "sigma2_draws": PMMH_adapt_correl["sigma2_draws"],
                          "phi_draws": PMMH_adapt_correl["phi_draws"],
                          "time": t_PMMH_adapt_correl,
                          "acc_ratio": PMMH_adapt_correl["acc_ratio"]})
    
    return {
            "PMMH_adapt_nocorrel": PMMH_adapt_nocorrel_results, 
            "PMMH_adapt_correl": PMMH_adapt_correl_results}
     



#---------------------------PMCMC standard no correlation---------------------
def run_PMMH(y_gen, N_mcmc, burnin, x_0, s_diag_nocorrel, s_diag_correl, s_adapt_nocorrel, s_adapt_correl, m_diag_nocorrel, 
             m_diag_correl, m_adapt_nocorrel, m_adapt_correl, rho_diag_correl, rho_adapt_correl):
    t0 = time.perf_counter()

    print("starting PMMH_diag_nocorrel")
    PMMH_diag_nocorrel = PMCMC_correl_std(ys = y_gen, N_mcmc = N_mcmc, x_first = x_0, s = s_diag_nocorrel, m_latent = m_diag_nocorrel, burnin = burnin, rho = 0)


    t1 = time.perf_counter()
    t_PMMH_diag_nocorrel = t1 - t0           
    print(f"PMMH_diag_nocorrel - completed. time: {t_PMMH_diag_nocorrel} seconds\n")                       
    #-----------------------------------------------------------------------------


    #---------------------------PMCMC standard with correlation-------------------
    print("starting PMMH_diag_correl \n")
    PMMH_diag_correl = PMCMC_correl_std(ys = y_gen, N_mcmc = N_mcmc, x_first = x_0, s = s_diag_correl, m_latent = m_diag_correl, burnin = burnin, rho = rho_diag_correl)


    t2 = time.perf_counter()
    t_PMMH_diag_correl = t2 - t1         
    print(f"PMMH_diag_correl - completed. time: {t_PMMH_diag_correl} seconds\n")               
    #------------------------------------------------------------------------------


    #---------------------------PMCMC adaptive no correlation----------------------
    print("starting PMMH_adapt_nocorrel")
    PMMH_adapt_nocorrel = PMCMC_correl_adapt(ys = y_gen, N_mcmc = N_mcmc, x_first = x_0, s = s_adapt_nocorrel, m_latent = m_adapt_nocorrel, burnin = burnin, rho = 0)


    t3 = time.perf_counter()
    t_PMMH_adapt_nocorrel = t3 - t2
    print(f"PMMH_adapt_nocorrel - completed. time: {t_PMMH_adapt_nocorrel} seconds\n")                           
    #--------------------------------------------------------------------------------------


    #-------------------------------------PMCMC adaptive with correlation------------------
    print("starting PMMH_adapt_correl")
    PMMH_adapt_correl = PMCMC_correl_adapt(ys = y_gen, N_mcmc = N_mcmc, x_first = x_0, s = s_adapt_correl, m_latent = m_adapt_correl, burnin = burnin, rho = rho_adapt_correl)


    t4 = time.perf_counter()
    t_PMMH_adapt_correl = t4 - t3        
    print(f"PMMH_adapt_correl - completed. time {t_PMMH_adapt_correl} seconds\n")            

    PMMH_diag_nocorrel_results = pd.DataFrame({"mu_draws": PMMH_diag_nocorrel["mu_draws"],
                          "sigma2_draws": PMMH_diag_nocorrel["sigma2_draws"],
                          "phi_draws": PMMH_diag_nocorrel["phi_draws"],
                          "time": t_PMMH_diag_nocorrel,
                          "acc_ratio": PMMH_diag_nocorrel["acc_ratio"]})
    PMMH_diag_correl_results = pd.DataFrame({"mu_draws": PMMH_diag_correl["mu_draws"],
                          "sigma2_draws": PMMH_diag_correl["sigma2_draws"],
                          "phi_draws": PMMH_diag_correl["phi_draws"],
                          "time": t_PMMH_diag_correl,
                          "acc_ratio": PMMH_diag_correl["acc_ratio"]})
    PMMH_adapt_nocorrel_results = pd.DataFrame({"mu_draws": PMMH_adapt_nocorrel["mu_draws"],
                          "sigma2_draws": PMMH_adapt_nocorrel["sigma2_draws"],
                          "phi_draws": PMMH_adapt_nocorrel["phi_draws"],
                          "time": t_PMMH_adapt_nocorrel,
                          "acc_ratio": PMMH_adapt_nocorrel["acc_ratio"]})
    PMMH_adapt_correl_results = pd.DataFrame({"mu_draws": PMMH_adapt_correl["mu_draws"],
                          "sigma2_draws": PMMH_adapt_correl["sigma2_draws"],
                          "phi_draws": PMMH_adapt_correl["phi_draws"],
                          "time": t_PMMH_adapt_correl,
                          "acc_ratio": PMMH_adapt_correl["acc_ratio"]})
    
    return {"PMMH_diag_nocorrel": PMMH_diag_nocorrel_results, 
            "PMMH_diag_correl": PMMH_diag_correl_results, 
            "PMMH_adapt_nocorrel": PMMH_adapt_nocorrel_results, 
            "PMMH_adapt_correl": PMMH_adapt_correl_results}
     
    #---------------------------------------------------------------------------------------

def run_PM_IS(y_gen, N, burnin, x_0, s_diag_nocorrel, s_diag_correl, s_adapt_nocorrel, s_adapt_correl, m_diag_nocorrel, 
             m_diag_correl, m_adapt_nocorrel, m_adapt_correl, rho_diag_correl, rho_adapt_correl):
    #------------------------------------PMCMC standard no correlation---------------------
    t0 = time.perf_counter()
    print("starting PM_IS_diag_nocorrel")
    PM_IS_diag_nocorrel = PM_IS(ys = y_gen, N_mcmc = N, x_first = x_0, s = s_diag_nocorrel, m_latent = m_diag_nocorrel, burnin = burnin, rho = 0)
    
    t1 = time.perf_counter()
    t_PM_IS_diag_nocorrel = t1 - t0           
    print(f"PM_IS_diag_nocorrel - completed. time: {t_PM_IS_diag_nocorrel} seconds\n")                       
    #--------------------------------------------------------------------------------------
    #------------------------------------PMCMC standard with correlation-------------------
    print("starting PM_IS_diag_correl \n")
    PM_IS_diag_correl = PM_IS(ys = y_gen, N_mcmc = N, x_first = x_0, s = s_diag_correl, m_latent = m_diag_correl, burnin = burnin, rho = rho_diag_correl)

    t2 = time.perf_counter()
    t_PM_IS_diag_correl = t2 - t1         
    print(f"PM_IS_diag_correl - completed. time: {t_PM_IS_diag_correl} seconds\n")               
    #--------------------------------------------------------------------------------------
    #------------------------------------PMCMC adaptive no correlation---------------------
    print("starting PM_IS_adapt_nocorrel")
    PM_IS_adapt_nocorrel = PM_IS_adaptive(ys = y_gen, N_mcmc = N, x_first = x_0, s = s_adapt_nocorrel, m_latent = m_adapt_nocorrel, burnin = burnin, rho = 0)

    t3 = time.perf_counter()
    t_PM_IS_adapt_nocorrel = t3 - t2
    print(f"PM_IS_adapt_nocorrel - completed. time: {t_PM_IS_adapt_nocorrel} seconds\n")                           
    #--------------------------------------------------------------------------------------
    #-------------------------------------PMCMC adaptive with correlation------------------
    print("starting PM_IS_adapt_correl")
    PM_IS_adapt_correl = PM_IS_adaptive(ys = y_gen, N_mcmc = N, x_first = x_0, s = s_adapt_correl, m_latent = m_adapt_correl, burnin = burnin, rho = rho_adapt_correl)

    t4 = time.perf_counter()
    t_PM_IS_adapt_correl = t4 - t3        
    print(f"PM_IS_adapt_correl - completed. time {t_PM_IS_adapt_correl} seconds\n")                
    #---------------------------------------------------------------------------------------

    PM_IS_diag_nocorrel_results = pd.DataFrame({"mu_draws": PM_IS_diag_nocorrel["mu_draws"],
                          "sigma2_draws": PM_IS_diag_nocorrel["sigma2_draws"],
                          "phi_draws": PM_IS_diag_nocorrel["phi_draws"],
                          "time": t_PM_IS_diag_nocorrel,
                          "acc_ratio": PM_IS_diag_nocorrel["acc_ratio"]})
    PM_IS_diag_correl_results = pd.DataFrame({"mu_draws": PM_IS_diag_correl["mu_draws"],
                          "sigma2_draws": PM_IS_diag_correl["sigma2_draws"],
                          "phi_draws": PM_IS_diag_correl["phi_draws"],
                          "time": t_PM_IS_diag_correl,
                          "acc_ratio": PM_IS_diag_correl["acc_ratio"]})
    PM_IS_adapt_nocorrel_results = pd.DataFrame({"mu_draws": PM_IS_adapt_nocorrel["mu_draws"],
                          "sigma2_draws": PM_IS_adapt_nocorrel["sigma2_draws"],
                          "phi_draws": PM_IS_adapt_nocorrel["phi_draws"],
                          "time": t_PM_IS_adapt_nocorrel,
                          "acc_ratio": PM_IS_adapt_nocorrel["acc_ratio"]})
    PM_IS_adapt_correl_results = pd.DataFrame({"mu_draws": PM_IS_adapt_correl["mu_draws"],
                          "sigma2_draws": PM_IS_adapt_correl["sigma2_draws"],
                          "phi_draws": PM_IS_adapt_correl["phi_draws"],
                          "time": t_PM_IS_adapt_correl,
                          "acc_ratio": PM_IS_adapt_correl["acc_ratio"]})
    
    return {"PM_IS_diag_nocorrel": PM_IS_diag_nocorrel_results, 
            "PM_IS_diag_correl": PM_IS_diag_correl_results, 
            "PM_IS_adapt_nocorrel": PM_IS_adapt_nocorrel_results, 
            "PM_IS_adapt_correl": PM_IS_adapt_correl_results}

#------------------------------------Part 3 - Saving the chains as csv------------------------------
#---------------------------------------------------------------------------------------------------

"""
Specify the parameters below, the function will output the chains in csv files.
The files are saved in the folders, corresponding to the specifics of models, which is created where the code is stored.
the files will be saved in the following format: 
PMMH_diag_nocorrel_T_700_101_{np.random.choice(10**5)}.csv

Choose the method to specify which variations to run:
method = "PMMH", "PM_IS", PMMH_adapt, "both"
"""




if __name__ == "__main__":
    main(
        method = "PMMH", \
        T_obs = 10, \
        run_seed = int(np.random.default_rng().integers(0, 2**32, \
        dtype=np.uint32)), \
        real_pars = {"mu": -0.86, "sigma2_eta": 0.0225, "phi": 0.97},\
        N_mcmc = 100,\
        burnin = 10,\
        x_0 = xstart(mu = -0.6, phi = 0.8, sigma2_eta = 0.1), \

        s_PMMH_diag_nocorrel = 0.1,\
        s_PMMH_diag_correl = 0.1, \
        s_PMMH_adapt_nocorrel = 1.88, \
        s_PMMH_adapt_correl = 1.88, \
            
        s_PM_IS_diag_nocorrel = 0.3, \
        s_PM_IS_diag_correl = 0.3, \
        s_PM_IS_adapt_nocorrel = 1,\
        s_PM_IS_adapt_correl = 1, \


        m_PMMH_diag_nocorrel = 50, \
        m_PMMH_diag_correl = 20, \
        m_PMMH_adapt_nocorrel = 50, \
        m_PMMH_adapt_correl = 20, \
        
        m_PM_IS_diag_nocorrel = 1000, \
        m_PM_IS_diag_correl = 350, \
        m_PM_IS_adapt_nocorrel = 1000, \
        m_PM_IS_adapt_correl = 350, \


        rho_PMMH_diag_correl = 0.99,\
        rho_PMMH_adapt_correl = 0.99, \
        rho_PM_IS_diag_correl = 0.99,\
        rho_PM_IS_adapt_correl = 0.99,\
        )

