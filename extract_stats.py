import numpy as np
import pandas as pd
from pathlib import Path

folder = Path(r"C:\Users\haykg\Documents\THEdissertation\T_200_PMMH")



dfs = []                                                                        # a lits to store all dataframes
for fp in folder.glob("*.csv"):
    df = pd.read_csv(fp)
    dfs.append(df)


#----------------------------------------------gettign the average across the dfs---------------------------------------

# 1) Name the value columns (exclude the first)
value_cols = dfs[0].columns[1:]

# 2) Concatenate the value parts along rows with a key per DataFrame
stacked = pd.concat([df[value_cols] for df in dfs],
                    axis=0, keys=range(len(dfs)), names=["src"])

# 3) Average across the 'src' level (i.e., across DataFrames) for each row index
avg_values = stacked.groupby(level=1).mean()

# 4) Reattach the first column (aligned by index)
result = dfs[0].iloc[:, [0]].join(avg_values)

result.to_csv("full_T200.csv", index = False)