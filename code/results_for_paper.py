import os
import glob

import numpy as np
import pandas as pd

import _visualize_results as visualize_results

code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

data_file_prefix = "experiments.high_dim_contextual"
filenames = glob.glob(os.path.join(data_path,f"{data_file_prefix}*.json"))
print("Reading\n"+'\n'.join(filenames)+"...")
results_dict = visualize_results.read_combine_and_process_json(filenames)

results_sorted = [results_dict[key] for key in sorted(results_dict.keys())]

algs = [result["alg_label"] for result in results_sorted]
print(algs)

title = "High dim contextual bandit (p=5, 5 actions)"

visualize_results.plot_many(
    results_sorted, 
    plot_confidence=True,
    plot_baseline_rewards=False, 
    plot_random_timesteps=False,
    include_mean_safety=False,
    moving_avg_window=20, 
    title=title,
    figsize=(13,5),
    colors=None
)

#%%
result = results_sorted[0]

starting_times = {
    "Average" : result["num_random_timesteps"],
    "Final" : -1
}

algs_included = algs[:3]

def create_long_table_entry(result, timescale):
    assert timescale in ["Average", "Final"]
    t_start = starting_times[timescale]
    
    table_entry = {
        "Setting" : result["experiment_name"],
        "Algorithm" : result["alg_label"],
        "iid samples" : result["num_random_timesteps"],
        "Timesteps" : result["num_alg_timesteps"]
    }
    
    

    n_runs_sqrt = np.sqrt(result["num_runs"])

    outcomes = {
        "reward" : "mean_reward",
        "safety" : "safety_ind"
    }
    
    for outcome_label, outcome_key in outcomes.items():
        # Note: indexing trick to preserve timestep dimension
        data_per_run = result[outcome_key][:,t_start:None,:].mean(axis=(1,2)) 
        mean_outcome = data_per_run.mean() # Average over runs
        standard_error = data_per_run.std() / n_runs_sqrt
        column_label = timescale+" "+outcome_label
        table_entry[column_label] = f"{mean_outcome:0.03f} ({standard_error:0.03f})"
    return table_entry

long_table_entries = [
    create_long_table_entry(result, "Average") 
    for result in results_sorted if result["alg_label"] in algs_included
]
df_long = pd.DataFrame(long_table_entries)
df = df_long.pivot(index=["Setting","iid samples","Timesteps"],columns="Algorithm")

s = df.style
print(s.to_latex(multicol_align="c",hrules=True))