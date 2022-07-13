import os
import glob

import numpy as np
import pandas as pd

import _visualize_results as visualize_results

code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

#%% Data processing functions
def create_long_table_entry(result, timescale, alg_relabeler):
    starting_times = {
        "Average" : result["num_random_timesteps"],
        "Final" : -1
    }
    t_start = starting_times[timescale]
    
    table_entry = {
        "Setting" : result["experiment_name"],
        "Algorithm" : alg_relabeler.get(result["alg_label"], result["alg_label"]),
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

def make_table_entry(results_list, algs_to_include=None, alg_relabeler={}):
    long_table_entries = [
        create_long_table_entry(result, "Average", alg_relabeler) 
        for result in results_sorted 
        if (algs_to_include is None) or (result["alg_label"] in algs_to_include)
    ]
    df_long = pd.DataFrame(long_table_entries)
    df_wide = df_long.pivot(index=["Setting","iid samples","Timesteps"],columns="Algorithm")
    return df_wide
    

if __name__ == "__main__":
    #%% Make table
    experiment_list = [
        'all_safe',
        'dosage_bandit_zero_correlation',
        'dosage_bandit_negative_correlation',
        'dosage_bandit_positive_correlation',
        'high_dim_contextual_5',
        'high_dim_contextual_10',
        'high_dim_contextual_15',
        'polynomial_bandit',
        'power_checker_5',
        'power_checker_10',
        'power_checker_15',
        'uniform_bandit',
    ]
    
    algs_to_include = [
        'Pretest all',
        'SPT',
        'SPT (fallback) (safe)',
    ]
    
    alg_relabeler = {"SPT (fallback) (safe)" : "SPT (fallback)"}
    
    df = pd.DataFrame()
    
    for experiment_name in experiment_list:
        filenames = glob.glob(os.path.join(data_path,f"{experiment_name}*.json"))
        print("Reading\n"+'\n'.join(filenames)+"...")
        results_dict = visualize_results.read_combine_and_process_json(filenames)
        results_sorted = [results_dict[key] for key in sorted(results_dict.keys())]
        
        df_wide = make_table_entry(results_sorted)
        df = df.append(df_wide)

filename = "tmp1.csv"
df.to_csv(os.path.join(data_path,filename))
s = df.style
print(s.to_latex(multicol_align="c",hrules=True))