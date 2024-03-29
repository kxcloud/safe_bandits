import os
import glob

import numpy as np
import pandas as pd

import _visualize_results as visualize_results

code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

#%% Data processing functions
def create_long_table_entry(result, timescale, alg_relabeler, setting_relabeler):
    starting_times = {
        "Average" : result["num_random_timesteps"],
        "Final" : -1
    }
    t_start = starting_times[timescale]
    
    experiment_name = result["experiment_name"]
    for original, new in setting_relabeler.items():
        experiment_name = experiment_name.replace(original, new)
    
    table_entry = {
        "Setting" : experiment_name,
        "Algorithm" : alg_relabeler.get(result["alg_label"], result["alg_label"]),
        "Burn-in" : result["num_random_timesteps"],
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

def make_table_entry(results_list, timescale, algs_to_include=None, alg_relabeler={}, setting_relabeler={}):
    long_table_entries = [
        create_long_table_entry(
            result, timescale, alg_relabeler=alg_relabeler, setting_relabeler=setting_relabeler
        )
        for result in results_sorted 
        if (algs_to_include is None) or (result["alg_label"] in algs_to_include)
    ]
    df_long = pd.DataFrame(long_table_entries)
    df_wide = df_long.pivot(index=["Setting","Burn-in","Timesteps"],columns="Algorithm")
    return df_wide
    
def latex_bold(string):
    return "\\textbf{"+string+"}"

def latex_colorbox(string, color):
    return "\\colorbox{"+color+"}{"+string+"}"

def bold_entry(row, timescale, safety_tol=0.1):
    # Hacky way to bold the highest reward alg that satisfies safety constraint
    rewards = [float(entry.split(" ")[0]) for entry in row[timescale+" reward"]]
    safetys = [float(entry.split(" ")[0]) for entry in row[timescale+" safety"]]
    
    max_safe_reward = -np.inf
    max_safe_reward_idx = None
    for idx, (r, s) in enumerate(zip(rewards, safetys)):
        if r > max_safe_reward and s > 1-safety_tol - 1e-8:
            max_safe_reward = r
            max_safe_reward_idx = idx
            
    row[timescale+" reward"].iloc[max_safe_reward_idx] = latex_bold(
        f"{max_safe_reward:0.3f}"
    ) + " " + row[timescale+" reward"].iloc[max_safe_reward_idx].split(" ")[1]
    
def highlight_entry(row, timescale, safety_tol=0.1):
    # Hacky way to highlight entries that are safe
    safetys = [float(entry.split(" ")[0]) for entry in row[timescale+" safety"]]
    
    for idx, s in enumerate(safetys):
        if s > 1 - safety_tol - 1e-8:
            color = "tabgreen" # define this in paper
        else:
            color = "white" # want a color box to preserve alignment
        colored_value = latex_colorbox(f"{s:0.3f}", color)
        row[timescale+" safety"].iloc[idx] = (
            colored_value + " " + row[timescale+" safety"].iloc[idx].split(" ")[1]
        )

if __name__ == "__main__":    
    #%% Make table
    experiment_list_standard = [
        'all_safe',
        'dosage_bandit_zero_correlation',
        'dosage_bandit_negative_correlation',
        'dosage_bandit_positive_correlation',
        # 'polynomial_bandit',
        'power_checker_5',
        'power_checker_10',
        'power_checker_15',
        'uniform_bandit',
    ]   
    
    experiment_list_context = [
        'contextual_bandit_dot_0',
        'contextual_bandit_dot_minus_50',
        'contextual_bandit_dot_plus_50',
        'noisy_bandit_2_p5',
        'noisy_bandit_2_p10',
        'noisy_bandit_2_p15',
    ]
    
    experiment_list = experiment_list_standard + experiment_list_context
    
    algs_to_include = [
        'Pretest all',
        'SPT',
        'SPT (fallback) (safe)',
    ]
    
    alg_relabeler = {"SPT (fallback) (safe)" : "SPT (fallback)", "Pretest all" : "Pretest All"}
    
    setting_relabeler = {
        "Power checker" : "Single-arm detection", 
        "High-dim context": "Noisy context",
        "Noisy bandit v2, (d_noise=5)" : "Noisy bandit (d=5)",
        "Noisy bandit v2, (d_noise=10)" : "Noisy bandit (d=10)",
        "Noisy bandit v2, (d_noise=15)" : "Noisy bandit (d=15)",
        "Reward-safety corr": "dot",
        "d=1, " : "",
        "Contextual bandit" : "Orthogonal actions"
    }
    
    timescales = ["Average", "Final"]
    dfs = [pd.DataFrame(), pd.DataFrame()]
    
    for experiment_name in experiment_list:
        filenames = glob.glob(os.path.join(data_path,f"{experiment_name}*.json"))
        print("Reading\n"+'\n'.join(filenames)+"...")
        results_dict = visualize_results.read_combine_and_process_json(filenames)
        results_sorted = [results_dict[key] for key in sorted(results_dict.keys())]
        
        for idx, timescale in enumerate(timescales):
            df_wide = make_table_entry(
                results_sorted, timescale=timescale, algs_to_include=algs_to_include, 
                alg_relabeler=alg_relabeler, setting_relabeler=setting_relabeler
            )
            dfs[idx] = dfs[idx].append(df_wide)


    for df, timescale in zip(dfs, timescales):
        for _, row in df.iterrows():
            bold_entry(row, timescale=timescale)
            highlight_entry(row, timescale=timescale)
        print(df.style.to_latex(multicol_align="c",hrules=True))