import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _visualize_results as visualize_results

code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

def get_dfs_for_plot(results_list, moving_avg_window=None):
    result_keys = ["mean_reward", "safety_ind"]
    df_r = pd.DataFrame()
    df_s = pd.DataFrame()

    for result in results_list:
        alg_label = result["alg_label"]
        t_0 = result["num_random_timesteps"]    
        for result_key, df in zip(result_keys, [df_r, df_s]):
            data = result[result_key][:,t_0:]
            
            mean = data.mean(axis=(0,2))
            se = data.std(axis=(0,2)) / np.sqrt(result["num_runs"])
            if moving_avg_window:
                mean = pd.Series(mean).rolling(moving_avg_window, closed="both").mean()
                se = pd.Series(se).rolling(moving_avg_window, closed="both").mean()
            df[alg_label] = mean
            df[alg_label+"_se"] = se
    return df_r, df_s        

def plot_df(df, ax, algs_to_include, include_ci=False, alg_relabeler={}):
    for alg_idx, alg_label in enumerate(algs_to_include):
        data = df[alg_label]
        color = f"C{alg_idx}"
        if alg_label[-3:] != "_se":
            alg_label_legend = alg_relabeler.get(alg_label, alg_label)
            ax.plot(data, label=alg_label_legend, c=color, lw=2)
            if include_ci:
                ci_width = 1.96*df[alg_label+"_se"]
                ax.plot(data+ci_width, ls="--", c=color, alpha=0.5, lw=1)
                ax.plot(data-ci_width, ls="--", c=color, alpha=0.5, lw=1)
            
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
    # "Oracle",
]

alg_relabeler = {"SPT (fallback) (safe)" : "SPT (fallback)"}
setting_relabeler = {"Power checker" : "Single-arm detection"}

nrows=(len(experiment_list)+1)//2

fig, axes = plt.subplots(
    nrows=nrows, ncols=5, figsize=(10,14),
    gridspec_kw={"width_ratios":[1,1,1e-5,1,1]}
)
renderer = fig.canvas.get_renderer()

axes_flat = axes[:,[0,1,3,4]].flatten()

# REPLACE 36 WITH CORRECT LENGTH MANUALLY IF RESIZE
width_across_two_subplots = axes[0,0].set_title("0"*36).get_window_extent(renderer=renderer).width
sp_width = axes[0,0].set_title(" ").get_window_extent(renderer=renderer).width

for dummy_ax in axes[:,2]:
    dummy_ax.axis("off")
    
if len(experiment_list) % 2 == 1:
    axes[-1,-1].axis("off")

for experiment_idx, experiment in enumerate(experiment_list):
    filenames = glob.glob(os.path.join(data_path,f"hide\\{experiment}*.json"))
    print("Reading\n"+'\n'.join(filenames)+"...")
    results_dict = visualize_results.read_combine_and_process_json(filenames)
    results_sorted = [results_dict[key] for key in sorted(results_dict.keys())]
    df_r, df_s = get_dfs_for_plot(results_sorted, moving_avg_window=10)
    
    ax_r = axes_flat[2*experiment_idx]
    ax_s = axes_flat[2*experiment_idx+1]
    
    ax_s.axhline(0.9, ls="--", c="black", lw=1)
    
    ax_s.set_ylim([0.49,1.02])
    ax_s.set_yticks([0.5,0.9,1])
    
    plot_df(df_r, ax_r, algs_to_include, include_ci=True, alg_relabeler=alg_relabeler)
    plot_df(df_s, ax_s, algs_to_include, include_ci=False, alg_relabeler=alg_relabeler)
    
    experiment_name = results_sorted[0]["experiment_name"]
    for original, new in setting_relabeler.items():
        experiment_name = experiment_name.replace(original, new)
    
    title_width = ax_r.set_title(experiment_name, loc="left").get_window_extent(renderer=renderer).width
    padding = " "*round((width_across_two_subplots - title_width)/2/sp_width)
    ax_r.set_title(padding+experiment_name, loc="left")

# Set all xticks and axis labels
for row_idx in range(nrows):
    for col_idx in [0, 1, 4]:
        if not (row_idx == 0 and col_idx == 1):
            axes[row_idx, col_idx].label_outer()
    if row_idx != nrows-1:
        axes[row_idx,3].set_xticks([])
axes[0,1].set_xticks([])
axes_flat[0].set_ylabel("Reward", labelpad=-5)
axes_flat[1].set_ylabel("Safety", labelpad=-15)
axes[-1,0].set_xlabel("Timestep")

axes_flat[1].legend()

plt.tight_layout()