import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

def plot(results, axes, moving_avg_window=None, color=None):
    ax_reward, ax_safety, ax_safety_ind, ax_agreement = axes
    result_keys = ["mean_reward", "mean_safety", "safety_ind", "agreed_with_baseline"]
    titles = ["Mean rewards", "Mean safety", "Safety indicator", "Agreed with baseline policy"]
    
    for ax, result_key, title in zip(axes, result_keys, titles):
        data = results[result_key].mean(axis=0)
        if moving_avg_window:
            data = pd.Series(data).rolling(moving_avg_window).mean()
        
        label = results["alg_label"] if ax is ax_agreement else None
        ax.plot(data, label=label, c=color, lw=1.75)
        ax.set_title(title)
   
    ax_reward.set_xlabel("Timestep")
    ax_safety_ind.axhline(1-results["alpha"], ls="--", c="gray", lw=1)  

    # Add vertical tick where random timesteps end
    for ax in axes:
        ax.axvline(
            x=results["num_random_timesteps"], 
            alpha=0.5, c="grey", lw=1, ymax=0.02
        )
    return axes

def plot_many(results_list, colors, moving_avg_window=None, title=""):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(11,3.75))
    
    for results, color in zip(results_list, colors):
        plot(results, axes, moving_avg_window, color)
    
    axes[-1].legend()
    
    # WARNING: this assumes best_safe_reward is the same for all runs
    axes[0].axhline(results["best_safe_reward"], ls=":", c="black", lw=1)         
        
    plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    plt.suptitle(title)
    plt.show()

def read_and_process_json(filename):
    with open(os.path.join(data_path,filename), 'r') as f:
        results_dict= json.load(f)
    
    for _, result in results_dict.items():
        for key, item in result.items():
            if type(item) == list:
                result[key] = np.array(item)
    
    return results_dict
                
#%% Plot data
filename1 = "2021_11_24_sinusoidal_bandit.json"
filename2 = "2021_11_24_polynomial_bandit.json"
results1 = read_and_process_json(filename1)
results2 = read_and_process_json(filename2)

#%%
runs = [
 'Unsafe e-greedy',
 'Unsafe TS',
 'FWER pretest: e-greedy',
 'FWER pretest: TS',
 'Propose-test TS',
 'Propose-test TS (random split)',
 'Propose-test TS (unsafe FWER fallback)',
 'Propose-test TS (safe FWER fallback)',
 'Full-sample proposal objective'
]

colors = {run_name: f"C{idx}" for idx, run_name in enumerate(runs)}

subset = [
  # 'Unsafe e-greedy',
  # 'Unsafe TS',
  'FWER pretest: e-greedy',
  'FWER pretest: TS',
  'Propose-test TS',
  'Propose-test TS (random split)',
   # 'Propose-test TS (unsafe FWER fallback)',
  'Propose-test TS (safe FWER fallback)',
   # 'Full-sample proposal objective'
]

colors = [colors[key] for key in subset]
title = ""
plot_many([results1[key] for key in subset], moving_avg_window=20, colors=colors, title=title+" Sinusoidal bandit")
plot_many([results2[key] for key in subset], moving_avg_window=20, colors=colors, title=title+" Polynomial bandit")