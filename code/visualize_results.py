import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

def plot(
        results, 
        axes, 
        plot_confidence=True, 
        moving_avg_window=None, 
        color=None,
        plot_random_timesteps=True
    ):
    ax_reward, ax_safety, ax_safety_ind, ax_agreement = axes
    result_keys = ["mean_reward", "mean_safety", "safety_ind", "agreed_with_baseline"]
    titles = ["Mean rewards", "Mean safety", "Safety indicator", "Agreed with baseline policy"]
    
    t_0 = 0 if plot_random_timesteps else results["num_random_timesteps"]
    t_final = results["num_random_timesteps"] + results["num_alg_timesteps"]
    timesteps = range(t_0, t_final)
    
    for ax, result_key, title in zip(axes, result_keys, titles):
        data = results[result_key][:,t_0:]
        mean = data.mean(axis=0)
      
        if moving_avg_window:
            mean = pd.Series(mean).rolling(moving_avg_window).mean()
            
        label = results["alg_label"] if ax is ax_agreement else None
        lines = ax.plot(timesteps, mean, label=label, c=color, lw=2)
        
        if plot_confidence:
            num_runs = results[result_key].shape[0]
            std = data.std(axis=0)
            ci_width = std * 1.96 / np.sqrt(num_runs)
            
            if moving_avg_window:
                ci_width = pd.Series(ci_width).rolling(moving_avg_window).mean()
                        
            ax.plot(timesteps, mean+ci_width, c=lines[0].get_color(), lw=1, ls="--")
            ax.plot(timesteps, mean-ci_width, c=lines[0].get_color(), lw=1, ls="--")
                
        ax.set_title(title)
   
    ax_reward.set_xlabel("Timestep")
    ax_safety_ind.axhline(1-results["alpha"], ls="--", c="gray", lw=1)  

    # Add vertical tick where random timesteps end
    for ax in axes:
        ax.axvline(
            x = results["num_random_timesteps"],
            alpha=0.5, 
            c="grey", 
            lw=1, 
            ymax=0.02
        )
    return axes

def plot_many(
        results_list, 
        plot_baseline_rewards=True, 
        plot_confidence=True,
        colors=None, 
        moving_avg_window=None, 
        plot_random_timesteps=True,
        figsize=(11,3.75), 
        title=""
    ):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=figsize)
    
    if colors is None:
        colors = [None]*len(results_list)
        
    for results, color in zip(results_list, colors):
        
        plot(
            results, 
            axes=axes, 
            plot_confidence=plot_confidence,
            moving_avg_window=moving_avg_window,
            color=color, 
            plot_random_timesteps=plot_random_timesteps
        )
    
    axes[-1].legend()
    
    # WARNING: this assumes best_safe_reward is the same for all runs
    if plot_baseline_rewards:
        axes[0].axhline(results["best_safe_reward"], ls=":", c="black", lw=1)      
        axes[0].axhline(results["baseline_reward"], ls=":", c="black", lw=1)
        
    plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    plt.suptitle(title)
    plt.show()

def read_and_process_json(filename):
    with open(os.path.join(data_path,filename), 'r') as f:
        results_dict = json.load(f)
    
    for _, result in results_dict.items():
        for key, item in result.items():
            if type(item) == list:
                result[key] = np.array(item)
    
    return results_dict

def read_combine_and_process_json(filenames):
    """ 
    Combine multiple versions of the same runs.
    
    Assumes: each file is a dict keyed by the run name, with each run being a 
    dict with values that are either single values or lists. Lists will
    be appended.
    """
    for idx, filename in enumerate(filenames):
        with open(os.path.join(data_path,filename), 'r') as f:
            results_dict = json.load(f)
            
        if idx == 0:
            combined_results_dict = results_dict
        else:
            for run_label, run_data in combined_results_dict.items():
                assert run_label in results_dict, f"Run {run_label} missing from dict during merge."
                
                for setting, data in run_data.items():
                    new_data = results_dict[run_label][setting]
                    if type(data) is list:
                        data.extend(new_data)
                    elif type(data) is str:
                        assert data == new_data, (
                            f"Run setting mismatch during merge (setting={setting})."
                        )
                    elif setting == "duration":
                        run_data[setting] += new_data

    for _, result in combined_results_dict.items():
        for key, item in result.items():
            if type(item) == list:
                result[key] = np.array(item)
    
    return combined_results_dict         

if __name__ == "__main__":                
    #%% Plot data
    filename1 = "2021_11_30_out_of_sample_comparison.json"
    filename2 = "2021_11_30_out_of_sample_comparison_B.json"
    results1 = read_combine_and_process_json([filename1, filename2])
    # results1 = read_and_process_json(filename1)
    # results2 = read_and_process_json(filename2)
    
    #%%
    runs = [
        'Unsafe e-greedy',
        'Unsafe TS',
        'FWER pretest (all): e-greedy',
        'FWER pretest (all): TS',
        'Propose-test TS',
        'Propose-test TS (OOS covariance)',
        'Propose-test TS (random split)',
        'Propose-test TS (random) (OOS)',
        'Propose-test TS (safe FWER fallback [all])'
    ]
    
    colors = {run_name: f"C{idx}" for idx, run_name in enumerate(runs)}
    
    subset = [
        # 'Unsafe e-greedy',
        # 'Unsafe TS',
        'FWER pretest (all): e-greedy',
        'FWER pretest (all): TS',
        'Propose-test TS',
        'Propose-test TS (OOS covariance)',
        'Propose-test TS (random split)',
        'Propose-test TS (random) (OOS)',
        'Propose-test TS (safe FWER fallback [all])'
    ]
    
    colors = [colors[key] for key in subset]
    title = ""
    
    for result_list in [results1]: #, results2]:
        plot_many(
            [result_list[key] for key in subset], 
            moving_avg_window=20, 
            colors=colors, 
            title=title+" Sinusoidal bandit",
            figsize=(13,7),
            plot_confidence=True,
            plot_baseline_rewards=False
        )
