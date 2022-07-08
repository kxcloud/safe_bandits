import json
import os

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('tableau-colorblind10')
import numpy as np
import pandas as pd

code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

def tabulate_final_timestep_results(
        results_list,
        include_random_timesteps,
        include_mean_safety,
        moving_avg_window=None
    ):
    if include_mean_safety:
        data_keys = ["mean_reward", "mean_safety", "safety_ind", "agreed_with_baseline"]
    else:
        data_keys = ["mean_reward", "safety_ind", "agreed_with_baseline"]
            
    records = []
    for results in results_list:
        record = [results["alg_label"]]
        for data_key in data_keys:
            data = results[data_key][:,-1]
             
            # Average over runs (axis 0) and instances/patients (axis 2)
            mean = data.mean()
             
            if moving_avg_window:
               mean = pd.Series(mean).rolling(moving_avg_window).mean()
             
            num_runs = results[data_key].shape[0]
            std = data.std()
            ci_width = std * 1.96 / np.sqrt(num_runs)
         
            if moving_avg_window:
                ci_width = pd.Series(ci_width).rolling(moving_avg_window).mean()
            
            record = record + [mean, ci_width]
        records.append(record)
        
    column_names = ["alg_label"]
    for data_key in data_keys:
        column_names += [data_key, data_key+"_moe"]
    df = pd.DataFrame(records, columns=column_names)
    return df

def plot(
        results, 
        axes, 
        plot_confidence=True, 
        moving_avg_window=None, 
        color=None,
        plot_random_timesteps=True,
        include_mean_safety=True,
    ):
    
    if include_mean_safety:
        ax_reward, ax_safety, ax_safety_ind, ax_agreement = axes
        result_keys = ["mean_reward", "mean_safety", "safety_ind", "agreed_with_baseline"]
        titles = ["Mean rewards", "Mean safety", "Safety indicator", "Agreed with baseline policy"]
        
    if not include_mean_safety:
        ax_reward, ax_safety_ind, ax_agreement = axes
        result_keys = ["mean_reward", "safety_ind", "agreed_with_baseline"]
        titles = ["Mean rewards", "Safety indicator", "Agreed with baseline policy"]
    
    t_0 = 0 if plot_random_timesteps else results["num_random_timesteps"]
    t_final = results["num_random_timesteps"] + results["num_alg_timesteps"]
    timesteps = range(t_0, t_final)
    
    for ax, result_key, title in zip(axes, result_keys, titles):
        data = results[result_key][:,t_0:]
        
        # Average over runs (axis 0) and instances/patients (axis 2)
        mean = data.mean(axis=(0,2))
        
        if moving_avg_window:
            mean = pd.Series(mean).rolling(moving_avg_window).mean()
            
        label = results["alg_label"] if ax is ax_agreement else None
        lines = ax.plot(timesteps, mean, label=label, c=color, lw=2.5)
        
        if plot_confidence:
            num_runs = results[result_key].shape[0]
            std = data.std(axis=(0,2))
            ci_width = std * 1.96 / np.sqrt(num_runs)
            
            if moving_avg_window:
                ci_width = pd.Series(ci_width).rolling(moving_avg_window).mean()
                        
            ax.plot(timesteps, mean+ci_width, c=lines[0].get_color(), lw=1, ls="--")
            ax.plot(timesteps, mean-ci_width, c=lines[0].get_color(), lw=1, ls="--")
                
        ax.set_title(title)
   
    ax_reward.set_xlabel("Timestep")
    ax_safety_ind.axhline(1-results["alpha"], ls="--", c="gray", lw=1)  

    # Add vertical tick where random timesteps end
    if plot_random_timesteps:
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
        include_mean_safety=True,
        figsize=(11,3.75), 
        title=""
    ):
    
    ncols = 4 if include_mean_safety else 3
    fig, axes = plt.subplots(nrows=1, ncols=ncols, sharex=True, figsize=figsize)
    
    if colors is None:
        colors = [None]*len(results_list)
        
    for results, color in zip(results_list, colors):
        plot(
            results, 
            axes=axes, 
            plot_confidence=plot_confidence,
            moving_avg_window=moving_avg_window,
            color=color, 
            plot_random_timesteps=plot_random_timesteps,
            include_mean_safety=include_mean_safety
        )
    
    axes[-1].legend()
    
    # WARNING: this assumes best_safe_reward is the same for all runs
    if plot_baseline_rewards:
        axes[0].axhline(results["best_safe_reward"], ls=":", c="black", lw=1)      
        axes[0].axhline(results["baseline_reward"], ls=":", c="black", lw=1)
        
    plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    plt.suptitle(title)
    plt.show()

def plot_action_dist(results_list, num_to_plot, drop_first_action, figsize=(6,4), title=None):
    assert drop_first_action in [0,1]
    fig, axes = plt.subplots(
        ncols=len(results_list), sharex=True, sharey=True, figsize=figsize
    )
    axes[0].set_xlabel("Action")
    fig.suptitle(title)
    
    action_freqs = [
        np.mean(result["action_inds"][:,:,:,drop_first_action:], axis=(0,2))
        for result in results_list
    ]
    max_prob = np.max([np.max(action_freq) for action_freq in action_freqs])
    
    for ax, action_freq, result in zip(axes, action_freqs, results_list):
        ax.set_yticks([])
        ax.set_title(result["alg_label"])
        action_space = result["action_space"][drop_first_action:]
        annotation_x = (action_space[0] + action_space[-1])*0.8
        
        num_timesteps = action_freq.shape[0]
        
        timesteps_to_plot = np.linspace(0, num_timesteps-1, num_to_plot).astype(int)
        alphas = np.linspace(0.4, 1, num_to_plot)
        for t_idx, t in enumerate(timesteps_to_plot):
            height = (num_to_plot-t_idx)*max_prob
            
            ax.plot(action_space, height+action_freq[t,:], c="C0", alpha=alphas[t_idx])
            ax.plot(action_space, np.full(len(action_space), height), lw=1, ls=":", c="black", alpha=0.5)
            
            if t_idx in [0, 1, num_to_plot-1]:
                ax.annotate(f"t={t}", (annotation_x, height+max_prob*0.1))
    return axes 

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
                    if type(data) is str or setting == "action_space":
                        assert data == new_data, (
                            f"Run setting mismatch during merge (setting={setting}).\n"
                            f"data={data}, new_data={new_data}"
                        )
                    elif type(data) is list:
                        data.extend(new_data)

                    elif setting == "duration":
                        run_data[setting] += new_data

    for _, result in combined_results_dict.items():
        for key, item in result.items():
            if type(item) == list:
                result[key] = np.array(item)
    
    return combined_results_dict         
