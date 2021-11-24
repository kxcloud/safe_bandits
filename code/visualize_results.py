import json
import os

import matplotlib.pyplot as plt
import numpy as np

code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

def plot(results, axes):
    
    ax_reward, ax_safety, ax_safety_ind, ax_agreement = axes
    
    ax_reward.plot(results["mean_reward"].mean(axis=0))
    ax_reward.set_title("Mean rewards")
        
    ax_safety.plot(results["mean_safety"].mean(axis=0))
    ax_safety.set_title("Mean safety")
    
    ax_safety_ind.plot(results["safety_ind"].mean(axis=0))
    ax_safety_ind.set_title("Safety indicator")     
    ax_safety_ind.axhline(1-results["alpha"], ls="--", c="gray", lw=1)
    
    ax_agreement.plot(results["agreed_with_baseline"].mean(axis=0), label=results["alg_label"])
    ax_agreement.set_title("Agreed with baseline policy")
    
    ax_reward.set_xlabel("Timestep")

    # Label random timesteps
    for ax in axes:
        ax.axvline(
            x=results["num_random_timesteps"], 
            alpha=0.5, c="grey", lw=1, ymax=0.02
        )
    return axes

def plot_many(results_list):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(12,7))
    
    for results in results_list:
        plot(results, axes)
    
    # WARNING: this assumes best_safe_reward is the same for all runs
    axes[0].axhline(results["best_safe_reward"], ls=":", c="black", lw=1)         
        
    plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
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

plot_many(results1.values())
plot_many(results2.values())