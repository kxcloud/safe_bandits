from functools import partial

import numpy as np

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _visualize_results as visualize_results
import _utils as utils

import run_settings.settings_2022_07_07 as run_settings

bandit_constructor = partial(BanditEnv.get_dosage_example, num_actions=20, param_count=10)

num_runs = 500

#%% Run
results_dict = {}
for alg_label, learning_algorithm in run_settings.alg_dict.items():
    results = bandit_learning.evaluate(
        alg_label,
        bandit_constructor,
        learning_algorithm,
        baseline_policy = run_settings.baseline_policy,
        num_random_timesteps=100,
        num_alg_timesteps=200,
        num_runs=num_runs,
        num_instances=1,
        alpha=0.1,
        safety_tol=run_settings.safety_tol
    )
    results_dict[alg_label] = results

total_duration = sum([results["duration"] for results in results_dict.values()])
print(f"Total duration: {total_duration:0.02f} minutes.")
utils.print_run_counts_by_time(num_runs, total_duration)

bandit_learning.save_to_json(results_dict, "2022_07_07_fallback_C.json")

# %% Plot

filename0 = "2022_07_07_fallback_A.json"
filename1 = "2022_07_07_fallback_B.json"
filename2 = "2022_07_07_fallback_C.json"
filenames = [filename0, filename1, filename2]
results_dict = visualize_results.read_combine_and_process_json(filenames)

colors = None #["C1", "C3", "C2"]

title = "Dosage example"

visualize_results.plot_many(
    results_dict.values(), 
    plot_confidence=True,
    plot_baseline_rewards=False, 
    plot_random_timesteps=False,
    include_mean_safety=False,
    moving_avg_window=20, 
    title=title,
    figsize=(13,5),
    colors=colors
)

visualize_results.plot_action_dist(
    results_dict.values(), 
    num_to_plot=5, 
    drop_first_action=False, 
    figsize=(14,4), 
    title=title
)

