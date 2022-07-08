from functools import partial

import numpy as np

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _visualize_results as visualize_results
import _utils as utils

import run_settings.settings_2022_07_08 as run_settings

# bandit_constructor = partial(BanditEnv.get_dosage_example, num_actions=20, param_count=10)
# bandit_constructor = partial(BanditEnv.get_power_checker, num_actions=20, effect_size=0.5)

bandit_constructor = partial(
    BanditEnv.get_uniform_armed_bandit,
    means=[1, 1.5, 2], 
    prob_negative=[0, 0.15, 0.4]
)

num_runs = 500

#%% Run
results_dict = {}
for alg_label, learning_algorithm in run_settings.alg_dict.items():
    results = bandit_learning.evaluate(
        alg_label,
        bandit_constructor,
        learning_algorithm,
        baseline_policy = run_settings.baseline_policy,
        num_random_timesteps=5,
        num_alg_timesteps=395,
        num_runs=num_runs,
        num_instances=1,
        alpha=0.1,
        safety_tol=run_settings.safety_tol
    )
    results_dict[alg_label] = results

total_duration = sum([results["duration"] for results in results_dict.values()])
print(f"Total duration: {total_duration:0.02f} minutes.")
utils.print_run_counts_by_time(num_runs, total_duration)

bandit_learning.save_to_json(results_dict, "2022_07_08_uniform_armed_2_C.json")

# %% Plot

filenames = [
    "2022_07_08_uniform_armed_2_A.json",
    "2022_07_08_uniform_armed_2_B.json",
    "2022_07_08_uniform_armed_2_C.json",
]
results_dict = visualize_results.read_combine_and_process_json(filenames)

colors = None #["C1", "C3", "C2"]

title = "Uniform armed bandit"

visualize_results.plot_many(
    results_dict.values(), 
    plot_confidence=True,
    plot_baseline_rewards=True, 
    plot_random_timesteps=False,
    include_mean_safety=False,
    moving_avg_window=5, 
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

