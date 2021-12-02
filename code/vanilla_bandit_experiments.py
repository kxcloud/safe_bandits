from functools import partial

import numpy as np

import BanditEnv
import bandit_learning
import visualize_results

wrapped_partial = bandit_learning.wrapped_partial
baseline_policy = lambda x: 0

bandit_constructor = partial(BanditEnv.get_random_action_bandit, num_actions=30, outcome_correlation=0)

#%% Run
results_dict = {}
for alg_label, learning_algorithm in bandit_learning.alg_dict.items():
    results = bandit_learning.evaluate(
        alg_label,
        bandit_constructor,
        learning_algorithm,
        baseline_policy = baseline_policy,
        num_random_timesteps=10,
        num_alg_timesteps=70,
        num_runs=10_000,
        alpha=0.1,    
    )
    results_dict[alg_label] = results

total_duration = sum([results["duration"] for results in results_dict.values()])
print(f"Total duration: {total_duration:0.02f} minutes.")

bandit_learning.save_to_json(results_dict, "2021_12_01_vanilla_bandit_B.json")

#%% Plot

filename0 = "2021_12_01_vanilla_bandit.json"
filename1 = "2021_12_01_vanilla_bandit_B.json"
filenames = [filename0, filename1]
results_dict = visualize_results.read_combine_and_process_json(filenames)

subset = [
    # 'Unsafe e-greedy',
    # 'Unsafe TS',
    # 'FWER pretest (all): e-greedy',
    'FWER pretest (all): TS',
    'Propose-test TS',
    'Propose-test TS (OOS covariance)',
    'Propose-test TS (random split)',
    'Propose-test TS (random) (OOS)',
    # 'Propose-test TS (safe FWER fallback [all])'
]
    
title = f"Random vanilla bandit"
# bandit_constructor().plot(title=title)
visualize_results.plot_many(
    [results_dict[alg_label] for alg_label in subset], 
    plot_confidence=True,
    plot_baseline_rewards=False, 
    plot_random_timesteps=False,
    moving_avg_window=10, 
    title=title,
    figsize=(13,5)
)
