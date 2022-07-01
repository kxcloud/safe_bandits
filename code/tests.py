from functools import partial

import numpy as np

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _visualize_results as visualize_results
import _utils as utils
from alg_settings.settings_2022_06_29 import alg_dict

wrapped_partial = utils.wrapped_partial
baseline_policy = lambda x: 0

bandit_constructor = partial(BanditEnv.get_dosage_example, num_actions=20, param_count=10)

EPSILON = 0.1
safety_tol = 0.1
alpha=0.1



results_dict = {}
for alg_label, action_selection in alg_dict.items():
    results = bandit_learning.evaluate(
        alg_label,
        bandit_constructor,
        action_selection,
        baseline_policy = bandit_learning.baseline_policy,
        num_random_timesteps=1000,
        num_alg_timesteps=10,
        num_instances=10,
        num_runs=30,
        alpha=0.1,  
        safety_tol=safety_tol,
    )
    run_label = f"{alg_label}"
    results_dict[run_label] = results

#%%
visualize_results.plot_many(
    results_dict.values(), 
    plot_confidence=True,
    plot_baseline_rewards=True, 
    plot_random_timesteps=False,
    include_mean_safety=False,
    moving_avg_window=None, 
    figsize=(13,5),
)
