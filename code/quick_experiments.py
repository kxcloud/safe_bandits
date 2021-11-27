import numpy as np

import BanditEnv
import bandit_learning
import visualize_results

alg_dict = bandit_learning.alg_dict

results_dict = {}
for alg_label, action_selection in alg_dict.items():
    results = bandit_learning.evaluate(
        alg_label,
        BanditEnv.get_sinusoidal_bandit,
        action_selection,
        baseline_policy = bandit_learning.baseline_policy,
        num_random_timesteps=10,
        num_alg_timesteps=60,
        num_runs=50,
        alpha=0.1,    
    )
    results_dict[alg_label] = results
    
    total_duration = sum([results["duration"] for results in results_dict.values()])
    print(f"Total duration: {total_duration:0.02f} minutes.")