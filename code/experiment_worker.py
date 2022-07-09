import sys
import importlib

import numpy as np

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _visualize_results as visualize_results
import _utils as utils


if __name__ == "__main__":
    settings_filename = sys.argv[2]
    process_idx = sys.argv[3]
    run_settings = importlib.import_module(settings_filename)
    
    #%% Run
    results_dict = {}
    for alg_label, learning_algorithm in run_settings.alg_dict.items():
        results = bandit_learning.evaluate(
            alg_label,
            run_settings.bandit_constructor,
            learning_algorithm,
            baseline_policy = run_settings.baseline_policy,
            num_random_timesteps=5,
            num_alg_timesteps=395,
            num_runs=run_settings.num_runs,
            num_instances=1,
            alpha=0.1,
            safety_tol=run_settings.safety_tol
        )
        results_dict[alg_label] = results
    
    total_duration = sum([results["duration"] for results in results_dict.values()])
    print(f"Total duration: {total_duration:0.02f} minutes.")
    utils.print_run_counts_by_time(run_settings.num_runs, total_duration)
    
    bandit_learning.save_to_json(results_dict, f"{settings_filename}_{process_idx}.json")

