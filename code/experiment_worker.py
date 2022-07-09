import sys
import importlib

import _bandit_learning as bandit_learning
import _utils as utils

def main(argv):
    settings_filename = argv[1]
    data_filename = argv[2]
    experiment_settings = importlib.import_module(settings_filename)
    
    results_dict = {}
    for alg_label, learning_algorithm in experiment_settings.alg_dict.items():
        results = bandit_learning.evaluate(
            alg_label,
            experiment_settings.bandit_constructor,
            learning_algorithm,
            baseline_policy = experiment_settings.baseline_policy,
            num_random_timesteps=5,
            num_alg_timesteps=395,
            num_runs=experiment_settings.num_runs,
            num_instances=1,
            alpha=0.1,
            safety_tol=experiment_settings.safety_tol
        )
        results_dict[alg_label] = results
    
    total_duration = sum([results["duration"] for results in results_dict.values()])
    print(f"Total duration: {total_duration:0.02f} minutes.")
    utils.print_run_counts_by_time(experiment_settings.num_runs, total_duration)
    
    bandit_learning.save_to_json(results_dict, data_filename)

if __name__ == "__main__":
    main(sys.argv)