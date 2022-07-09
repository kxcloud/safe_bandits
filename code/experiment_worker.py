import sys
import importlib

import _bandit_learning as bandit_learning
import _utils as utils

def main(argv):
    settings_filename = argv[1]
    data_filename = argv[2]
    num_runs = argv[3]
    experiment_settings = importlib.import_module(settings_filename)
    
    results_dict = {}
    for alg_label, learning_algorithm in experiment_settings.alg_dict.items():
        results = experiment_settings.evaluator(
            alg_label=alg_label,
            learning_algorithm=learning_algorithm,
            num_runs=num_runs,
        )
        results_dict[alg_label] = results
    
    total_duration = sum([results["duration"] for results in results_dict.values()])
    print(f"Total duration: {total_duration:0.02f} minutes.")
    utils.print_run_counts_by_time(num_runs, total_duration)

    bandit_learning.save_to_json(results_dict, data_filename)

if __name__ == "__main__":
    main(sys.argv)