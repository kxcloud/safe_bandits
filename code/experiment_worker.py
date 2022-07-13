import os
import sys
import importlib

import _bandit_learning as bandit_learning
import _utils as utils

def main(argv):
    settings_filename = argv[1]
    algorithm_filename = argv[2]
    data_filename = argv[3]
    num_runs = int(argv[4])
    progress_dir = argv[5]
    process_idx = int(argv[6])
    experiment_settings = importlib.import_module(settings_filename)
    algorithm_settings = importlib.import_module(algorithm_filename)
    
    alg_dict = algorithm_settings.get_alg_dict(
        baseline_policy = experiment_settings.baseline_policy,
        safety_tol = experiment_settings.safety_tol
    )
    
    # Sloppy hack to get better runtime estimates across multiple processes
    alg_dict_items = alg_dict.items()
    if progress_dir is not None and process_idx % 2 == 1:
        alg_dict_items = reversed(alg_dict_items)
    
    results_dict = {}
    for alg_label, learning_algorithm in alg_dict_items:
        results = experiment_settings.evaluator(
            alg_label=alg_label,
            learning_algorithm=learning_algorithm,
            num_runs=num_runs,
        )
        results_dict[alg_label] = results
        if progress_dir is not None:
            tmp_filename = os.path.join(progress_dir,f"{alg_label}_{process_idx}")
            with open(tmp_filename, "w") as f:
                f.write("")
    
    total_duration = sum([results["duration"] for results in results_dict.values()])
    print(f"Total duration: {total_duration:0.02f} minutes.")
    utils.print_run_counts_by_time(num_runs, total_duration)

    bandit_learning.save_to_json(results_dict, data_filename)

if __name__ == "__main__":
    main(sys.argv)