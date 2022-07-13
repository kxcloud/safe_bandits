import glob
import os 
import subprocess
import importlib

import _visualize_results as visualize_results
import ProgressTracker

CONDA_ENVIRONMENT_NAME = "base"
code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

# CHANGE SETTINGS HERE
import experiments.algorithms_2022_07_12 as algorithm_settings
experiment_list = [
    'all_safe',
    'dosage_bandit',
    'dosage_bandit_negative_correlation',
    'dosage_bandit_positive_correlation',
    'high_dim_contextual',
    'polynomial_bandit',
    'power_checker_10',
    'power_checker_15',
    'power_checker_5',
    'uniform_bandit',
]
num_processes = 2
num_runs = 2

#%% Run experiments
for experiment_name in experiment_list:
    print(f"Running experiment {experiment_name}...")
    experiment_module = "experiments."+experiment_name
    experiment_settings = importlib.import_module(experiment_module)
    if num_processes is None:
        import experiment_worker
        data_filename = os.path.join(data_path, f"{experiment_name}.json")
        argv = [
            None, 
            experiment_settings.__name__, 
            algorithm_settings.__name__, 
            data_filename, num_runs, 
            None, 
            0
        ]
        experiment_worker.main(argv)
    else:
        progress_dir = os.path.join(data_path,f"{experiment_name}_TMP")
        num_algs = len(algorithm_settings.get_alg_dict(None,None))
        pbar = ProgressTracker.ProgressTracker(
            total_steps = num_processes*num_algs,
            progress_dir = progress_dir,
        )
        worker_filename = os.path.join(code_path, "experiment_worker.py")
        activate = f"conda activate {CONDA_ENVIRONMENT_NAME}"
        subprocesses = []
        for process_idx in range(num_processes):
            data_filename = os.path.join(data_path,f"{experiment_name}_{process_idx}.json")
            argv = [
                worker_filename, 
                experiment_settings.__name__, 
                algorithm_settings.__name__, 
                data_filename, 
                str(num_runs),
                progress_dir,
                str(process_idx)
            ]
            run = " ".join(["python", *argv])
            command = " & ".join([activate, run])
            subp = subprocess.Popen(command, shell=True)
            subprocesses.append(subp)
        pbar.monitor(subprocesses)

#%% Plot
filenames = glob.glob(os.path.join(data_path,f"{experiment_name}*.json"))
print("Reading\n"+'\n'.join(filenames)+"...")
results_dict = visualize_results.read_combine_and_process_json(filenames)

results_sorted = [results_dict[key] for key in sorted(results_dict.keys())]

title = results_sorted[0]["experiment_name"]

visualize_results.plot_many(
    results_sorted, 
    plot_confidence=True,
    plot_baseline_rewards=False, 
    plot_random_timesteps=False,
    include_mean_safety=False,
    moving_avg_window=20, 
    title=title,
    figsize=(13,5),
    colors=None
)