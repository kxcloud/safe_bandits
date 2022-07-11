import glob
import os 
import subprocess

import _visualize_results as visualize_results
import ProgressTracker

CONDA_ENVIRONMENT_NAME = "base"
code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

#%% CHANGE SETTINGS HERE
import experiments.dosage_bandit_positive_correlation as experiment_settings
num_processes = None
num_runs = 2
data_file_prefix = experiment_settings.__name__

#%% Run experiments
if num_processes is None:
    import experiment_worker
    data_filename = os.path.join(data_path, f"{data_file_prefix}.json")
    argv = [None, experiment_settings.__name__, data_filename, num_runs, None, 0]
    experiment_worker.main(argv)
else:
    progress_dir = os.path.join(data_path,f"{data_file_prefix}_TMP")
    pbar = ProgressTracker.ProgressTracker(
        total_steps = num_processes*len(experiment_settings.alg_dict),
        progress_dir = progress_dir,
        title = data_file_prefix
    )
    worker_filename = os.path.join(code_path, "experiment_worker.py")
    activate = f"conda activate {CONDA_ENVIRONMENT_NAME}"
    subprocesses = []
    for process_idx in range(num_processes):
        data_filename = os.path.join(data_path,f"{data_file_prefix}_{process_idx}.json")
        argv = [
            worker_filename, 
            experiment_settings.__name__, 
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
filenames = glob.glob(os.path.join(data_path,f"{data_file_prefix}*.json"))
results_dict = visualize_results.read_combine_and_process_json(filenames)

values_sorted = [results_dict[key] for key in sorted(results_dict.keys())]

title = "Dosage bandit (outcome correlation: 0.5)"

visualize_results.plot_many(
    values_sorted, 
    plot_confidence=True,
    plot_baseline_rewards=False, 
    plot_random_timesteps=False,
    include_mean_safety=False,
    moving_avg_window=5, 
    title=title,
    figsize=(13,5),
    colors=None
)

visualize_results.plot_action_dist(
    values_sorted, 
    num_to_plot=10, 
    drop_first_action=False, 
    figsize=(13,6), 
    title=title
)