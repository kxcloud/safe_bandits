import glob
import os 
import subprocess

import _visualize_results as visualize_results
import ProgressBar 

CONDA_ENVIRONMENT_NAME = "base"
code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

#%% CHANGE SETTINGS HERE
import experiments.uniform_bandit as experiment_settings
num_processes = 3 
num_runs = 10
data_file_prefix = experiment_settings.__name__

#%% Run experiments
if num_processes is None:
    import experiment_worker
    data_filename = os.path.join(data_path, f"{data_file_prefix}.json")
    argv = [None, experiment_settings.__name__, data_filename, num_runs, os.path.join(data_path,f"{data_file_prefix}_TMP"), 1]
    experiment_worker.main(argv)
else:
    progress_dir = os.path.join(data_path,f"{data_file_prefix}_TMP")
    pbar = ProgressBar.ProgressBar(
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

title = "Uniform armed bandit"

visualize_results.plot_many(
    results_dict.values(), 
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
    results_dict.values(), 
    num_to_plot=10, 
    drop_first_action=False, 
    figsize=(13,6), 
    title=title
)