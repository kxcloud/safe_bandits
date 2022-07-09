import glob
import os 
import subprocess

import _visualize_results as visualize_results

CONDA_ENVIRONMENT_NAME = "base"
code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

#%% CHANGE SETTINGS HERE
import experiments.dosage_bandit as experiment_settings
num_processes = 6
data_file_prefix = experiment_settings.__name__

#%% Run experiments
if num_processes is None:
    import experiment_worker
    data_filename = os.path.join(data_path, f"{data_file_prefix}.json")
    argv = [None, experiment_settings.__name__, data_filename]
    experiment_worker.main(argv)
else:
    worker_filename = os.path.join(code_path, "experiment_worker.py")
    activate = f"conda activate {CONDA_ENVIRONMENT_NAME}"
    subprocesses = []
    for process_idx in range(num_processes):
        data_filename = os.path.join(data_path,f"{data_file_prefix}_{process_idx}.json")
        run = " ".join(["python", worker_filename, experiment_settings.__name__, data_filename])
        command = " & ".join([activate, run])
        subp = subprocess.Popen(command, shell=True)
        subprocesses.append(subp)
    exit_codes = [p.wait() for p in subprocesses] 

#%% Plot
filenames = glob.glob(os.path.join(data_path,f"{data_file_prefix}*.json"))
results_dict = visualize_results.read_combine_and_process_json(filenames)

title = "Dosage bandit"

visualize_results.plot_many(
    results_dict.values(), 
    plot_confidence=True,
    plot_baseline_rewards=True, 
    plot_random_timesteps=False,
    include_mean_safety=False,
    moving_avg_window=5, 
    title=title,
    figsize=(13,5),
    colors=None
)

visualize_results.plot_action_dist(
    results_dict.values(), 
    num_to_plot=5, 
    drop_first_action=False, 
    figsize=(14,4), 
    title=title
)