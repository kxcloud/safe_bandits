import sys
import os 

import numpy as np

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _visualize_results as visualize_results
import _utils as utils

import experiment_2022_07_08.py as experiment_settings

code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")


#%% Run experiment workers

for process_idx in range(experiment_settings.num_proceses):
    os.subprocesses()


#%% Plot

filenames = [
    "2022_07_08_uniform_armed_2_A.json",
    "2022_07_08_uniform_armed_2_B.json",
    "2022_07_08_uniform_armed_2_C.json",
]
results_dict = visualize_results.read_combine_and_process_json(filenames)

colors = None #["C1", "C3", "C2"]

title = "Uniform armed bandit"

visualize_results.plot_many(
    results_dict.values(), 
    plot_confidence=True,
    plot_baseline_rewards=True, 
    plot_random_timesteps=False,
    include_mean_safety=False,
    moving_avg_window=5, 
    title=title,
    figsize=(13,5),
    colors=colors
)

visualize_results.plot_action_dist(
    results_dict.values(), 
    num_to_plot=5, 
    drop_first_action=False, 
    figsize=(14,4), 
    title=title
)

