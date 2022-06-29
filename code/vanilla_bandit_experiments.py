from functools import partial

import numpy as np

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _visualize_results as visualize_results
import _utils as utils

wrapped_partial = utils.wrapped_partial
baseline_policy = lambda x: 0

bandit_constructor = partial(BanditEnv.get_dosage_example, num_actions=20, param_count=10)

EPSILON = 0.1
safety_tol = 0.1
num_runs = 1000

alg_dict = {
    "FWER pretest" : utils.wrapped_partial(
            bandit_learning.alg_fwer_pretest_eps_greedy, 
            baseline_policy=baseline_policy,
            num_actions_to_test=np.inf,
            epsilon=EPSILON
        ),
    "SPT" : utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=False, 
            use_out_of_sample_covariance=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=EPSILON
        ),
    "SPT (smart explore)" : utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts_smart_explore, 
            random_split=False, 
            use_out_of_sample_covariance=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=EPSILON
        ),
}
#%% Run
results_dict = {}
for alg_label, learning_algorithm in alg_dict.items():
    results = bandit_learning.evaluate(
        alg_label,
        bandit_constructor,
        learning_algorithm,
        baseline_policy = baseline_policy,
        num_random_timesteps=100,
        num_alg_timesteps=200,
        num_runs=num_runs,
        num_instances=1,
        alpha=0.1,    
        safety_tol=0.1
    )
    results_dict[alg_label] = results

total_duration = sum([results["duration"] for results in results_dict.values()])
print(f"Total duration: {total_duration:0.02f} minutes.")
utils.print_run_counts_by_time(num_runs, total_duration)

bandit_learning.save_to_json(results_dict, "2022_06_28_dosage_example_C.json")

# %% Plot

filename0 = "2022_06_28_dosage_example.json"
filename1 = "2022_06_28_dosage_example_B.json"
filename2 = "2022_06_28_dosage_example_C.json"
filenames = [filename0, filename1, filename2]
results_dict = visualize_results.read_combine_and_process_json(filenames)

colors = ["C1", "C3", "C2"]

title = "Dosage bandit"
# bandit_constructor().plot(title=title)
visualize_results.plot_many(
    results_dict.values(), 
    plot_confidence=False,
    plot_baseline_rewards=True, 
    plot_random_timesteps=False,
    include_mean_safety=False,
    moving_avg_window=10, 
    title=title,
    figsize=(13,5),
    colors=colors
)
