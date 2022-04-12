from functools import partial

import numpy as np

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _visualize_results as visualize_results
import _utils as utils

baseline_policy = lambda x: 0

EPSILON = 0.1

alg_dict = {
    # "FWER pretest: TS (test 3)" : utils.wrapped_partial(
    #         bandit_learning.alg_fwer_pretest_ts, 
    #         baseline_policy=baseline_policy,
    #         num_actions_to_test=3,
    #         epsilon=EPSILON
    #     ),
    # "FWER pretest: TS (test 5)" : utils.wrapped_partial(
    #         bandit_learning.alg_fwer_pretest_ts, 
    #         baseline_policy=baseline_policy,
    #         num_actions_to_test=5,
    #         epsilon=EPSILON
    #     ),
    "Unsafe TS" : utils.wrapped_partial(
        bandit_learning.alg_unsafe_ts, 
        epsilon=EPSILON
        ),
    "FWER pretest: TS" : utils.wrapped_partial(
            bandit_learning.alg_fwer_pretest_ts, 
            baseline_policy=baseline_policy,
            num_actions_to_test=np.inf,
            epsilon=EPSILON
        ),
    # "Propose-test TS" : utils.wrapped_partial(
    #         bandit_learning.alg_propose_test_ts, 
    #         random_split=False, 
    #         use_out_of_sample_covariance=False,
    #         baseline_policy=baseline_policy,
    #         objective_temperature=1,
    #         epsilon=EPSILON
    #     ),
    "SPT-TS" : utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts_smart_explore, 
            random_split=False, 
            use_out_of_sample_covariance=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=EPSILON
        ),
    # "Propose-test TS (random split)" : utils.wrapped_partial(
    #         bandit_learning.alg_propose_test_ts, 
    #         random_split=True, 
    #         use_out_of_sample_covariance=False,
    #         baseline_policy=baseline_policy,
    #         objective_temperature=1,
    #         epsilon=EPSILON
    #     ),
    # "Propose-test TS (OOS covariance)" : utils.wrapped_partial(
    #         bandit_learning.alg_propose_test_ts, 
    #         random_split=True, 
    #         use_out_of_sample_covariance=True,
    #         baseline_policy=baseline_policy,
    #         objective_temperature=1,
    #         epsilon=EPSILON
    #     ),
    # "Propose-test TS (random split) (OOS cov)" : utils.wrapped_partial(
    #         bandit_learning.alg_propose_test_ts, 
    #         random_split=True, 
    #         use_out_of_sample_covariance=True,
    #         baseline_policy=baseline_policy,
    #         objective_temperature=1,
    #         epsilon=EPSILON
    #     )
}

num_runs = 500

total_duration = 0
num_actions_settings = [32] #, 100]
for num_actions in num_actions_settings:
    bandit_constructor = partial(
        BanditEnv.get_power_checker, num_actions=10, effect_size = 0.5,
    )
    
    results_dict = {}
    for alg_label, action_selection in alg_dict.items():
        results = bandit_learning.evaluate(
            alg_label,
            bandit_constructor,
            action_selection,
            baseline_policy = bandit_learning.baseline_policy,
            num_random_timesteps=10,
            num_alg_timesteps=20,
            num_instances=10,
            num_runs=num_runs,
            alpha=0.1,  
            safety_tol=0,
        )
        run_label = f"{alg_label}"
        results_dict[run_label] = results
    
    total_duration += sum([results["duration"] for results in results_dict.values()])
    
    filename = f"2022_04_12_power_checker_10_in_parallel.json"
    bandit_learning.save_to_json(results_dict, filename)
    
print(f"Total duration: {total_duration:0.02f} minutes.")
utils.print_run_counts_by_time(num_runs, total_duration)

#%% Plot

filename1 = f"2022_04_12_power_checker_10_in_parallel.json"

# filename2 = f"2021_11_30_random_polynomial_{num_actions}_actions_B.json"
# results_dict = visualize_results.read_combine_and_process_json([filename1,filename2])
results_dict = visualize_results.read_and_process_json(filename1)

title = None #f"Power testing bandit - hard to detect unsafe actions"
# bandit_constructor().plot(title=title)
visualize_results.plot_many(
    results_dict.values(), 
    plot_confidence=True,
    plot_baseline_rewards=True, 
    plot_random_timesteps=False,
    include_mean_safety=False,
    moving_avg_window=10, 
    title=title,
    figsize=(13,5)
)
