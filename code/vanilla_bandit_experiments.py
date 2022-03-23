from functools import partial

import numpy as np

import BanditEnv
import bandit_learning
import visualize_results

wrapped_partial = bandit_learning.wrapped_partial
baseline_policy = lambda x: 0

bandit_constructor = partial(BanditEnv.get_dosage_example, num_actions=20, param_count= 10)

EPSILON = 0.1
safety_tol = 0.1

alg_dict = {
    "Unsafe TS" : wrapped_partial(
        bandit_learning.alg_unsafe_ts, 
        epsilon=EPSILON
        ),
    "FWER pretest: TS (test all)" : wrapped_partial(
            bandit_learning.alg_fwer_pretest_ts, 
            baseline_policy=baseline_policy,
            num_actions_to_test=np.inf,
            epsilon=EPSILON,
        ),
    "Propose-test TS" : wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=False, 
            use_out_of_sample_covariance=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=EPSILON,
        ),
    "Propose-test TS (smart explore)" : wrapped_partial(
            bandit_learning.alg_propose_test_ts_smart_explore, 
            random_split=False, 
            use_out_of_sample_covariance=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=EPSILON,
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
        num_runs=3000,
        alpha=0.1,    
        safety_tol=0.1
    )
    results_dict[alg_label] = results

total_duration = sum([results["duration"] for results in results_dict.values()])
print(f"Total duration: {total_duration:0.02f} minutes.")

bandit_learning.save_to_json(results_dict, "2022_03_22_dosage_example_C.json")

# %% Plot

filename0 = "2022_03_22_dosage_example.json"
filename1 = "2022_03_22_dosage_example_B.json"
filename2 = "2022_03_22_dosage_example_C.json"
filenames = [filename0, filename1, filename2]
results_dict = visualize_results.read_combine_and_process_json(filenames)

    
title = f"Dosage example"
# bandit_constructor().plot(title=title)
visualize_results.plot_many(
    results_dict.values(), 
    plot_confidence=True,
    plot_baseline_rewards=False, 
    plot_random_timesteps=False,
    moving_avg_window=10, 
    title=title,
    figsize=(13,8)
)
