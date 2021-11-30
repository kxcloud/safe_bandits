from functools import partial

import numpy as np

import BanditEnv
import bandit_learning
import visualize_results

wrapped_partial = bandit_learning.wrapped_partial
baseline_policy = lambda x: 0

alg_dict = {
    # "FWER pretest: TS (test 3)" : wrapped_partial(
    #         bandit_learning.alg_fwer_pretest_ts, 
    #         baseline_policy=baseline_policy,
    #         num_actions_to_test=3,
    #         epsilon=0.1
    #     ),
    # "FWER pretest: TS (test 5)" : wrapped_partial(
    #         bandit_learning.alg_fwer_pretest_ts, 
    #         baseline_policy=baseline_policy,
    #         num_actions_to_test=5,
    #         epsilon=0.1
    #     ),
    "FWER pretest: TS (test all)" : wrapped_partial(
            bandit_learning.alg_fwer_pretest_ts, 
            baseline_policy=baseline_policy,
            num_actions_to_test=np.inf,
            epsilon=0.1
        ),
    "Propose-test TS" : wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=False, 
            use_out_of_sample_covariance=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=0.1
        ),
    "Propose-test TS (random split)" : wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=True, 
            use_out_of_sample_covariance=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=0.1
        ),
    "Propose-test TS (OOS covariance)" : wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=True, 
            use_out_of_sample_covariance=True,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=0.1
        ),
    "Propose-test TS (random split) (OOS cov)" : wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=True, 
            use_out_of_sample_covariance=True,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=0.1
        )
}

total_duration = 0
num_actions_settings = [6, 12]
for num_actions in num_actions_settings:
    bandit_constructor = partial(
        BanditEnv.get_random_polynomial_bandit, num_actions=num_actions, outcome_correlation=0
    )
    
    results_dict = {}
    for alg_label, action_selection in alg_dict.items():
        results = bandit_learning.evaluate(
            alg_label,
            bandit_constructor,
            action_selection,
            baseline_policy = bandit_learning.baseline_policy,
            num_random_timesteps=10,
            num_alg_timesteps=250,
            num_runs=1000,
            alpha=0.1,    
        )
        run_label = f"{alg_label}"
        results_dict[run_label] = results
    
    total_duration += sum([results["duration"] for results in results_dict.values()])
    
    filename = f"2021_11_30_random_polynomial_{num_actions}_actions_B.json"
    bandit_learning.save_to_json(results_dict, filename)
    
print(f"Total duration: {total_duration:0.02f} minutes.")

#%% Plot

for num_actions in num_actions_settings:
    filename1 = f"2021_11_30_random_polynomial_{num_actions}_actions.json"
    filename2 = f"2021_11_30_random_polynomial_{num_actions}_actions_B.json"
    results_dict = visualize_results.read_combine_and_process_json([filename1,filename2])
    
    title = f"Random polynomial bandit (num_actions={num_actions})"
    # bandit_constructor().plot(title=title)
    visualize_results.plot_many(
        results_dict.values(), 
        plot_confidence=True,
        plot_baseline_rewards=False, 
        plot_random_timesteps=False,
        moving_avg_window=10, 
        title=title,
        figsize=(13,5)
    )
