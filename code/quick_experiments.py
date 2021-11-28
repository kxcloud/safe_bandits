from functools import partial

import numpy as np

import BanditEnv
import bandit_learning
import visualize_results

wrapped_partial = bandit_learning.wrapped_partial
baseline_policy = lambda x: 0

alg_dict = {
    "FWER pretest: TS (test 3)" : wrapped_partial(
            bandit_learning.alg_fwer_pretest_ts, 
            baseline_policy=baseline_policy,
            num_actions_to_test=3,
            epsilon=0.1
        ),
    "FWER pretest: TS (test 5)" : wrapped_partial(
            bandit_learning.alg_fwer_pretest_ts, 
            baseline_policy=baseline_policy,
            num_actions_to_test=5,
            epsilon=0.1
        ),
    "FWER pretest: TS (test all)" : wrapped_partial(
            bandit_learning.alg_fwer_pretest_ts, 
            baseline_policy=baseline_policy,
            num_actions_to_test=np.inf,
            epsilon=0.1
        ),
    "Propose-test TS" : wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=False, 
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=0.1
        )
}

total_duration = 0
for num_actions in [4, 8, 16]:
    bandit_constructor = partial(BanditEnv.get_random_polynomial_bandit, num_actions=num_actions)
    
    results_dict = {}
    for alg_label, action_selection in alg_dict.items():
        results = bandit_learning.evaluate(
            alg_label,
            bandit_constructor,
            action_selection,
            baseline_policy = bandit_learning.baseline_policy,
            num_random_timesteps=10,
            num_alg_timesteps=150,
            num_runs=300,
            alpha=0.1,    
        )
        run_label = f"{alg_label}"
        results_dict[run_label] = results
    
    total_duration += sum([results["duration"] for results in results_dict.values()])
    
    bandit_learning.save_to_json(results_dict, f"2021_11_28_random_polynomial_{num_actions}_actions.json")
    
    title = f"Random polynomial bandit (num_actions={num_actions})"
    bandit_constructor().plot(title=title)
    visualize_results.plot_many(results_dict.values(), plot_baseline_rewards=False, moving_avg_window=10, title=title)
    
print(f"Total duration: {total_duration:0.02f} minutes.")