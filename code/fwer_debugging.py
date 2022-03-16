import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import BanditEnv
import bandit_learning

wrapped_partial = bandit_learning.wrapped_partial
baseline_policy = lambda x: 0

fwer = wrapped_partial(
    bandit_learning.alg_fwer_pretest_ts, 
    baseline_policy=baseline_policy,
    num_actions_to_test=np.inf,
    epsilon=0
)

ALPHA = 0.1
EFFECT_SIZE = 1
N_RUNS = 1_000
SAMPLES_PER_ACTION = 3
EFFECT_SIZES =  [0.5,1,2,4]

start_time = time.time()
results = []
for effect_size in EFFECT_SIZES:
    for num_actions in range(5, 200, 20):        
        
        false_positive_run_count = 0
        sum_of_detection_rates = 0
        
        for _ in range(N_RUNS):
            # Setup bandit
            safety_means = np.random.normal(size=num_actions) * effect_size
            safety_means[0] = 0
            
            safety_inds = safety_means >= safety_means[baseline_policy(0)]
            num_safe = sum(safety_inds)
            
            bandit = BanditEnv.get_standard_bandit(safety_means, outcome_std_dev=1)
            for _ in range(SAMPLES_PER_ACTION):
                for action in bandit.action_space:
                    bandit.sample()
                    bandit.act(action)
            
            test_results = bandit_learning.test_many_actions(
                x=0, 
                a_baseline=0, 
                num_actions_to_test=np.inf, 
                alpha=ALPHA, 
                phi_XA=np.array(bandit.phi_XA), 
                S=bandit.S, 
                bandit=bandit, 
                correct_for_multiple_testing=True
            )
            
            true_positives = 0
            false_positives = 0
            for tested_safe_action in test_results:
                if safety_inds[tested_safe_action]:
                    true_positives += 1
                else:
                    false_positives += 1
            
            false_negatives = num_safe - true_positives
            true_negatives = num_actions - num_safe - false_positives
            
            if false_positives > 0:
                false_positive_run_count += 1
            
            sum_of_detection_rates += (true_positives - 1)/(num_actions -1) # don't count action 0
            
        error_rate = false_positive_run_count/N_RUNS
        avg_detection_rate = sum_of_detection_rates / N_RUNS
    
        results.append((effect_size, num_actions, error_rate, avg_detection_rate))
duration = (time.time() - start_time)/60

print(f"Runtime: {duration:0.02f} minutes.")
res = pd.DataFrame(results, columns=["effect_size", "num_actions", "error_rate", "avg_num_detected"])   

#%%
fig, ax = plt.subplots()
ax.set_xlabel("Num actions")
ax.set_ylabel("Familywise error rate")

for effect_size in EFFECT_SIZES:
    subset = res[res["effect_size"] == effect_size]
    ax.plot(subset["num_actions"], subset["error_rate"], label=effect_size)

ax.legend(title="Effect size")
ax.set_title("Multi-armed bandit testing with Bonferonni correction:\nError rates")

#%%
fig, ax = plt.subplots()
ax.set_xlabel("Num actions")
ax.set_ylabel("Average (across runs) % of true positives identified")

for effect_size in EFFECT_SIZES:
    subset = res[res["effect_size"] == effect_size]
    ax.plot(subset["num_actions"], subset["avg_num_detected"], label=effect_size)

ax.legend(title="Effect size")
ax.set_title("Multi-armed bandit testing with Bonferonni correction:\n Power")

#%%


fig, ax = plt.subplots(figsize=(12,6))

ax.scatter(bandit.A, bandit.S, label="Observed")
ax.scatter(bandit.action_space, bandit.safety_param, marker="s")
ax.set_xlabel("Arm")
ax.set_ylabel("Safety")

print()
print(f"Test results ({len(test_results)} of {num_actions} pass):")
print(test_results)