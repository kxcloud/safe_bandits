import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _utils as utils

baseline_policy = lambda x: 0

fwer = utils.wrapped_partial(
    bandit_learning.alg_fwer_pretest_ts, 
    baseline_policy=baseline_policy,
    num_actions_to_test=np.inf,
    epsilon=0
)


def test_env(num_actions, effect_size):
    """
    A bandit with many obviously-low-reward actions.
    """
    num_good_actions = 1
    
    theta_reward = np.zeros(num_actions)
    theta_reward[1:num_good_actions+1] = 100
    
    theta_safety = -np.ones(num_actions) * 100
    theta_safety[0] = 0
    theta_safety[1] = effect_size
    
    action_space = range(num_actions)
    
    def feature_vector(x, a):
        if type(x) is np.ndarray:
            phi_xa = np.zeros((len(x), num_actions))
            phi_xa[:, a] = 1
        else:
            phi_xa = np.zeros(num_actions)
            phi_xa[a] =1
        return phi_xa
                
    bandit = BanditEnv.BanditEnv(
        x_dist=lambda : 0, 
        action_space=action_space,
        feature_vector=feature_vector,
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_std_dev=1,
        outcome_correlation=0
    )
    return bandit

ALPHA = 0.1
N_RUNS = 300
SAMPLES_PER_ACTION = 6
EFFECT_SIZES =  [0.5, 1, 2]

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