import copy
import time
import json
import os
import random
from functools import partial

import numpy as np
from scipy.stats import norm

import _utils as utils

code_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(code_path)
data_path = os.path.join(project_path,"data")

COV_STABILIZATION_AMT = 1e-8
print(f"Covariance stabilization: {COV_STABILIZATION_AMT}")

def bsample(list_of_data):
    n = len(list_of_data[0])
    bs_indices = np.random.choice(n, size=n)
    bs_data = [data[bs_indices] for data in list_of_data]
    return bs_data

def get_best_action(x, param, bandit, available_actions=None):
    if available_actions is None:
        available_actions = bandit.action_space
        
    max_value = -np.inf
    a_max = None
    for a in available_actions:    
        action_value = bandit.feature_vector(x, a) @ param
        if action_value > max_value:
            max_value = action_value
            a_max = a
    return a_max

#%% Algorithm subroutines

def estimate_safety_param_and_covariance(phi_XA, S):
    n, p = phi_XA.shape
    phi_outer = [np.outer(phi, phi) for phi in phi_XA]
    stabilization_term = np.identity(p) * COV_STABILIZATION_AMT
    
    cov_H = np.mean(phi_outer, axis=0) + stabilization_term

    beta_hat_S = utils.linear_regression(phi_XA, S)
    residuals_sq = (phi_XA @ beta_hat_S - S)**2
    
    cov_G = np.mean(
        [phi_sq * resid_sq for (phi_sq, resid_sq) in zip(phi_outer, residuals_sq)],
        axis=0
    ) + stabilization_term 
    
    sqrt_G = np.linalg.cholesky(cov_G)
    sqrt_cov = np.linalg.solve(cov_H, sqrt_G)
    # Note: cov = sqrt_cov @ sqrt_cov.T
    return beta_hat_S, sqrt_cov

def test_safety(
        x, a, a_baseline, beta_hat_S, sqrt_cov, alpha, phi, n, safety_tol
    ):
    """ 
    Runs multiple tests if multiple values of beta_hat_S are passed. 
    NOTE: does not resample covariance matrix, which could be problematic.
    """   
    phi_diff = phi(x, a) - phi(x, a_baseline)
    
    std_err = np.sqrt(np.sum((phi_diff @ sqrt_cov)**2))
    critical_value = norm.ppf(1-alpha) * std_err / np.sqrt(n)
       
    test_stats = beta_hat_S @ phi_diff
    
    test_results = test_stats + safety_tol >= critical_value
    
    info = {"phi_diff" : phi_diff}
    return test_results, info

def test_many_actions(
        x, 
        a_baseline, 
        num_actions_to_test, 
        alpha, 
        phi_XA, 
        S, 
        bandit, 
        correct_for_multiple_testing,
        safety_tol
    ):
    beta_hat_S, sqrt_cov = estimate_safety_param_and_covariance(phi_XA, S)
    
    if num_actions_to_test >= len(bandit.action_space):
        actions_to_test = bandit.action_space
        num_actions_to_test = len(bandit.action_space)
    else:    
        actions_to_test = np.random.choice(
            bandit.action_space, size=num_actions_to_test, replace=False
        )
        
    if correct_for_multiple_testing: 
        alpha = alpha/num_actions_to_test  
    
    safe_actions = []
    for a in actions_to_test:
        test_result, _ = test_safety(
            x = x,
            a = a,
            a_baseline = a_baseline,
            beta_hat_S = beta_hat_S,
            sqrt_cov = sqrt_cov,
            alpha = alpha,
            phi = bandit.feature_vector,
            n = len(phi_XA),
            safety_tol = safety_tol
        )
        if test_result:
            safe_actions.append(a)
    
    return safe_actions

def get_expected_improvement_objective(
        x, a_baseline, phi_XA, R, S, bandit, alpha, sqrt_cov, temperature,
        safety_tol
    ):
    """ 
    NOTE: currently bootstraps only the reward
    """
    
    phi_XA_bs, R_bs = bsample([phi_XA, R])    
    beta_hat_R_bs = utils.linear_regression(phi_XA_bs, R_bs)
        
    num_bs_samples = 200
    beta_hats_S_bs = np.zeros((num_bs_samples, phi_XA.shape[1]))
    for bs_idx in range(num_bs_samples):
        phi_XA_bs, S_bs = bsample([phi_XA, S])
        beta_hats_S_bs[bs_idx,:] = utils.linear_regression(phi_XA_bs, S_bs)
    
    def get_pass_prob_and_improvement(a):
        test_results, info = test_safety(
            x = x,
            a = a,
            a_baseline = a_baseline,
            beta_hat_S = beta_hats_S_bs,
            sqrt_cov = sqrt_cov,
            alpha = alpha,
            phi = bandit.feature_vector,
            n = len(R),
            safety_tol = safety_tol
        )
        estimated_pass_prob = np.mean(test_results)
        estimated_improvement = info["phi_diff"] @ beta_hat_R_bs
        return estimated_pass_prob, estimated_improvement
    
    def expected_improvement(a):
        if a == a_baseline:
            return 0
        
        estimated_pass_prob, estimated_improvement = get_pass_prob_and_improvement(a)
        value = estimated_improvement * estimated_pass_prob**temperature
        return value
    
    return expected_improvement, get_pass_prob_and_improvement

def maximize(objective, input_values):
    objective_vals = [objective(val) for val in input_values]
    argmax_idx = np.argmax(objective_vals)
    argmax = input_values[argmax_idx]
    return argmax

#%% Algorithms

def alg_eps_greedy(x, bandit, alpha, epsilon, safety_tol):
    if random.random() < epsilon:
        return np.random.choice(bandit.action_space), {}
    
    beta_hat_R = utils.linear_regression(bandit.get_phi_XA(), bandit.get_R())
    
    a = get_best_action(x, beta_hat_R, bandit)
    return a, {}

def alg_fwer_pretest_eps_greedy(
        x, bandit, alpha, baseline_policy, num_actions_to_test, epsilon, safety_tol
    ):   
    if random.random() < epsilon:
        return np.random.choice(bandit.action_space), {}
    
    a_baseline = baseline_policy(x)
    
    safe_actions = test_many_actions(
        x = x,
        a_baseline = a_baseline,
        num_actions_to_test = num_actions_to_test,
        alpha = alpha,
        phi_XA = bandit.get_phi_XA(),
        S = bandit.get_S(),
        bandit = bandit,
        correct_for_multiple_testing=True,
        safety_tol = safety_tol
    )
         
    if len(safe_actions) == 0:
        return a_baseline, {}
    
    beta_hat_R = utils.linear_regression(bandit.get_phi_XA(), bandit.get_R())
    a_hat = get_best_action(x, beta_hat_R, bandit, available_actions=safe_actions)
    return a_hat, {}

def alg_unsafe_ts(x, bandit, alpha, epsilon, safety_tol):
    if random.random() < epsilon:
        return np.random.choice(bandit.action_space), {}
    
    phi_XA_bs, R_bs = bsample([bandit.get_phi_XA(), bandit.get_R()])    
    beta_hat_R_bs = utils.linear_regression(phi_XA_bs, R_bs)
    
    a = get_best_action(x, beta_hat_R_bs, bandit)
    return a, {}

def alg_fwer_pretest_ts(
        x, bandit, alpha, baseline_policy, num_actions_to_test, epsilon, safety_tol
    ):
    if random.random() < epsilon:
        return np.random.choice(bandit.action_space), {}
    
    a_baseline = baseline_policy(x)
    
    safe_actions = test_many_actions(
        x = x,
        a_baseline = a_baseline,
        num_actions_to_test = num_actions_to_test,
        alpha = alpha,
        phi_XA = bandit.get_phi_XA(),
        S = bandit.get_S(),
        bandit = bandit,
        correct_for_multiple_testing = True,
        safety_tol = safety_tol
    )
    
    info = {"safe_actions" : safe_actions}
    if len(safe_actions) == 0:
        return a_baseline, info
    
    if len(safe_actions) == 1:
        return safe_actions[0], info
  
    phi_XA_bs, R_bs = bsample([bandit.get_phi_XA(), bandit.get_R()])    
    beta_hat_R_bs = utils.linear_regression(phi_XA_bs, R_bs)
    
    a_hat = get_best_action(x, beta_hat_R_bs, bandit, available_actions=safe_actions)
    info["beta_hat_R_bs"] = beta_hat_R_bs
    return a_hat, info

def alg_propose_test_ts(
        x, 
        bandit, 
        alpha, 
        baseline_policy, 
        random_split, 
        objective_temperature, 
        use_out_of_sample_covariance,
        epsilon,
        safety_tol
    ):
    a_baseline = baseline_policy(x)

    if random.random() < epsilon:
        return np.random.choice(bandit.action_space), {}
        
    X = bandit.get_X()
    phi_XA = bandit.get_phi_XA()
    R = bandit.get_R()
    S = bandit.get_S()
    
    if random_split:
        n = len(X)
        shuffled_indices = np.random.choice(n, size=n, replace=False)
        for data in X, phi_XA, R, S:
            data[:] = data[shuffled_indices]
    
    phi_XA_1, R_1, S_1 = phi_XA[::2], R[::2], S[::2]
    phi_XA_2, R_2, S_2 = phi_XA[1::2], R[1::2], S[1::2]
    
    # ESTIMATE SURROGATE OBJECTIVE ON SAMPLE 1
    beta_hat_S_2, sqrt_cov_2 = estimate_safety_param_and_covariance(phi_XA_2, S_2)
    
    if use_out_of_sample_covariance:
        sqrt_cov = sqrt_cov_2
    else:
        _, sqrt_cov_1 = estimate_safety_param_and_covariance(phi_XA_1, S_1)
        sqrt_cov = sqrt_cov_1
    
    expected_improvement, split = get_expected_improvement_objective(
        x, a_baseline, phi_XA_1, R_1, S_1, bandit, alpha, sqrt_cov, objective_temperature,
        safety_tol
    )
    
    a_hat = maximize(expected_improvement, bandit.action_space)
    
    # TEST SELECTION ON SAMPLE 2      
    safety_test = partial(
        test_safety,
        x = x,
        a_baseline = a_baseline,
        beta_hat_S = beta_hat_S_2,
        sqrt_cov = sqrt_cov_2,
        alpha = alpha,
        phi = bandit.feature_vector,
        n = len(S_2),
        safety_tol = safety_tol
    )
    
    test_result, _ = safety_test(a=a_hat)
    
    info = {
        "objective" : expected_improvement, 
        "beta_hat_S_2": beta_hat_S_2,
        "split_objective": split,
        "safety_test" : safety_test
    }

    if test_result:
        return a_hat, info
    else:
        return a_baseline, info

def alg_propose_test_ts_smart_explore(
        x, 
        bandit, 
        alpha, 
        baseline_policy, 
        random_split, 
        objective_temperature, 
        use_out_of_sample_covariance,
        epsilon,
        safety_tol
    ):
    """
    Matches Propose Test TS except uniform random exploration is replaced
    by maximization of (a bootstrap resampling of) Propose-Test objective
    which is NOT tested for safety.
    """
    
    a_baseline = baseline_policy(x)
    
    X = bandit.get_X()
    phi_XA = bandit.get_phi_XA()
    R = bandit.get_R()
    S = bandit.get_S()
    
    if random_split:
        n = len(X)
        shuffled_indices = np.random.choice(n, size=n, replace=False)
        for data in X, phi_XA, R, S:
            data[:] = data[shuffled_indices]
    
    phi_XA_1, R_1, S_1 = phi_XA[::2], R[::2], S[::2]
    phi_XA_2, R_2, S_2 = phi_XA[1::2], R[1::2], S[1::2]
    
    # ESTIMATE SURROGATE OBJECTIVE ON SAMPLE 1
    beta_hat_S_2, sqrt_cov_2 = estimate_safety_param_and_covariance(phi_XA_2, S_2)
    
    if use_out_of_sample_covariance:
        sqrt_cov = sqrt_cov_2
    else:
        _, sqrt_cov_1 = estimate_safety_param_and_covariance(phi_XA_1, S_1)
        sqrt_cov = sqrt_cov_1
    
    expected_improvement, split = get_expected_improvement_objective(
        x, a_baseline, phi_XA_1, R_1, S_1, bandit, alpha, sqrt_cov, objective_temperature, safety_tol
    )
    
    a_hat = maximize(expected_improvement, bandit.action_space)
    
    if random.random() < epsilon:
        return a_hat, {}
    
    # TEST SELECTION ON SAMPLE 2      
    safety_test = partial(
        test_safety,
        x = x,
        a_baseline = a_baseline,
        beta_hat_S = beta_hat_S_2,
        sqrt_cov = sqrt_cov_2,
        alpha = alpha,
        phi = bandit.feature_vector,
        n = len(S_2),
        safety_tol = safety_tol
    )
    
    test_result, _ = safety_test(a=a_hat)
    
    info = {
        "objective" : expected_improvement, 
        "beta_hat_S_2": beta_hat_S_2,
        "split_objective": split,
        "safety_test" : safety_test
    }

    if test_result:
        return a_hat, info
    else:
        return a_baseline, info

def alg_propose_test_ts_fwer_fallback(
        x, bandit, alpha, baseline_policy, correct_alpha, num_actions_to_test, epsilon, safety_tol
    ):
    a_baseline = baseline_policy(x)

    alpha = alpha/2 if correct_alpha else alpha

    if random.random() < epsilon:
        return np.random.choice(bandit.action_space), {}
    
    a_safe_ts, _ = alg_propose_test_ts(
        x, bandit, alpha, baseline_policy, random_split=True, use_out_of_sample_covariance=True,
        objective_temperature=1, epsilon=0, safety_tol=safety_tol
    )
    
    if a_safe_ts == a_baseline:
        a_pretest, _ = alg_fwer_pretest_eps_greedy(
            x, bandit, alpha, baseline_policy, num_actions_to_test=num_actions_to_test, epsilon=0, safety_tol=safety_tol
        )
        return a_pretest, {}
    else:
        return a_safe_ts, {}

#%% Evaluation functions

def get_reward_baselines(bandit, baseline_policy, safety_tol, num_samples=2000):
    bandit.reset(num_timesteps=1, num_instances=num_samples)
    X = bandit.sample()
        
    phi_baseline = bandit.feature_vectorized(X, baseline_policy(X))
    safety_baseline = phi_baseline @ bandit.safety_param
    
    # Figure out the best reward obtainable by oracle safe policy
    num_actions = len(bandit.action_space)
    action_reward_masked_by_safety = np.zeros((num_samples, num_actions))
    for a_idx, a in enumerate(bandit.action_space):
        phi_X_a = bandit.feature_vectorized(X, a)
        safety = phi_X_a @ bandit.safety_param
        is_safe = safety >= safety_baseline - safety_tol
        
        reward = phi_X_a @ bandit.reward_param
        
        reward_masked_by_safety = np.array([
             r if safe else -np.inf for r, safe in zip(reward, is_safe)
        ])
        
        action_reward_masked_by_safety[:, a_idx] = reward_masked_by_safety
    
    best_safe_rewards = np.max(action_reward_masked_by_safety, axis=1)
    assert len(best_safe_rewards) == num_samples
    best_average_safe_reward = np.mean(best_safe_rewards)
    baseline_average_reward = np.mean(phi_baseline @ bandit.reward_param)
    return best_average_safe_reward, baseline_average_reward

def evaluate(
        alg_label,
        bandit_constructor,
        learning_algorithm,
        baseline_policy,
        num_random_timesteps,
        num_alg_timesteps,
        num_instances,
        num_runs,
        alpha, 
        safety_tol,
        print_time=True
    ):
    start_time = time.time()

    total_timesteps = num_random_timesteps + num_alg_timesteps
    results = {
        "bandit_name" : bandit_constructor().__class__.__name__,
        "alg_label" : alg_label,
        "alg_name" : learning_algorithm.__name__,
        "num_random_timesteps" : num_random_timesteps,
        "num_alg_timesteps" : num_alg_timesteps,
        "num_instances" : num_instances,
        "mean_reward" : np.zeros((num_runs, total_timesteps, num_instances)),
        "mean_safety" : np.zeros((num_runs, total_timesteps, num_instances)),
        "safety_ind" : np.zeros((num_runs, total_timesteps, num_instances), dtype=bool),
        "agreed_with_baseline" : np.zeros((num_runs, total_timesteps, num_instances), dtype=bool),
        "alpha" : alpha,
        "safety_tol" : safety_tol,
        "duration" : None
    }

    for run_idx in range(num_runs):
        bandit = bandit_constructor()
        bandit.reset(total_timesteps, num_instances)
        
        for i in range(num_random_timesteps):
            bandit.sample() # Note: required to step bandit forward
            a_batch = np.random.choice(bandit.action_space, size=num_instances)
            bandit.act(a_batch)

        for t in range(num_alg_timesteps):
            x_batch = bandit.sample()
            a_batch = []
            for x in x_batch:
                a, _ = learning_algorithm(x=x, bandit=bandit, alpha=alpha, safety_tol=safety_tol)
                a_batch.append(a)
            bandit.act(a_batch)
        
        results["mean_reward"][run_idx] = bandit.R_mean
        results["mean_safety"][run_idx] = bandit.S_mean
        
        agreement = baseline_policy(bandit.X) == bandit.A
        results["agreed_with_baseline"][run_idx] = agreement
        
        # Evaluate safety
        phi_baseline = np.zeros_like(bandit.phi_XA)
        for t in range(total_timesteps):
            for instance_idx in range(num_instances):
                x = bandit.X[t, instance_idx]
                a_safe = baseline_policy(x)
                phi_baseline[t, instance_idx] = bandit.feature_vector(x, a_safe)
        safety_baseline = phi_baseline @ bandit.safety_param
        results["safety_ind"][run_idx] = bandit.S_mean >= safety_baseline - safety_tol - 1e-8

    best_safe_reward, baseline_reward = get_reward_baselines(
        bandit_constructor(), baseline_policy, safety_tol
    )              
    results["best_safe_reward"] = best_safe_reward
    results["baseline_reward"] = baseline_reward    
    
    duration = (time.time() - start_time)/60
    results["duration"] = duration
    
    if print_time:
        print(
            f"Evaluated {learning_algorithm.__name__} {num_runs}"
            f" times in {duration:0.02f} minutes."
        )
    return results

def save_to_json(results_dict, filename):
    results = {}
    
    # Make JSON-compatible
    for run_label, run_data in results_dict.items():
        run_data = copy.deepcopy(run_data)
        for label, item in run_data.items():
            if type(item) is np.ndarray:
                run_data[label] = item.tolist()
        results[run_label] = run_data
        
    with open(os.path.join(data_path, filename), 'w') as f:
        json.dump(results, f)
            
#%% Evaluation
baseline_policy = lambda x : 0

alg_dict = {
    "Unsafe e-greedy" : utils.wrapped_partial(alg_eps_greedy, epsilon=0.1),
    "Unsafe TS" : utils.wrapped_partial(alg_unsafe_ts, epsilon=0.1),
    "FWER pretest (all): e-greedy" : utils.wrapped_partial(
            alg_fwer_pretest_eps_greedy, baseline_policy=baseline_policy, num_actions_to_test=np.inf, epsilon=0.1
        ),
    "FWER pretest (all): TS" :  utils.wrapped_partial(
            alg_fwer_pretest_ts, baseline_policy=baseline_policy, num_actions_to_test=np.inf, epsilon=0.1
        ),
    "Propose-test TS" : utils.wrapped_partial(
            alg_propose_test_ts, random_split=False, baseline_policy=baseline_policy, objective_temperature=1, use_out_of_sample_covariance=False, epsilon=0.1
        ),
    "Propose-test TS (OOS covariance)" : utils.wrapped_partial(
            alg_propose_test_ts, random_split=False, baseline_policy=baseline_policy, objective_temperature=1, use_out_of_sample_covariance=True, epsilon=0.1
        ),
    "Propose-test TS (random split)" : utils.wrapped_partial(
            alg_propose_test_ts, random_split=True, baseline_policy=baseline_policy, objective_temperature=1, use_out_of_sample_covariance=False, epsilon=0.1
        ),
    "Propose-test TS (random) (OOS)" : utils.wrapped_partial(
            alg_propose_test_ts, random_split=True, baseline_policy=baseline_policy, objective_temperature=1, use_out_of_sample_covariance=True, epsilon=0.1
        ),
    "Propose-test TS (safe FWER fallback [all])" : utils.wrapped_partial(
            alg_propose_test_ts_fwer_fallback, correct_alpha=True, num_actions_to_test=np.inf, baseline_policy=baseline_policy, epsilon=0.1
        ),  
}