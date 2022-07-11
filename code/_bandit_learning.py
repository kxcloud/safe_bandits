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
        action_value = bandit.feature_vectorized(x, a) @ param
        if action_value > max_value:
            max_value = action_value
            a_max = a
    return a_max

#%% Algorithm subroutines

def estimate_safety_param_and_covariance(phi_XA, S, sqrt_weights):
    n, p = phi_XA.shape
    assert n > 0, "Must have at least one data point."
    
    phi_outer = [np.outer(phi, phi) for phi in phi_XA]
    cov_H = np.einsum("i,ijk->jk", sqrt_weights, phi_outer)/n
    
    beta_hat_S = utils.linear_regression(phi_XA, S, sqrt_weights)
    residuals_sq = (sqrt_weights*(phi_XA @ beta_hat_S - S))**2
    cov_G = np.mean(
        [phi_sq * resid_sq for (phi_sq, resid_sq) in zip(phi_outer, residuals_sq)],
        axis=0
    )
    
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
    
    test_stats = beta_hat_S @ phi_diff.squeeze()
    
    test_results = test_stats + safety_tol >= critical_value
    
    info = {"phi_diff" : phi_diff}
    return test_results, info

def test_many_actions(
        x, 
        a_baseline, 
        actions_to_test,
        alpha, 
        beta_hat_S, 
        sqrt_cov,
        bandit, 
        correct_for_multiple_testing,
        safety_tol
    ):
    if correct_for_multiple_testing: 
        alpha = alpha/len(actions_to_test)
    
    safe_actions = []
    for a in actions_to_test:
        test_result, _ = test_safety(
            x = x,
            a = a,
            a_baseline = a_baseline,
            beta_hat_S = beta_hat_S,
            sqrt_cov = sqrt_cov,
            alpha = alpha,
            phi = bandit.feature_vectorized,
            n = (bandit.t+1) * bandit.num_instances,
            safety_tol = safety_tol
        )
        if test_result:
            safe_actions.append(a)
    
    return safe_actions

def get_expected_improvement_objective(
        x, a_baseline, phi_XA, R, S, W, bandit, alpha, sqrt_cov, temperature,
        safety_tol, thompson_sampling
    ):
    """ 
    NOTE: currently bootstraps only the reward
    """
    
    if thompson_sampling:
        phi_XA_bs, R_bs = bsample([phi_XA, R])    
        beta_hat_R = utils.linear_regression(phi_XA_bs, R_bs, weights=None)
    else:
        beta_hat_R = utils.linear_regression(phi_XA, R, weights=None)
        
    num_bs_samples = 200
    beta_hats_S_bs = np.zeros((num_bs_samples, phi_XA.shape[1]))
    for bs_idx in range(num_bs_samples):
        phi_XA_bs, S_bs, W_bs = bsample([phi_XA, S, W])
        beta_hats_S_bs[bs_idx,:] = utils.linear_regression(phi_XA_bs, S_bs, W_bs)
    
    def get_pass_prob_and_improvement(a):
        test_results, info = test_safety(
            x = x,
            a = a,
            a_baseline = a_baseline,
            beta_hat_S = beta_hats_S_bs,
            sqrt_cov = sqrt_cov,
            alpha = alpha,
            phi = bandit.feature_vectorized,
            n = len(R),
            safety_tol = safety_tol
        )
        estimated_pass_prob = np.mean(test_results)
        estimated_improvement = info["phi_diff"].squeeze() @ beta_hat_R
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

def get_e_greedy_action_and_probs(a_greedy, epsilon, action_space):
    if random.random() < epsilon:
        a = np.random.choice(action_space)
    else:
        a = a_greedy
    
    if a == a_greedy:
        return a, 1-epsilon+epsilon/len(action_space)
    else:
        return a, epsilon/len(action_space)

def get_splits(random_seeds, overlap):
    # Given an array of random seeds and an overlap criterion, returns indices
    # for sample splitting
    assert 0 <= overlap <= 1, "overlap must be between 0 and 1"
    indices_0 = np.nonzero(random_seeds <= 0.5 + overlap/2)[0]
    indices_1 = np.nonzero(random_seeds > 0.5 - overlap/2)[0]
    
    if len(indices_0) == 0:
        indices_0 = np.array([0])
        
    if len(indices_1) == 0:
        indices_1 = np.array([0])
    
    return indices_0, indices_1

#%% Algorithms

def alg_eps_greedy(x, bandit, alpha, epsilon, safety_tol):
    beta_hat_R = utils.linear_regression(bandit.get_phi_XA(), bandit.get_R())
    a_max = get_best_action(x, beta_hat_R, bandit)
    
    a, a_prob = get_e_greedy_action_and_probs(a_max, epsilon(bandit.t), bandit.action_space)
    return a, a_prob, {}

def alg_fwer_pretest_eps_greedy(
        x, bandit, alpha, baseline_policy, epsilon, safety_tol
    ):
    
    a_baseline = baseline_policy(x)
    phi_XA = bandit.get_phi_XA()
    
    try:
        beta_hat_S, sqrt_cov = estimate_safety_param_and_covariance(phi_XA, bandit.get_S(), bandit.get_W())
    except np.linalg.linalg.LinAlgError:
        print(f"Linalg error on covariance estimation (t={bandit.t}). Sampling randomly.")
        return np.random.choice(bandit.action_space), None, {"safe_actions" : [a_baseline]}
    
    safe_actions = [a_baseline] + test_many_actions(
        x = x,
        a_baseline = a_baseline,
        actions_to_test = [a for a in bandit.action_space if a != a_baseline],
        alpha = alpha,
        beta_hat_S = beta_hat_S,
        sqrt_cov = sqrt_cov,
        bandit = bandit,
        correct_for_multiple_testing=True,
        safety_tol = safety_tol
    )
    info = {"safe_actions" : safe_actions}
         
    if len(safe_actions) == 1:
        a_selected = a_baseline
    else:
        beta_hat_R = utils.linear_regression(phi_XA, bandit.get_R(), None)
        a_selected = get_best_action(x, beta_hat_R, bandit, available_actions=safe_actions)
        info["beta_hat_R_bs"] = beta_hat_R
    
    a, a_prob = get_e_greedy_action_and_probs(a_selected, epsilon(bandit.t), bandit.action_space)
    return a, a_prob, info
    
def alg_unsafe_ts(x, bandit, alpha, epsilon, safety_tol):
    if random.random() < epsilon(bandit.t):
        return np.random.choice(bandit.action_space), None, {}
    
    phi_XA_bs, R_bs = bsample([bandit.get_phi_XA(), bandit.get_R()])    
    beta_hat_R_bs = utils.linear_regression(phi_XA_bs, R_bs, None)
    
    a = get_best_action(x, beta_hat_R_bs, bandit)
    return a, None, {}

def alg_fwer_pretest_ts(
        x, bandit, alpha, baseline_policy, epsilon, safety_tol
    ):
    if random.random() < epsilon(bandit.t):
        return np.random.choice(bandit.action_space), None, {}
    
    a_baseline = baseline_policy(x)
    phi_XA = bandit.get_phi_XA()
    
    try:
        beta_hat_S, sqrt_cov = estimate_safety_param_and_covariance(phi_XA, bandit.get_S(), bandit.get_W())
    except np.linalg.linalg.LinAlgError:
        print(f"Linalg error on covariance estimation (t={bandit.t}). Sampling randomly.")
        return np.random.choice(bandit.action_space), None, {"safe_actions" : [a_baseline]}
    
    safe_actions = [a_baseline] + test_many_actions(
        x = x,
        a_baseline = a_baseline,
        actions_to_test = [a for a in bandit.action_space if a != a_baseline],
        alpha = alpha,
        beta_hat_S = beta_hat_S,
        sqrt_cov = sqrt_cov,
        bandit = bandit,
        correct_for_multiple_testing=True,
        safety_tol = safety_tol
    )
    
    info = {"safe_actions" : safe_actions}
    
    if len(safe_actions) == 1:
        return safe_actions[0], None, info
  
    phi_XA_bs, R_bs = bsample([bandit.get_phi_XA(), bandit.get_R()])    
    beta_hat_R_bs = utils.linear_regression(phi_XA_bs, R_bs, None)
    
    a_hat = get_best_action(x, beta_hat_R_bs, bandit, available_actions=safe_actions)
    info["beta_hat_R_bs"] = beta_hat_R_bs
    return a_hat, None, info

def alg_propose_test_ts(
        x, 
        bandit, 
        alpha, 
        baseline_policy, 
        random_split, 
        objective_temperature, 
        use_out_of_sample_covariance,
        sample_overlap,
        thompson_sampling,
        epsilon,
        safety_tol
    ):
    if random.random() < epsilon(bandit.t):
        return np.random.choice(bandit.action_space), None, {}
    
    a_baseline = baseline_policy(x)
        
    X = bandit.get_X().copy()
    phi_XA = bandit.get_phi_XA().copy()
    R = bandit.get_R().copy()
    S = bandit.get_S().copy()
    W = bandit.get_W().copy()
    
    if random_split:
        n = len(X)
        shuffled_indices = np.random.choice(n, size=n, replace=False)
        for data in X, phi_XA, R, S:
            data[:] = data[shuffled_indices]
    
    indices_0, indices_1 = get_splits(bandit.get_U(), sample_overlap)
    phi_XA_1, R_1, S_1, W_1 = phi_XA[indices_0], R[indices_0], S[indices_0], W[indices_0]
    phi_XA_2, R_2, S_2, W_2 = phi_XA[indices_1], R[indices_1], S[indices_1], W[indices_1]
    
    # ESTIMATE SURROGATE OBJECTIVE ON SAMPLE 1
    try:
        beta_hat_S_2, sqrt_cov_2 = estimate_safety_param_and_covariance(phi_XA_2, S_2, W_2)
    except np.linalg.linalg.LinAlgError:
        print(f"Linalg error on test set covariance estimation (t={bandit.t}). Sampling randomly.")
        return np.random.choice(bandit.action_space), None, {}
    
    if use_out_of_sample_covariance:
        sqrt_cov = sqrt_cov_2
    else:
        try:
            _, sqrt_cov_1 = estimate_safety_param_and_covariance(phi_XA_1, S_1, W_1)
            sqrt_cov = sqrt_cov_1
        except np.linalg.linalg.LinAlgError:
            print(f"Linalg error on propose set covariance estimation (t={bandit.t}). Sampling randomly.")
            return np.random.choice(bandit.action_space), None, {}
    
    expected_improvement, split = get_expected_improvement_objective(
        x, a_baseline, phi_XA_1, R_1, S_1, W_1, bandit, alpha, sqrt_cov, 
        objective_temperature, safety_tol, thompson_sampling
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
        phi = bandit.feature_vectorized,
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

    a = a_hat if test_result else a_baseline
    return a, None, info
    

def alg_propose_test_ts_smart_explore(
        x, 
        bandit, 
        alpha, 
        baseline_policy, 
        random_split, 
        objective_temperature, 
        use_out_of_sample_covariance,
        sample_overlap,
        thompson_sampling,
        epsilon,
        safety_tol
    ):
    """
    Matches Propose Test TS except uniform random exploration is replaced
    by 1/2 chance of playing proposed action, 1/2 chance of uniform.
    
    OLD: ... replaced by maximization of (a bootstrap resampling of) Propose-
    Test objective which is NOT tested for safety.
    """
    
    a_baseline = baseline_policy(x)
    
    X = bandit.get_X().copy()
    phi_XA = bandit.get_phi_XA().copy()
    R = bandit.get_R().copy()
    S = bandit.get_S().copy()
    W = bandit.get_W().copy()
    
    if random_split:
        n = len(X)
        shuffled_indices = np.random.choice(n, size=n, replace=False)
        for data in X, phi_XA, R, S:
            data[:] = data[shuffled_indices]
    
    indices_0, indices_1 = get_splits(bandit.get_U(), sample_overlap)
    phi_XA_1, R_1, S_1, W_1 = phi_XA[indices_0], R[indices_0], S[indices_0], W[indices_0]
    phi_XA_2, R_2, S_2, W_2 = phi_XA[indices_1], R[indices_1], S[indices_1], W[indices_1]
    
    # ESTIMATE SURROGATE OBJECTIVE ON SAMPLE 1
    try:
        beta_hat_S_2, sqrt_cov_2 = estimate_safety_param_and_covariance(phi_XA_2, S_2, W_2)
    except np.linalg.linalg.LinAlgError:
        print(f"Linalg error on test set covariance estimation (t={bandit.t}). Sampling randomly.")
        return np.random.choice(bandit.action_space), None, {}
    
    if use_out_of_sample_covariance:
        sqrt_cov = sqrt_cov_2
    else:
        try:
            _, sqrt_cov_1 = estimate_safety_param_and_covariance(phi_XA_1, S_1, W_1)
            sqrt_cov = sqrt_cov_1
        except np.linalg.linalg.LinAlgError:
            print(f"Linalg error on propose set covariance estimation (t={bandit.t}). Sampling randomly.")
            return np.random.choice(bandit.action_space), None, {}
    
    expected_improvement, split = get_expected_improvement_objective(
        x, a_baseline, phi_XA_1, R_1, S_1, W_1, bandit, alpha, sqrt_cov, 
        objective_temperature, safety_tol, thompson_sampling
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
        phi = bandit.feature_vectorized,
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
    
    a_selected = a_hat if test_result else a_baseline
    
    a_probs = np.full(len(bandit.action_space), fill_value=epsilon(bandit.t)/len(bandit.action_space)/2)
    a_probs[bandit.action_idx[a_hat]] += epsilon(bandit.t)/2
    a_probs[bandit.action_idx[a_selected]] += 1-epsilon(bandit.t)
    a = np.random.choice(bandit.action_space, p=a_probs)
    a_prob = a_probs[bandit.action_idx[a]]
    return a, a_prob, info
    

def alg_propose_test_ts_fwer_fallback(
        x, bandit, alpha, baseline_policy, correct_alpha, num_actions_to_test, epsilon, safety_tol
    ):
    a_baseline = baseline_policy(x)

    alpha = alpha/2 if correct_alpha else alpha

    if random.random() < epsilon(bandit.t):
        return np.random.choice(bandit.action_space), None, {}
    
    a_safe_ts, _, _ = alg_propose_test_ts(
        x,
        bandit,
        alpha, 
        baseline_policy, 
        random_split=True,
        use_out_of_sample_covariance=False,
        objective_temperature=1, 
        thompson_sampling=False,
        sample_overlap=0,
        epsilon=lambda t: 0, 
        safety_tol=safety_tol
    )
    
    if a_safe_ts == a_baseline:
        a_pretest, _ , _ = alg_fwer_pretest_eps_greedy(
            x, 
            bandit, 
            alpha, 
            baseline_policy, 
            epsilon=lambda t: 0, 
            safety_tol=safety_tol
        )
        return a_pretest, None, {}
    else:
        return a_safe_ts, None, {}

#%% Evaluation functions

def get_reward_baselines(bandit, baseline_policy, safety_tol, num_samples=2000):
    bandit.reset(num_timesteps=1, num_instances=num_samples)
    X = bandit.sample()
        
    phi_baseline = bandit.feature_vectorized(X, [baseline_policy(x) for x in X])
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

    action_space = bandit_constructor().action_space
    
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
        "action_inds" : np.zeros((num_runs, num_alg_timesteps, num_instances, len(action_space)), dtype=bool),
        "action_space" : action_space,
        "alpha" : alpha,
        "safety_tol" : safety_tol,
        "num_runs" : num_runs,
        "duration" : None
    }
    
    for run_idx in range(num_runs):
        bandit = bandit_constructor()
        bandit.reset(total_timesteps, num_instances)
        
        for i in range(num_random_timesteps):
            bandit.sample() # Note: required to step bandit forward
            a_batch = np.random.choice(bandit.action_space, size=num_instances)
            a_probs_batch = np.full(num_instances, 1/len(bandit.action_space))
            bandit.act(a_batch, a_probs_batch)

        for t in range(num_alg_timesteps):
            x_batch = bandit.sample()
            a_batch = []
            a_prob_batch = []
            for instance_idx, x in enumerate(x_batch):
                a, a_prob, _ = learning_algorithm(
                    x=x, bandit=bandit, alpha=alpha, safety_tol=safety_tol
                )
                a_batch.append(a)
                results["action_inds"][run_idx, t, instance_idx, bandit.action_idx[a]] = 1
                a_prob_batch.append(a_prob)
            bandit.act(a_batch, a_prob_batch)
        
        results["mean_reward"][run_idx] = bandit.R_mean
        results["mean_safety"][run_idx] = bandit.S_mean
        
        agreement = [baseline_policy(x) == a for (x,a) in zip(bandit.X, bandit.A)]
        results["agreed_with_baseline"][run_idx] = agreement
        
        # Evaluate safety
        phi_baseline = np.zeros_like(bandit.phi_XA)
        for t in range(total_timesteps):
            for instance_idx in range(num_instances):
                x = bandit.X[t, instance_idx]
                a_safe = baseline_policy(x)
                phi_baseline[t, instance_idx] = bandit.feature_vectorized(x, a_safe)
        safety_baseline = phi_baseline @ bandit.safety_param
        results["safety_ind"][run_idx] = bandit.S_mean >= safety_baseline - safety_tol - 1e-8

    duration = (time.time() - start_time)/60
    results["duration"] = duration

    best_safe_reward, baseline_reward = get_reward_baselines(
        bandit_constructor(), baseline_policy, safety_tol
    )              
    results["best_safe_reward"] = best_safe_reward
    results["baseline_reward"] = baseline_reward    
        
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