import time
import random
from functools import partial, update_wrapper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.stats import norm

import BanditEnv

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def linear_regression(x_mat, y, penalty=0.01):
    return np.linalg.solve(x_mat.T @ x_mat + penalty * np.identity(x_mat.shape[1]), x_mat.T @ y)

def bsample(list_of_data):
    n = len(list_of_data[0])
    bs_indices = np.random.choice(n, size=n)
    bs_data = [data[bs_indices] for data in list_of_data]
    return bs_data

def get_best_action(x, param, bandit):
    max_value = -np.inf
    a_max = None
    for a in bandit.action_space:    
        action_value = bandit.feature_vector(x, a) @ param
        if action_value > max_value:
            max_value = action_value
            a_max = a
    return a_max

def evaluate_param_est(param_est, param, bandit, x_vals=None):
    if x_vals is None:
        x_vals = np.array([bandit.x_dist() for _ in range(500)])
    a = [get_best_action(x, param_est, bandit) for x in x_vals]
    
    average_value = np.mean(bandit.feature_vector(x_vals, a) @ param)
    return average_value
        
#%% Algorithms

def alg_eps_greedy(bandit, alpha, epsilon=0.1):
    X_new = bandit.sample()
    
    if random.random() < epsilon:
        return np.random.choice(bandit.action_space)
    
    beta_hat = linear_regression(np.array(bandit.phi_XA), np.array(bandit.R))
    
    a = get_best_action(X_new, beta_hat, bandit)
    return a

def alg_unsafe_ts(bandit, alpha, epsilon=0.1):
    X_new = bandit.sample()

    if random.random() < epsilon:
        return np.random.choice(bandit.action_space)
    
    phi_XA_bs, R_bs = bsample([np.array(bandit.phi_XA), np.array(bandit.R)])    
    beta_hat_bs = linear_regression(phi_XA_bs, R_bs)
    
    a = get_best_action(X_new, beta_hat_bs, bandit)
    return a
    
def safe_ts_preprocess_test(phi_XA, X, S, bandit):
    n, p = phi_XA.shape
    phi_outer = [np.outer(phi, phi) for phi in phi_XA]
    cov_H = np.mean(phi_outer, axis=0) + np.identity(p) * 0.01   

    beta_hat_S = np.linalg.solve(cov_H, phi_XA.T @ S/n)
    residuals_sq = (phi_XA @ beta_hat_S - S)**2
    
    cov_H_inv = np.linalg.inv(cov_H)
    cov_G = np.mean(
        [phi_sq * resid_sq for (phi_sq, resid_sq) in zip(phi_outer, residuals_sq)],
        axis=0
    )
    cov = cov_H_inv @ cov_G @ cov_H_inv
        
    test_vars = {
        "beta_hat_S": beta_hat_S,
        "cov": cov,
    }
    
    return test_vars

def test_safety(x, a, a_baseline, beta_hat_S, cov, alpha, phi, n):
    """ 
    Runs multiple tests if multiple values of beta_hat_S are passed. 
    NOTE: does not resample covariance matrix, which could be problematic.
    """    
    phi_diff = phi(x, a) - phi(x, a_baseline)

    std_err = np.sqrt(phi_diff.T @ cov @ phi_diff)
    critical_value = norm.ppf(1-alpha) * std_err / np.sqrt(n)
    
    test_stats = beta_hat_S @ phi_diff
    
    test_results = test_stats > critical_value
    info = {"phi_diff" : phi_diff}
    return test_results, info

def alg_safe_ts(bandit, alpha, baseline_policy, epsilon=0.1):
    X_new = bandit.sample()

    a_baseline = baseline_policy(X_new)

    if random.random() < epsilon:
        return np.random.choice(bandit.action_space)
    
    X = np.array(bandit.X)
    phi_XA = np.array(bandit.phi_XA)
    R = np.array(bandit.R)
    S = np.array(bandit.S)
    
    phi_XA_1, X_1, R_1, S_1 = phi_XA[::2], X[::2], R[::2], S[::2]
    phi_XA_2, X_2, R_2, S_2 = phi_XA[1::2], X[1::2], R[1::2], S[1::2]
    
    # ESTIMATE SURROGATE OBJECTIVE ON SAMPLE 1
    test_vars_1 = safe_ts_preprocess_test(phi_XA_1, X_1, S_1, bandit)
    
    beta_hat_R_1 = linear_regression(phi_XA_1, R_1)
        
    num_bs_samples = 50
    beta_hats_S_bs = np.zeros((num_bs_samples, len(test_vars_1["beta_hat_S"])))
    for bs_idx in range(num_bs_samples):
        phi_XA_1_bs, S_1_bs = bsample([phi_XA_1, S_1])
        beta_hats_S_bs[bs_idx,:] = linear_regression(phi_XA_1_bs, S_1_bs)
    
    def objective(a):
        test_results, info = test_safety(
            x = X_new,
            a = a,
            a_baseline = a_baseline,
            beta_hat_S = test_vars_1["beta_hat_S"],
            cov = test_vars_1["cov"],
            alpha = alpha,
            phi = bandit.feature_vector,
            n = len(X_1)
        )
        estimated_pass_prob = np.mean(test_results)
        estimated_improvement = info["phi_diff"] @ beta_hat_R_1
        value = -estimated_improvement * estimated_pass_prob
        return value
    
    objective_vals = [objective(a) for a in bandit.action_space]
    argmax = np.argmax(objective_vals)
    a_hat = bandit.action_space[argmax]
    
    # TEST SELECTION ON SAMPLE 2
    test_vars_2 = safe_ts_preprocess_test(phi_XA_2, X_2, S_2, bandit)
        
    test_result, _ = test_safety(
        x = X_new,
        a = a_hat,
        a_baseline = a_baseline,
        beta_hat_S = test_vars_2["beta_hat_S"],
        cov = test_vars_2["cov"],
        alpha = alpha,
        phi = bandit.feature_vector,
        n = len(X_2)
    )

    if test_result:
        return a_hat
    else:
        return a_baseline

def get_best_average_safe_reward(bandit, baseline_policy, num_samples=1000):
    X = np.array([bandit.x_dist() for _ in range(num_samples)])
        
    phi_baseline = bandit.feature_vector(X, baseline_policy(X))
    safety_baseline = phi_baseline @ bandit.safety_param
    
    # Figure out the best reward obtainable by oracle safe policy
    num_actions = len(bandit.action_space)
    action_reward_masked_by_safety = np.zeros((num_samples, num_actions))
    
    for a_idx, a in enumerate(bandit.action_space):
        phi_X_a = bandit.feature_vector(X, a)
        safety = phi_X_a @ bandit.safety_param
        is_safe = safety >= safety_baseline
        
        reward = phi_X_a @ bandit.reward_param

        reward_masked_by_safety = np.array([
             r if safe else -np.inf for r, safe in zip(reward, is_safe)
        ])
        
        action_reward_masked_by_safety[:, a_idx] = reward_masked_by_safety
    
    best_safe_rewards = np.max(action_reward_masked_by_safety, axis=1)
    assert len(best_safe_rewards) == num_samples
    best_average_safe_reward = np.mean(best_safe_rewards)
    return best_average_safe_reward

def evaluate(
        bandit_constructor,
        action_selection, # TODO: rename to clarify difference in type to baseline?
        baseline_policy,
        num_random_timesteps,
        num_alg_timesteps,
        num_runs,
        alpha, 
        print_time=True
    ):
    start_time = time.time()

    total_timesteps = num_random_timesteps + num_alg_timesteps
    results = {
        "alg_name" : action_selection.__name__,
        "num_random_timesteps" : num_random_timesteps,
        "num_alg_timesteps" : num_alg_timesteps,
        "mean_reward" : np.zeros((num_runs, total_timesteps)),
        "mean_safety" : np.zeros((num_runs, total_timesteps)),
        "safety_ind" : np.zeros((num_runs, total_timesteps), dtype=bool),
        "alpha" : alpha
    }

    for run_idx in range(num_runs):
        bandit = bandit_constructor()
        for _ in range(num_random_timesteps):
            bandit.sample() # Note: required to step bandit forward
            a = np.random.choice(bandit.action_space)
            bandit.act(a)

        for t in range(num_alg_timesteps):
            a = action_selection(bandit, alpha=alpha)
            bandit.act(a)
            
        results["mean_reward"][run_idx] = bandit.R_mean
        results["mean_safety"][run_idx] = bandit.S_mean
        
        # Evaluate safety
        X = np.array(bandit.X)
        phi_baseline = bandit.feature_vector(X, baseline_policy(X))
        safety_baseline = phi_baseline @ bandit.safety_param
        results["safety_ind"][run_idx] = bandit.S_mean >= safety_baseline
            
    duration = (time.time() - start_time)/60
    if print_time:
        print(
            f"Evaluated {action_selection.__name__} {num_runs}"
            f" times in {duration:0.02f} minutes."
        )
    return results

def plot(results, axes):
    ax_reward, ax_safety, ax_safety_ind = axes
    
    ax_reward.plot(results["mean_reward"].mean(axis=0))
    ax_reward.set_title("Mean rewards")
    
    ax_safety.plot(results["mean_safety"].mean(axis=0))
    ax_safety.set_title("Mean safety")
    
    ax_safety_ind.plot(results["safety_ind"].mean(axis=0), label=results["alg_name"])
    ax_safety_ind.set_title("Safety indicator")     
    ax_safety_ind.axhline(1-results["alpha"], ls="--", c="gray", lw=1)
    
    ax_reward.set_xlabel("Timestep")

    # Label random timesteps
    for ax in axes:
        ax.axvline(
            x=results["num_random_timesteps"], 
            alpha=0.5, c="grey", lw=1, ymax=0.02
        )
    return axes

def plot_many(results_list, best_safe_reward=None):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(12,7))
    
    for results in results_list:
        plot(results, axes)
        
    if best_safe_reward is not None:
        axes[0].axhline(best_safe_reward, ls=":", c="black", lw=1)
        
    plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    plt.show()

#%% Evaluation
results_list = []

bandit_constructor = BanditEnv.get_polynomial_bandit
baseline_policy = lambda x : 0

safe_ts = wrapped_partial(alg_safe_ts, baseline_policy=baseline_policy)

for action_selection in [alg_eps_greedy, alg_unsafe_ts, safe_ts]:
    results = evaluate(
        bandit_constructor,
        action_selection,
        baseline_policy = baseline_policy,
        num_random_timesteps=10,
        num_alg_timesteps=60,
        num_runs=2000,
        alpha=0.1,    
    )
    results_list.append(results)

best_safe_reward = get_best_average_safe_reward(bandit_constructor(), baseline_policy)


#%%
plot_many(results_list, best_safe_reward)