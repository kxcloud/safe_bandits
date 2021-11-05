import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.stats import norm

import BanditEnv

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

def alg_eps_greedy(bandit, epsilon=0.1):
    X_new = bandit.sample()
    
    if random.random() < epsilon:
        return np.random.choice(bandit.action_space)
    
    beta_hat = linear_regression(np.array(bandit.phi_XA), np.array(bandit.R))
    
    a = get_best_action(X_new, beta_hat, bandit)
    return a

def alg_naive_ts(bandit, epsilon=0.1):
    X_new = bandit.sample()

    if random.random() < epsilon:
        return np.random.choice(bandit.action_space)
    
    phi_XA_bs, R_bs = bsample([np.array(bandit.phi_XA), np.array(bandit.R)])    
    beta_hat_bs = linear_regression(phi_XA_bs, R_bs)
    
    a = get_best_action(X_new, beta_hat_bs, bandit)
    return a
    
def safe_ts_preprocess_test(phi_XA, X, S, baseline_policy_param, bandit):
    phi_X_allA = np.array([bandit.feature_vector(X, a) for a in bandit.action_space])
    phi_outer = [np.outer(phi, phi) for phi in phi_XA]
    cov_H = np.mean(phi_outer, axis=0)
    
    beta_hat_S = linear_regression(cov_H, S / len(S))
    residuals_sq = (phi_XA @ beta_hat_S - S)**2
    
    cov_H_inv = np.linalg.inv(cov_H)
    cov_G = np.mean(
        [phi_sq * resid_sq for (phi_sq, resid_sq) in zip(phi_outer, residuals_sq)],
        axis=0
    )
    cov = cov_H_inv @ cov_G @ cov_H_inv
    
    action_probs_base = softmax(phi_X_allA @ baseline_policy_param)
    
    test_vars = {
        "phi_X_allA": phi_X_allA,
        "beta_hat_S": beta_hat_S,
        "cov": cov,
        "action_probs_base": action_probs_base
    }
    
    return test_vars

def safe_ts_test(beta_hat_S, policy_param, test_vars, alpha):
    """ 
    Runs multiple tests if multiple values of beta_hat_S are passed. 
    NOTE: does not resample covariance matrix, which could be problematic.
    """
    action_probs = softmax(test_vars["phi_X_allA"] @ policy_param, axis=1)
    phi_X_diff = test_vars["phi_X_allA"] @ (action_probs - test_vars["action_probs_base"])

    std_err = np.sqrt(phi_X_diff.T @ test_vars["cov"] @ phi_X_diff)
    
    n = phi_X_diff.shape[0]
    critical_value = norm.ppf(1-alpha) * std_err / np.sqrt(n)
    test_stats = phi_X_diff @ beta_hat_S
    
    test_results = test_stats > critical_value
    return test_results, phi_X_diff #WARNING: returns extra data

def alg_safe_ts(bandit, epsilon, baseline_policy_param, alpha=0.1):
    X_new = bandit.sample()

    if random.random() < epsilon:
        return np.random.choice(bandit.action_space)
    
    X = np.array(bandit.X)
    phi_XA = np.array(bandit.phi_XA)
    R = np.array(bandit.R)
    S = np.array(bandit.S)
      
    phi_XA_1, X_1, R_1, S_1 = phi_XA[::2], X[::2], R[::2], S[::2]
    phi_XA_2, X_2, R_2, S_2 = phi_XA[1::2], X[1::2], R[1::2], S[1::2]
    
    # ESTIMATE SURROGATE OBJECTIVE ON SAMPLE 1
    test_vars_1 = safe_ts_preprocess_test(
        phi_XA_1, X_1, S_1, baseline_policy_param, bandit
    )
    
    beta_hat_R_1 = linear_regression(phi_XA_1.T @ phi_XA_1, R_1)
        
    num_bs_samples = 100
    beta_hats_S_bs = np.zeros((num_bs_samples, len(test_vars_1["beta_hat_S"])))
    for bs_idx in range(num_bs_samples):
        phi_XA_1_bs, S_1_bs = bsample(phi_XA_1, S_1)
        beta_hats_S_bs[bs_idx,:] = linear_regression(phi_XA_1_bs.T @ phi_XA_1_bs, S_1_bs)
    
    def objective(policy_param):
        test_results, phi_X_diff = safe_ts_test(
            beta_hats_S_bs, policy_param, test_vars_1, alpha
        )
        estimated_pass_prob = np.mean(test_results)
        estimated_improvement = phi_X_diff @ beta_hat_R_1
        penalty = np.sum(policy_param**2) * 0.01
        return estimated_improvement * estimated_pass_prob + penalty
    
    param_0 = np.zeros_like(test_vars_1["beta_hat_S"])
    result = minimize(objective, param_0, method="BFGS", options={'gtol':1e-5, 'disp':True})
    
    policy_param_hat = result.x.reshape(param_0.shape)
    
    # TEST SELECTION ON SAMPLE 2
    test_vars_2 = safe_ts_preprocess_test(
        phi_XA_2, X_2, S_2, baseline_policy_param, bandit
    )
        
    beta_hat_S_2 = test_vars_2["beta_hat_S"]
    test_result = safe_ts_test(beta_hat_S_2, policy_param_hat, test_vars_2, alpha)

    if test_result:
        action_policy_param = policy_param_hat
    else:
        action_policy_param = baseline_policy_param
    
    phi_Xnew_allA = np.array([bandit.feature_vector(X_new, a) for a in bandit.action_space]) 
    a_probs = softmax(phi_Xnew_allA @ action_policy_param)
    a = np.random.choice(bandit.action_space, p=a_probs)
    return a

num_random_steps = 2
num_alg_steps = 100

bandit = BanditEnv.get_polynomial_bandit()
for _ in range(num_random_steps):
    X_new = bandit.sample()
    a = np.random.choice(bandit.action_space)
    bandit.act(a)

losses = []
values = []
for t in range(num_alg_steps):
    a = alg_eps_greedy(bandit)
    bandit.act(a)
    
    beta_hat = linear_regression(np.array(bandit.phi_XA), np.array(bandit.R), penalty=1)
    value = evaluate_param_est(beta_hat, bandit.reward_param, bandit, x_vals=np.linspace(0,1,50))
    values.append(value)

    loss = np.mean((bandit.reward_param - beta_hat)**2)
    losses.append(loss)
    
optimal_value = evaluate_param_est(
    bandit.reward_param, 
    bandit.reward_param, 
    bandit, 
    x_vals=np.linspace(0,1,50)
)

fig1, ax = plt.subplots()
ax2 = ax.twinx()

rolling_mean_rewards = pd.Series(bandit.R_mean[num_random_steps:]).rolling(window=5).mean()
ax2.plot(range(num_alg_steps), losses, label="parameter MSE")
# ax.plot(range(num_alg_steps), rolling_mean_rewards, label="avg reward", C="C1")
ax.plot(range(num_alg_steps), values, label="value", C="C2")
ax.hlines(y=optimal_value, xmin=0, xmax=num_alg_steps, ls="--", color="gray", lw=1.5, label="optimal value")
ax.set_xlabel("timestep")
ax.set_ylabel("reward")

fig1.legend()
plt.show()

fig, axes = bandit.plot()