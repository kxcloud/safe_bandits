import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def alg_naive_safe_ts(bandit, epsilon=0.1):
    X_new = bandit.sample()

    if random.random() < epsilon:
        return np.random.choice(bandit.action_space)
    
    phi_XA_bs, R_bs = bsample([np.array(bandit.phi_XA), np.array(bandit.R)])    
    beta_hat_bs = linear_regression(phi_XA_bs, R_bs)
    
    a = get_best_action(X_new, beta_hat_bs, bandit)
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