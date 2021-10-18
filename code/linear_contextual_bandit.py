import time

import numpy as np
import matplotlib.pyplot as plt

import BanditEnv

def linear_regression(x_mat, y, penalty=0.01):
    return np.linalg.solve(x_mat.T @ x_mat + penalty * np.identity(x_mat.shape[1]), x_mat.T @ y)


def bsample(list_of_data):
    n = len(list_of_data[0])
    bs_indices = np.random.choice(n, size=n)
    bs_data = [data[bs_indices] for data in list_of_data]
    return bs_data
    

def alg_naive_ts(bandit):
    X_new = bandit.sample()
    
    phi_XA_bs, R_bs = bsample([np.array(bandit.phi_XA), np.array(bandit.R)])    
    beta_hat_bs = linear_regression(phi_XA_bs, R_bs)
    
    max_value = -np.inf
    a_max = None
    for a in bandit.action_space:    
        action_value = bandit.feature_vector(X_new, a) @ beta_hat_bs
        if action_value > max_value:
            max_value = action_value
            a_max = a
    return a_max

num_random_steps = 20
num_alg_steps = 300

bandit = BanditEnv.get_polynomial_bandit()
for _ in range(num_random_steps):
    X_new = bandit.sample()
    a = np.random.choice(bandit.action_space)
    bandit.act(a)

losses = []
for _ in range(num_alg_steps):
    a = alg_naive_ts(bandit)
    bandit.act(a)
    
    beta_hat = linear_regression(np.array(bandit.phi_XA), np.array(bandit.R))

    loss = np.mean((bandit.reward_param - beta_hat)**2)
    losses.append(loss)
    
    
plt.plot(range(num_alg_steps), losses)
plt.plot(range(num_alg_steps), bandit.R_mean[num_random_steps:])
plt.show()

fig, axes = bandit.plot()