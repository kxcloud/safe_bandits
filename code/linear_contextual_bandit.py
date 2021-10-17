import time

import numpy as np
import matplotlib.pyplot as plt

n_grid = 70
X_lin = np.linspace(0, 1, n_grid)

b = np.array([0, 1, 0, -1])
c = 0*np.array([-1, 0.5, 2, 0])/3
theta_true = np.concatenate((b, c))

def get_feature_vector(x, a):
    x_vec = np.array([np.ones_like(x), x, x**2, x**3])
    combined = np.concatenate((x_vec, a*x_vec)).T
    return combined

def linear_regression(x_mat, y, penalty=0.01):
    return np.linalg.solve(x_mat.T @ x_mat + penalty * np.identity(x_mat.shape[1]), x_mat.T @ y)


# DATA GENERATING PROCESS
def sample(x, reward_fn, safety_fn, reward_noise=1):
    R = np.random.normal(reward_fn(x), scale=reward_noise)
    S = np.random.binomial(1, safety_fn(x))
    return R, S