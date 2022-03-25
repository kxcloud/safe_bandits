from functools import partial, update_wrapper

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def linear_regression(x_mat, y, penalty=1e-8):
    return np.linalg.solve(x_mat.T @ x_mat + penalty * np.identity(x_mat.shape[1]), x_mat.T @ y)

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def get_normal_inv_cdf(loc, scale):
    f = lambda t: scipy.stats.norm.ppf(t, loc=loc, scale=scale)
    return f

def get_binomial_inv_cdf(n, p):
    f = lambda t: scipy.stats.binom.ppf(t, n=n, p=p)
    return f

def correlated_sampler(inv_cdf_1, inv_cdf_2, dependence):
    assert -1 <= dependence <= 1, "dependence must be between -1 and 1."
    
    abs_dependence = np.abs(dependence)
    U = np.random.uniform(0, 1, size=2)
    
    # Handle edge cases separately, because Beta distribution
    # doesn't accept zero-valued parameters.
    W1 = U[0]
    if dependence == 1:  
        W2 = U[0]
    elif dependence == 0:
        W2 = U[1]
    elif dependence == -1:
        W2 = 1 - U[0]
    else:
        Z = np.random.beta(a=abs_dependence, b=1-abs_dependence)
        W2 = Z*(U[0] if dependence > 0 else 1-U[0])+(1-Z)*U[1]
    
    X1 = inv_cdf_1(W1)
    X2 = inv_cdf_2(W2)
    return X1, X2
    
if __name__ == "__main__":
    n = 100
    num_dependence_vals = 7
    dependence_vals = np.linspace(-1, 1, num_dependence_vals)
    fig, axes = plt.subplots(nrows=num_dependence_vals, ncols=1, sharex=True, figsize=(4,20))
    for dependence, ax in zip(dependence_vals, axes):
        data = np.empty((n, 2))
        for i in range(n):    
            data[i,:] = correlated_sampler(get_normal_inv_cdf(0,1), get_binomial_inv_cdf(5, 0.5), dependence)
        ax.scatter(data[:,0], data[:,1])
        ax.set_title(f"dependence={dependence:0.02f}")
    plt.show()