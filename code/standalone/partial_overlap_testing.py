import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import softmax 

def choose_test_levels(Y_0, alpha, overlap):
    """ 
    Interpolates between (i) testing best action only with full sample split,
    and (ii) testing all actions evenly with Bonferroni correction.
    """
    d = Y_0.shape[1]
    Y_0_bar = Y_0.mean(axis=0)
    if overlap == 0:
        test_levels = np.zeros(d)
        test_levels[np.argmax(Y_0_bar)] = alpha
    else:
        temperature = np.log(1/overlap)
        test_levels = softmax(Y_0_bar * temperature) * alpha
    return test_levels

def test_hypotheses(Y_1, test_levels):
    results = []
    sqrt_n = np.sqrt(Y_1.shape[0])
    for k, alpha_k in enumerate(test_levels):
        data = Y_1[:, k]
        Z = data.mean() / data.std()
        result = sqrt_n * Z > norm.ppf(1-alpha_k)
        results.append(result)
    return results

def test(Y_0, Y_1, alpha, overlap):
    levels = choose_test_levels(Y_0, alpha, overlap)
    results = test_hypotheses(Y_1, levels)
    return results

num_runs = 4000
n = 30
d = 10
alpha = 0.1
effect_size = -0.1
overlaps = np.linspace(0, 1, 30)

results = { overlap : np.zeros(num_runs) for overlap in overlaps }

start_time = time.time()
for run_idx in range(num_runs):
    Y = np.random.normal(loc=effect_size, size=(n, d))
    random_seeds = np.random.uniform(size=n)
    
    for overlap in overlaps:
        indices_0 = np.nonzero(random_seeds <= 0.5 + overlap/2)[0]
        indices_1 = np.nonzero(random_seeds > 0.5 - overlap/2)[0]
        Y_0 = Y[indices_0]
        Y_1 = Y[indices_1]
        
        test_results = test(Y_0, Y_1, alpha, overlap)
        type_1_error = np.max(test_results)
        results[overlap][run_idx] = type_1_error
minutes = (time.time() - start_time)/60
print(f"Ran in {minutes:0.2f} minutes.")

type_1_error_rates = np.array([results[overlap].mean() for overlap in overlaps])
type_1_error_rate_std = np.array([results[overlap].std() for overlap in overlaps])
ci_width = type_1_error_rate_std * 1.96 / np.sqrt(num_runs)

fig, ax = plt.subplots()
ax.plot(overlaps, type_1_error_rates)
ax.plot(overlaps, type_1_error_rates+ci_width, ls="--", c="C0", lw=1)
ax.plot(overlaps, type_1_error_rates-ci_width, ls="--", c="C0", lw=1)
ax.set_xlabel("Sample overlap")
ax.set_ylabel("Type I error rate")
ax.axhline(alpha, ls="--", c="black", lw=1)
