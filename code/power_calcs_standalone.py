import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t

"""
    Simulate Pretest vs S-P-T on standard bandit data without actually
    generating the data in the context of a linear model.
    
    The bandit:
        Action 0 is the baseline, with known safety level 0.
        Action 1 has known infinite reward, safety "effect_size"
        Actions 2-"num_arms" have known low reward, safety -infinity
    
    As a result we only have to simulate what happens with Action 1, since by 
    construction actions 2 onward will never pass the safety test and action 1
    is guaranteed to be selected if it passes.
"""

def multiple_pretest(num_arms, samples_per_arm, effect_size, alpha=0.1, std_dev=1):
    alpha_corrected = alpha/num_arms
    z_score = norm.ppf(1-alpha_corrected)
    
    action_1_S = np.random.normal(effect_size, std_dev, size=samples_per_arm)
    
    mu_hat = np.mean(action_1_S)
    std_err = np.std(action_1_S) / np.sqrt(samples_per_arm)
    
    passed_test = mu_hat / std_err > z_score
    return passed_test
        
def propose_test(num_arms, samples_per_arm, effect_size, alpha=0.1, std_dev=1):
    propose_set_size = np.random.binomial(samples_per_arm, p=0.5)
    test_set_size = samples_per_arm - propose_set_size
    
    if propose_set_size <= 1 or test_set_size <= 1:
        # Not enough data
        return 0

    action_1_S_propose = np.random.normal(effect_size, std_dev, size=propose_set_size)
    if max(action_1_S_propose) < effect_size:
        # Propose step fails to propose action 1. 
        # Reasoning: if max(action_1_S_propose) > effect_size, then there is 
        # a bootstrap sample where each data point is equal to 
        # max(action_1_S_propose), which would then have 0 standard error. So
        # this would produce a passing safety test, which means the overall
        # bootstrap safety test pass probability must be greater than 0. 
        # Since the reward for action 1 is infinite, a nonzero probability
        # means it must be proposed.
        return 0
    
    action_1_S_test = np.random.normal(effect_size, std_dev, size=test_set_size)
    
    z_score = norm.ppf(1-alpha)
    mu_hat = np.mean(action_1_S_test)
    std_err = np.std(action_1_S_test) / np.sqrt(test_set_size)
        
    passed_test = mu_hat / std_err > z_score
    return passed_test

start_time = time.time()
results = []

N_RUNS = 2000
records = []
for num_arms in range(5, 200, 5):
    for samples_per_arm in range(6, 40, 2):
        for effect_size in np.linspace(0.1, 2, 8):
            num_passed_pretest = 0
            num_passed_spt = 0
            for _ in range(N_RUNS):
                num_passed_pretest += multiple_pretest(num_arms, samples_per_arm, effect_size)
                num_passed_spt += propose_test(num_arms, samples_per_arm, effect_size)
            pass_pct_pretest = num_passed_pretest / N_RUNS
            pass_pct_spt = num_passed_spt / N_RUNS
            record = (num_arms, samples_per_arm, effect_size, pass_pct_pretest, pass_pct_spt)
            records.append(record)

duration = (time.time() - start_time)/60
print(f"Runtime: {duration:0.02f} minutes.")

colnames = ["num_arms", "samples_per_arm", "effect_size", "pass_pct_pretest", "pass_pct_spt"]
res = pd.DataFrame.from_records(records, columns=colnames)
res.to_csv("G:\\System\\Documents\\ACADEMIC\\safe_bandits\\data\\fwer_debugging\\basic_power_calcs.csv", index=False)


#%% Plot

def qplot(x_var, curves_var, fixed_var, fixed_value, figsize=(8,5)):
    subset1 = res[res[fixed_var] == fixed_value]
    
    curve_values = subset1[curves_var].unique()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for idx, curve_value in enumerate(curve_values):
        color = f"C{idx}"
        subset2 = subset1[subset1[curves_var] == curve_value]
        
        x = subset2[x_var]
        y_pretest = subset2["pass_pct_pretest"]
        y_spt = subset2["pass_pct_spt"]
        
        ax.plot(x, y_pretest, c=color, label=curve_value)
        ax.plot(x, y_spt, c=color, ls="--")
    
    ax.set_xlabel(x_var)
    ax.set_ylabel("Power")
    ax.legend(title=curves_var)
    
    title = "Power of bandit testing algs in extreme case"
    subtitle1 = f"({fixed_var}={fixed_value})"
    subtitle2 = "Solid line: Pretest, dashed: Split-Propose-Test"
    
    ax.set_title("\n".join([title, subtitle1, subtitle2]))
    
    return ax

qplot("num_arms", "effect_size", "samples_per_arm", 10)

    

