import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from scipy.stats import norm, t

matplotlib.rcParams['figure.dpi'] = 100

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

N_RUNS = 50000
records = []
for num_arms in range(1,50):
    for samples_per_arm in [20]:
        for effect_size in [0.5]:
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

filepath = "G:\\System\\Documents\\ACADEMIC\\safe_bandits\\data\\fwer_debugging\\basic_power_calcs_2.csv"
res.to_csv(filepath, index=False)


#%% Plot

res = pd.read_csv(filepath)

def qplot(
        df, x_var, curves_var, fixed_var, fixed_value, 
        title=None,
        figsize=(8,5), 
        ax=None,
        show_legends=True,
        ):
    subset1 = df[df[fixed_var] == fixed_value]
    
    curve_values = subset1[curves_var].unique()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
            
    for idx, curve_value in enumerate(curve_values):
        color = f"C{idx}"
        subset2 = subset1[subset1[curves_var] == curve_value]
        
        x = subset2[x_var]
        y_pretest = subset2["pass_pct_pretest"]
        y_spt = subset2["pass_pct_spt"]
        
        ax.plot(x, y_pretest, c=color, label=f"{curve_value:0.01f}")
        ax.plot(x, y_spt, c=color, ls="--")
    
    ax.set_xlabel(x_var)
    ax.set_ylabel("Power")
    ax.set_ylim((0,1))
    
    if show_legends:
        ax.legend(title=curves_var)
        
        legend_types = [
            Line2D([0], [0], color="gray", label='Pretest All'),
            Line2D([0], [0], color='gray', ls="--", label='SPT'),
        ]
            
        ax_ghost = ax.twinx()
        ax_ghost.legend(handles=legend_types, loc="upper left")
        ax_ghost.set_yticklabels([])
            
    subtitle1 = f"({fixed_var}={fixed_value})"
    
    if title is not None:
        ax.set_title("\n".join([title, subtitle1]))
    else:
        ax.set_title(subtitle1)
    
    return ax

fig, ax = plt.subplots(figsize=(4,3.6))
ax.plot(res["num_arms"], res["pass_pct_pretest"], label="Pretest all", ls=":", lw=2)
ax.plot(res["num_arms"], res["pass_pct_pretest"], ls="-", lw=2, alpha=0.2, c="C0")
ax.plot(res["num_arms"], res["pass_pct_spt"], label="SPT", ls="-", lw=2)
ax.set_xlabel("Number of arms")
ax.set_ylabel("Power")
ax.legend()

title = "Power of bandit testing algorithms\nto detect single good arm"
subtitle = "(Samples per arm: 20, effect size: 0.5)"
ax.set_title(title+"\n"+subtitle, fontsize=10)
plt.tight_layout()