import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Settings
means=[1, 1.5, 2]
prob_negative=[0, 0.15, 0.4]

safety_tol=0.3
reward_param = np.array(means)
safety_param = 1 - np.array(prob_negative)

assert np.all(reward_param >= 0), "Means must be nonnegative"

lower_bounds = 2*(1-safety_param)*reward_param / (1-2*safety_param)
upper_bounds = 2*reward_param - lower_bounds

#%% Plot
action_space = range(len(reward_param))
fig, ax = plt.subplots(figsize=(len(action_space), 3))
ax.axhline(0, ls="--", c="black", lw=1)

lw = 3
for a in action_space:
    ax.plot([a, a], [lower_bounds[a], upper_bounds[a]], c="C0", lw=lw, solid_capstyle="butt")
    ax.plot([a, a], [lower_bounds[a], 0], c="red", lw=lw, solid_capstyle="butt")
    length = upper_bounds[a] - lower_bounds[a]
    tolerance = length*min(prob_negative[a], safety_tol)
    ax.plot([a,a], [-tolerance, 0], c="gold", lw=lw, solid_capstyle="butt")
    ax.scatter(a, means[a], c="C0", marker="o", s=40, zorder=10)
ax.set_xticks(action_space)
ax.set_xlabel("Action")    
ax.set_ylabel("Reward")
ax.set_title("Uniform armed bandit")

custom_lines = [Line2D([0], [0], color="C0", lw=lw),
                Line2D([0], [0], color="gold", lw=lw),
                Line2D([0], [0], color="red", lw=lw)]
ax.legend(custom_lines, ["Safe", "Unsafe (tolerable)", "Unsafe"])