import time

import numpy as np
import matplotlib.pyplot as plt

n_grid = 70
X_lin = np.linspace(0, 1, n_grid)


# KERNEL SMOOTHING
def box_car(t):
    return (np.abs(t) <= 0.5).astype(int)

def gaussian(t):
    return np.exp(-t**2/2)/np.sqrt(2 * np.pi)

def epanechnikov(t):
    return (np.abs(t) <= 1)*3/4*(1-t**2)

def get_kernel_estimator(X, y, kernel, h):
    def f(x):
        weighted_outcomes = 0
        total_weight = 0
        for x_i, y_i in zip(X,y):
            weight = kernel((x-x_i)/h)
            weighted_outcomes += weight * y_i
            total_weight += weight
        return weighted_outcomes/total_weight
    return f

def get_linear_smoother(X, kernel, h):
    """ Get the matrix L such that Y_hat = L Y. """
    n = len(X)
    
    L = np.empty((n,n), dtype=float)
    for i in range(n):
        L[i,i] = kernel(0)
        for j in range(i):
            L[i,j] = kernel((X[i]-X[j])/h)
            L[j,i] = L[i,j]
    
    for i in range(n):
        L[i, :] = L[i, :] / L[i, :].sum()
    return L

def get_confidence_interval(m_hat, error_bs, alpha=0.05):
    error_quantiles = np.quantile(error_bs,[alpha,1-alpha],axis=0)
    conf_interval = m_hat - error_quantiles
    return conf_interval

def apply_local_bootstrap(X, y, kernel, h, num_bs_samples):
    f_hat = get_kernel_estimator(X, y, kernel, h) 
    m_hat = f_hat(X_lin)
    
    n=len(X)
    KDE = get_linear_smoother(X, kernel, h)
            
    y_bs = np.empty((num_bs_samples, n))
    
    # For each X_i, sample all the bootstrap samples Y_i^*.
    for i in range(n):
        y_bs[:,i] = np.random.choice(y, size=num_bs_samples, p=KDE[i,:])

    error_bs = np.zeros((num_bs_samples, n_grid))  
    for b in range(num_bs_samples):
        f_hat_bs = get_kernel_estimator(X, y_bs[b,:], kernel, h)
        error_bs[b,:] = f_hat_bs(X_lin) - m_hat

    return m_hat, error_bs

def argmax_with_mask(values, mask):
    max_value = -np.inf
    argmax = None
    for idx, (value, is_valid) in enumerate(zip(values, mask)):
        if is_valid and value > max_value:
            max_value = value
            argmax = idx
    return argmax


# SAFE BANDIT EXPLORATION
h = 0.5
kernel = lambda t: 0.1 * gaussian(t) + epanechnikov(t)
s_baseline = 0.6
s_alpha = 0.05

fig5, ax5 = plt.subplots(figsize=(3,2))
ax5.plot(X_lin, kernel((X_lin-0.5)/h)/h, c="gray")
ax5.set_title(f"Kernel (h={h:0.03f})")
plt.show()

def get_naive_ts_action(X, R, S, x_default=0, s_baseline=s_baseline):
    """ Ignore safety """
    r_hat, error_bs = apply_local_bootstrap(X, R, kernel, h, 1)
    r_hat_bs = r_hat+error_bs[0]
    idx_max = np.argmax(r_hat_bs)
    X_hat = X_lin[idx_max]
    return X_hat

def get_naive_safe_ts_action(X, R, S, x_default=0, s_baseline=s_baseline):
    """ Select a policy, ignoring safety, then test it for safety. """
    X_1, R_1 = X[::2], R[::2]
    X_2, S_2 = X[1::2], S[1::2]
    
    r_hat, error_bs = apply_local_bootstrap(X_1, R_1, kernel, h, 1)
    r_hat_bs = r_hat+error_bs[0]
    idx_max = np.argmax(r_hat_bs)
    X_hat = X_lin[idx_max]
    
    s_hat, error_bs = apply_local_bootstrap(X_2, S_2, kernel, h, 50)
    m_hat_at_X_hat_bs = s_hat[idx_max] + error_bs[:, idx_max]
    lower_ci = np.quantile(m_hat_at_X_hat_bs, s_alpha)
    
    if lower_ci > s_baseline:
        return X_hat
    else:
        return x_default

def get_naive_safe_set_ts_action(X, R, S, x_default=0, s_baseline=s_baseline):
    """ Select a TS policy from a safe set. """
    X_1, _, S_1 = X[::2], R[::2], S[::2]
    X_2, R_2, _ = X[1::2], R[1::2], S[1::2]
    
    # Get pointwise safe region from first half of data
    s_hat, error_bs = apply_local_bootstrap(X_1, S_1, kernel, h, 50)
    m_hat = s_hat + error_bs
    lower_ci = np.quantile(m_hat, s_alpha, axis=0)
    pointwise_safe_values = lower_ci > s_baseline
        
    if pointwise_safe_values.sum() == 0:
        # print("Note: safe region was empty; returning default value.")
        return x_default
        
    # Run TS over this region
    r_hat, error_bs = apply_local_bootstrap(X_2, R_2, kernel, h, 1)
    r_hat_bs = r_hat + error_bs[0]
    
    idx_max = argmax_with_mask(r_hat_bs, pointwise_safe_values)
    X_hat = X_lin[idx_max]
    return X_hat
    

def get_safe_ts_action(X, R, S, x_default=0, s_baseline=s_baseline, plot_ax=None):
    X_1, R_1, S_1 = X[::2], R[::2], S[::2]
    X_2, _, S_2 = X[1::2], R[1::2], S[1::2]
    
    r_hat, error_bs = apply_local_bootstrap(X_1, R_1, kernel, h, 1)
    r_hat_bs = r_hat + error_bs[0]
    arg_of_default = np.argmin(np.abs(X_lin - x_default))
    reward_diff = r_hat_bs - r_hat_bs[arg_of_default]
    
    bs_samples = 30
    bs_test_results = np.zeros((bs_samples, n_grid))
    for idx in range(bs_samples):
        bs_indices = np.random.choice(len(X_1), size=len(X_1))
        X_1_bs = X_1[bs_indices]
        S_1_bs = S_1[bs_indices]
        s_hat_bs, error_bs_bs = apply_local_bootstrap(X_1_bs, S_1_bs, kernel, h, 30)      
        lower_ci_bs = np.quantile(s_hat_bs + error_bs_bs, 0.1, axis=0)
        bs_test_results[idx,:] = lower_ci_bs > s_baseline
    estimated_pass_prob = bs_test_results.mean(axis=0)
    
    if plot_ax is not None:
        plot_ax.plot(X_lin, reward_diff, label="estimated improvement", c="red")
        plot_ax.plot(X_lin, estimated_pass_prob, label="estimated pass probability",ls="--", c="orange")
        plot_ax.plot(X_lin, reward_diff * estimated_pass_prob, label="objective (the product)", lw=2, c="black")
    
    idx_max = np.argmax(reward_diff * estimated_pass_prob)
    X_hat = X_lin[idx_max]
    
    s_hat, error_bs = apply_local_bootstrap(X_2, S_2, kernel, h, 50)
    m_hat_at_X_hat_bs = s_hat[idx_max] + error_bs[:, idx_max]
    lower_ci = np.quantile(m_hat_at_X_hat_bs, 0.1)

    if lower_ci > s_baseline:
        return X_hat
    else:
        return x_default


# DATA GENERATING PROCESS
def sample(x, reward_fn, safety_fn, reward_noise=1):
    R = np.random.normal(reward_fn(x), scale=reward_noise)
    S = np.random.binomial(1, safety_fn(x))
    return R, S

def run_alg(reward_fn, safety_fn, action_selection, num_timesteps, X_0=None):
    if X_0 is None:
        X_0 = np.linspace(0, 1, num=10)
    n_0 = len(X_0)
    n_total = n_0 + num_timesteps
    X_all = np.empty(n_total)
    R_all = np.empty(n_total)
    S_all = np.empty(n_total)
    X_all[:n_0] = X_0
    R_all[:n_0], S_all[:n_0] = sample(X_0, reward_fn, safety_fn) 
    
    for t in range(num_timesteps):
        idx = n_0 + t
        X_hat = action_selection(X_all[:idx], R_all[:idx], S_all[:idx])
        R, S = sample(X_hat, reward_fn, safety_fn)
        X_all[idx], R_all[idx], S_all[idx] = X_hat, R, S

    return X_all, R_all, S_all

# RESULTS
r = lambda t: t
s = lambda t: 1-t
action_selections = {
    "TS (unsafe)" : get_naive_ts_action,
    "Naive single-test TS" : get_naive_safe_ts_action,
    "Naive safe set TS (unsafe)" : get_naive_safe_set_ts_action,
    "Optimal single-test TS" : get_safe_ts_action
}
num_timesteps = 50
x_default = 0

n_0 = 40 # Number of initial data points
X_0 = np.linspace(0, 1, n_0)**2

num_runs = 300

results = {}
total_duration = 0
for action_select_label, action_selection in action_selections.items():
    X = np.empty((num_runs, n_0+num_timesteps))
    R = np.empty((num_runs, n_0+num_timesteps))
    S = np.empty((num_runs, n_0+num_timesteps))
    print(f"Running {action_select_label}...")
    start_time = time.time()
    
    for run_idx in range(num_runs):
        if run_idx % np.ceil(num_runs/10) == 0:
            print(run_idx, end=".")
        X[run_idx], R[run_idx], S[run_idx] = run_alg(r, s, action_selection, num_timesteps, X_0)
    
    results[action_select_label] = (X, R, S)
    duration = (time.time() - start_time)/60
    total_duration += duration
    print(f"\n... {duration:0.02f} minutes for {num_runs}.")
print(f"Total duration: {total_duration:0.02f} minutes for {num_runs} runs per setting.")

#%% PLOTS
idx_of_datasets_to_plot = range(4)

fig, (ax_avg, ax_safety) = plt.subplots(
    ncols=2, sharex=True, sharey=True, figsize=(11,5)
)

fig2, axes = plt.subplots(
    nrows=1, ncols=len(idx_of_datasets_to_plot), sharex=True, sharey=True,
    figsize=(13,4)
)
fig3, axes3 = plt.subplots(
    nrows=2, ncols=len(idx_of_datasets_to_plot), sharex=True, figsize=(13,8)
)

fig4, axes4 = plt.subplots(
    nrows=1, ncols=len(idx_of_datasets_to_plot), sharex=True, sharey=True,
    figsize=(13,4)
)

for action_select_label, action_selection in action_selections.items():
    X, R, S = results[action_select_label]
    
    X_mean = X[:,-num_timesteps:].mean(axis=0)
    ax_avg.plot(range(num_timesteps), X_mean, label = action_select_label)
    
    safety_pct = (S[:,-num_timesteps:] > s_baseline).mean(axis=0)
    ax_safety.plot(range(num_timesteps), safety_pct)
    
    for plot_idx, dataset_idx in enumerate(idx_of_datasets_to_plot):
        ax = axes[plot_idx]
        ax.scatter(range(num_timesteps), X[dataset_idx,-num_timesteps:], label = action_select_label, s=20)
        ax.hlines(1-s_baseline, 0, num_timesteps, ls="--", lw=1, color="red")
        ax.set_title(f"Dataset {dataset_idx}")
        
    
if "Optimal single-test TS" in results:
    X_safe, R_safe, S_safe = results["Optimal single-test TS"]
    for plot_idx, dataset_idx in enumerate(idx_of_datasets_to_plot):
        X_i, R_i, S_i = X_safe[dataset_idx], R_safe[dataset_idx], S_safe[dataset_idx]
        
        r_hat = get_kernel_estimator(X_i, R_i, kernel, h)(X_lin)
        s_hat = get_kernel_estimator(X_i, S_i, kernel, h)(X_lin)
        
        ax_up = axes3[0,plot_idx]
        ax_down = axes3[1,plot_idx]
        
        point_size = 5
        line_width = 2
        ax_up.set_title(f"Dataset {dataset_idx}")
        
        ax_up.scatter(X[dataset_idx], R[dataset_idx], s=point_size, c="C2")
        ax_up.plot(X_lin, r_hat, c="C2", lw=line_width)
        ax_up.plot(X_lin, r(X_lin), c="gray", ls="--")
        
        ax_down.scatter(X[dataset_idx], S[dataset_idx], s=point_size, c="C2")
        ax_down.plot(X_lin, s_hat, lw=line_width,label ="Optimal single-test TS", c="C2")
        ax_down.plot(X_lin, s(X_lin), c="gray", ls="--", label="Truth")
        
        get_safe_ts_action(X_i,R_i,S_i, plot_ax=axes4[plot_idx])
        axes4[plot_idx].set_title(f"Dataset {dataset_idx}")
    axes4[-1].legend()  
    
ax_avg.hlines(1-s_baseline, 0, num_timesteps, ls="--", lw=1, color="red")#, label="Safety threshold")
fig.legend(loc=7)
ax_avg.set_xlabel("Timestep")
ax_avg.set_ylabel("Mean Action selected")
ax_safety.set_ylabel("% Safe")
ax_safety.hlines(1-s_alpha, 0, num_timesteps, ls="--", lw=1, color="gray")

axes[0].set_ylabel("Action selected")
axes[0].set_xlabel("Timestep")
axes[-1].legend()

axes3[0,0].set_ylabel("Reward")
axes3[1,0].set_ylabel("Safety")
axes3[-1,-1].legend()

plt.show()