from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning 
import _visualize_results as visualize_results
import _utils as utils

def plot_bandit(
        x,
        bandit, 
        figsize=(8.5,4), 
        title="", 
        safety_tol=0,
        reward_param=None, 
        safety_param=None, 
        baseline_policy=None,
        tested_safe_actions=None,
        axes=None,
    ):
    """
    Tool for plotting a vanilla (not contextual) bandit that shows what is safe
    and what is not.
    """
    reward_param = bandit.reward_param if reward_param is None else reward_param
    safety_param = bandit.safety_param if safety_param is None else safety_param

    if axes is None:
        _, (ax_reward, ax_safety) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    else:
        (ax_reward, ax_safety) = axes
     
    expected_reward = [
        bandit.feature_vector(x, a) @ reward_param for a in bandit.action_space
    ]
    expected_safety = [
        bandit.feature_vector(x, a) @ safety_param for a in bandit.action_space
    ]
    
    ax_safety.plot(bandit.action_space, expected_safety)
    ax_reward.set_title("Expected reward")
    ax_safety.set_title("Expected safety")
    ax_reward.set_xlabel("Action (a)")
    
    if baseline_policy is None:
        ax_reward.plot(bandit.action_space, expected_reward)
    
    # Highlight safe regions
    if baseline_policy is not None:
        ax_reward.plot(bandit.action_space, expected_reward, c="black", ls=":", label="not safe")
        a_baseline = baseline_policy(x)
        baseline_safety = bandit.feature_vector(x, a_baseline) @ safety_param
        
        safe_action_rewards = []           
        for action, reward, safety in zip(
                bandit.action_space, expected_reward, expected_safety
            ):
            if safety >= baseline_safety - safety_tol - 1e-8:
                safe_action_rewards.append(reward)
            else:
                safe_action_rewards.append(None)
        ax_reward.scatter(bandit.action_space, safe_action_rewards, c="C2", s=0.5)
        ax_reward.plot(bandit.action_space, safe_action_rewards, c="C2", lw=2.5, label="safe")
        ax_reward.legend()
        
    if tested_safe_actions is not None:
        tested_safe_action_rewards = []
        for action, reward in zip(bandit.action_space, expected_reward):
            if action in tested_safe_actions:
                tested_safe_action_rewards.append(reward)
            else:
                tested_safe_action_rewards.append(None)
        ax_reward.scatter(
            bandit.action_space, tested_safe_action_rewards, c="black", marker="D", s=65,
            zorder=9
        )
        ax_reward.scatter(
            bandit.action_space, tested_safe_action_rewards, c="gold", marker="D", label="tested safe",
            zorder=10
        )
        ax_reward.legend()
        
    plt.suptitle(title)
    return (ax_reward, ax_safety)

def plot_propose_test(info, action_space, show_test_result, title=""):
    split_objective_function = info["split_objective"]
    split = np.array([split_objective_function(a) for a in action_space])
    pass_prob = split[:,0]
    estimated_improvement = split[:,1]
    objective = pass_prob * estimated_improvement
    
    test_results = [
        pass_prob if info["safety_test"](a=a)[0] else None for a, pass_prob in zip(action_space, pass_prob)
    ]
    
    idx_max = np.argmax(objective)
    a_max = action_space[idx_max]
    
    fig, ax_prob = plt.subplots()
    ax_reward = ax_prob.twinx()
    ax_prob.plot(action_space, split[:,0], label="estimated pass probability", c="C0", ls="--")
    ax_prob.set_ylim([0,1.05])
    
    if show_test_result:
        ax_prob.scatter(action_space, test_results, c="gold")
        ax_prob.plot(action_space, test_results, label="would pass if tested", c="gold", lw=3.5)
    
    ax_reward.plot(action_space, split[:,1], label="estimated improvement", c="red")
    ax_reward.plot(action_space, objective, label="objective", lw=2, c="black")
    ax_reward.scatter(a_max, objective[idx_max], marker="*", c="black", s=100)

    ax_prob.set_title(title)
    ax_prob.set_ylabel("Probability")
    ax_reward.set_ylabel("Reward")
    ax_prob.set_xlabel("Action (a)")
    
    fig.legend(bbox_to_anchor=[1.4,0.4,0,0])
    return ax_prob, ax_reward

def get_confidence_interval(x, phi_XA, response, weights, alpha, bandit, baseline_policy):
    beta_hat, sqrt_cov = bandit_learning.estimate_safety_param_and_covariance(phi_XA, response, weights)
    
    mean = np.zeros(len(bandit.action_space))
    ci_width = np.zeros(len(bandit.action_space))
    
    for a_idx, a in enumerate(bandit.action_space):
        phi = bandit.feature_vector(x,a)
        if baseline_policy is not None:
            phi -= bandit.feature_vector(x, baseline_policy(x))
        mean[a_idx] = phi @ beta_hat
        ci_width[a_idx] = np.sqrt(np.sum((phi @ sqrt_cov)**2)/len(response)) * norm.ppf(1-alpha)
    
    return mean, ci_width

def plot_data_w_ci(x, bandit, level, safety_tol, baseline_policy, outcome=None):
    A = bandit.get_A()
    phi_XA = bandit.get_phi_XA()
    Y = bandit.get_R() if outcome == "reward" else bandit.get_S()
    W = bandit.get_W()
    
    color = "red" if outcome == "reward" else "C0"
    
    fig, axes = plt.subplots(
        nrows=2, ncols=3, sharey="row", sharex=True, figsize=(10,5)
    )
    
    n = len(A)
    titles = ["Split 0, pointwise CI", "Split 1, pointwise CI", "Whole dataset, joint CI"]
    indices = [*bandit_learning.get_splits(bandit.get_U(), overlap=0), np.arange(0,n,1)]
    levels = [level, level, level/len(bandit.action_space) ]
    
    for idx, (title, indices, level) in enumerate(zip(titles, indices, levels)):
        for subtract_baseline in [0,1]:
            ax = axes[subtract_baseline, idx]
            if subtract_baseline:
                if outcome != "reward":
                    ax.axhline(-safety_tol, ls=":", c="grey")
            else:
                ax.scatter(A[indices], Y[indices], c=color, s=7, alpha=0.5)
                ax.set_title(title)
                
            mean, ci_width = get_confidence_interval(
                x, phi_XA[indices], Y[indices], W[indices], level, bandit, 
                baseline_policy if subtract_baseline else None
            )
            ax.plot(bandit.action_space, mean, lw=2, c=color)
            ax.plot(bandit.action_space, mean-ci_width, ls="--", c=color)
            ax.plot(bandit.action_space, mean+ci_width, ls="--", c=color)
    
    axes[1,1].set_xlabel("Action")
    axes[0,0].set_ylabel("Reward" if outcome == "reward" else "Safety")
    axes[1,0].set_ylabel("Difference from a=0")
#%% 

EPSILON = lambda t: 0
safety_tol = 0
baseline_policy = lambda x: 0
# bandit_constructor = partial(BanditEnv.get_dosage_example, num_actions=20, param_count=10)

bandit_constructor = partial(
    BanditEnv.get_random_polynomial_bandit, num_actions=5, p=1, seed=47
)

alg_dict = {
    "FWER pretest" : utils.wrapped_partial(
            bandit_learning.alg_fwer_pretest_eps_greedy, 
            baseline_policy=baseline_policy,
            epsilon=EPSILON
        ),
    "SPT" : utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=False, 
            use_out_of_sample_covariance=False,
            sample_overlap=0,
            thompson_sampling=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=EPSILON
        )
}

#%% Search for disagreement between policies
num_random_timesteps = 100

search_for_disagreement = False

still_searching = True
while still_searching:
    bandit = bandit_constructor()
    bandit.reset(num_timesteps=num_random_timesteps, num_instances=1)
    
    for _ in range(num_random_timesteps):
        bandit.sample() # Note: required to step bandit forward
        a = np.random.choice(bandit.action_space)
        bandit.act([a], [1/len(bandit.action_space)])
    
    x = bandit.sample() 
    a_pretest, a_prob, info_pretest = alg_dict["FWER pretest"](x, bandit, alpha=0.1, safety_tol=safety_tol)
    a_pt, a_prob, info_pt = alg_dict["SPT"](x, bandit, alpha=0.1, safety_tol=safety_tol)
    
    if search_for_disagreement:
        still_searching = False
    
    if a_pretest != a_pt:
        still_searching = False

(ax_r, ax_s) = plot_bandit(x, bandit, title=f"True bandit with data (x={x[0]:0.2f})", baseline_policy=baseline_policy, safety_tol=safety_tol)
ax_r.scatter(bandit.A, bandit.R, s=8, alpha=0.5)
ax_s.scatter(bandit.A, bandit.S, s=8, alpha=0.5)

phi_XA = bandit.get_phi_XA()
R = bandit.get_R()
S = bandit.get_S()

beta_hat_R = utils.linear_regression(phi_XA, R, None)
beta_hat_S = utils.linear_regression(phi_XA, S, None)

if "beta_hat_R_bs" not in info_pretest: # This happens when no tests pass
    phi_XA_bs, R_bs = bandit_learning.bsample([phi_XA, R])    
    beta_hat_R_bs = utils.linear_regression(phi_XA_bs, R_bs, None)
else:
    beta_hat_R_bs = info_pretest["beta_hat_R_bs"]
    
plot_bandit(
    x,
    bandit, reward_param=beta_hat_R_bs, safety_param=beta_hat_S, 
    title=f"Estimated bandit with FWER (x={x[0]:0.2f}) (selected {a_pretest})", baseline_policy=baseline_policy,
    tested_safe_actions= info_pretest["safe_actions"],
    safety_tol=safety_tol
)

plot_propose_test(
    info_pt, bandit.action_space, show_test_result=True, 
    title=f"Split-Propose-Test objective (x={x[0]:0.2f}) (selected {a_pt})"
)

plot_data_w_ci(x, bandit, level=0.1, safety_tol=safety_tol, baseline_policy=baseline_policy)
plot_data_w_ci(x, bandit, level=0.1, safety_tol=safety_tol, baseline_policy=baseline_policy, outcome="reward")

#%% Plot SPT objective evolution
    
num_random_timesteps = 100
num_alg_timesteps = 30
plot_frequency = 10

bandit = bandit_constructor()
bandit.reset(num_timesteps=num_random_timesteps+num_alg_timesteps, num_instances=1)
for _ in range(num_random_timesteps):
    bandit.sample() # Note: required to step bandit forward
    a = np.random.choice(bandit.action_space)
    bandit.act([a], [1/len(bandit.action_space)])

spt_infos = []
for t in range(num_alg_timesteps):
    x = bandit.sample()
    a_pt, a_prob, info_pt = alg_dict["SPT"](x, bandit, alpha=0.1, safety_tol=safety_tol)
    bandit.act([a_pt], [a_prob])
    
    if t % plot_frequency == 0:
        ax_p, ax_r = plot_propose_test(info_pt, bandit.action_space, show_test_result=True, title=f"t={t}, x={x[0]:0.2f}")
        # ax_r.scatter(bandit.get_A()[::2], bandit.get_R()[::2], c="red", s=20, alpha=0.5)


    