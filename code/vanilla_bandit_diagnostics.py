from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt

import BanditEnv
import bandit_learning
import visualize_results

def plot_bandit(
        bandit, 
        figsize=(8.5,4), 
        title="", 
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
        bandit.feature_vector(0, a) @ reward_param for a in bandit.action_space
    ]
    expected_safety = [
        bandit.feature_vector(0, a) @ safety_param for a in bandit.action_space
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
        a_baseline = baseline_policy(0) # WARNING: assumes baseline policy ignores x 
        baseline_safety = bandit.feature_vector(0, a_baseline) @ safety_param
        
        safe_action_rewards = []           
        for action, reward, safety in zip(
                bandit.action_space, expected_reward, expected_safety
            ):
            if safety >= baseline_safety - 1e-8:
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
            zorder=9)
        ax_reward.scatter(
            bandit.action_space, tested_safe_action_rewards, c="gold", marker="D", label="tested safe",
            zorder=10)
        ax_reward.legend()
        
    plt.suptitle(title)
    return (ax_reward, ax_safety)

def plot_propose_test(info, action_space, title=""):
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
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(action_space, split[:,0], label="estimated pass probability", c="C0", ls="--")
    ax.plot(action_space, test_results, label="would pass if tested", c="gold", lw=3.5)
    
    ax2.plot(action_space, split[:,1], label="estimated improvement", c="red")
    ax2.plot(action_space, objective, label="objective", lw=2, c="black")
    ax2.scatter(a_max, objective[idx_max], marker="*", c="black", s=100)

    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax2.set_ylabel("Reward")
    ax.set_xlabel("Action (a)")
    
    fig.legend(bbox_to_anchor=[1.4,0.4,0,0])
#%% 

wrapped_partial = bandit_learning.wrapped_partial
baseline_policy = lambda x: 0

global_epsilon = 0

alg_dict = {
    "FWER pretest: TS (test all)" : wrapped_partial(
            bandit_learning.alg_fwer_pretest_ts, 
            baseline_policy=baseline_policy,
            num_actions_to_test=np.inf,
            epsilon=global_epsilon
        ),
    "Propose-test TS" : wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=False, 
            use_out_of_sample_covariance=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=global_epsilon
        ),
    "Propose-test TS (OOS covariance)" : wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=False, 
            use_out_of_sample_covariance=True,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=global_epsilon
        ),
}

#%% Search for disagreement between policies
num_random_timesteps = 100

searching_for_disagreement = True
while searching_for_disagreement:
    bandit = BanditEnv.get_random_action_bandit(num_actions=30, outcome_correlation=0, p=3)
    for _ in range(num_random_timesteps):
        bandit.sample() # Note: required to step bandit forward
        a = np.random.choice(bandit.action_space)
        bandit.act(a)
    
    x = bandit.sample() 
    a_pretest, info_pretest = alg_dict["FWER pretest: TS (test all)"](x, bandit, alpha=0.1)
    a_pt, info_pt = alg_dict["Propose-test TS"](x, bandit, alpha=0.1)
    
    if a_pretest != a_pt:
        searching_for_disagreement = False

plot_bandit(bandit, title="True bandit", baseline_policy=baseline_policy)

phi_XA = np.array(bandit.phi_XA)
R = np.array(bandit.R)
S = np.array(bandit.S)

beta_hat_R = bandit_learning.linear_regression(phi_XA, R)
beta_hat_S = bandit_learning.linear_regression(phi_XA, S)

if "beta_hat_R_bs" not in info_pretest: # This happens when no tests pass
    phi_XA_bs, R_bs = bandit_learning.bsample([phi_XA, R])    
    beta_hat_R_bs = bandit_learning.linear_regression(phi_XA_bs, R_bs)
else:
    beta_hat_R_bs = info_pretest["beta_hat_R_bs"]
    
plot_bandit(
    bandit, reward_param=beta_hat_R_bs, safety_param=beta_hat_S, 
    title=f"Estimated bandit with FWER (selected {a_pretest})", baseline_policy=baseline_policy,
    tested_safe_actions= info_pretest["safe_actions"]
)

plot_propose_test(info_pt, bandit.action_space, f"Propose-test objective (selected {a_pt})")
