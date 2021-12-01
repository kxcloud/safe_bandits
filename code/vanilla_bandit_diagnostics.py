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
    
    ax_reward.plot(bandit.action_space, expected_reward, c="black", ls=":", label="not safe")
    ax_safety.plot(bandit.action_space, expected_safety)
    ax_reward.set_title("Expected reward")
    ax_safety.set_title("Expected safety")
    ax_reward.set_xlabel("Action (a)")
    
    # Highlight safe regions
    if baseline_policy is not None:
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
        ax_reward.plot(bandit.action_space, safe_action_rewards, c="C2", lw=2.5, label="safe")
        ax_reward.legend()
        
    plt.suptitle(title)
    return (ax_reward, ax_safety)

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
            random_split=True, 
            use_out_of_sample_covariance=True,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=global_epsilon
        ),
}

num_random_timesteps = 50
bandit = BanditEnv.get_random_action_bandit(num_actions=50, outcome_correlation=0, p=6)
plot_bandit(bandit, title="True bandit", baseline_policy=baseline_policy)

for _ in range(num_random_timesteps):
    bandit.sample() # Note: required to step bandit forward
    a = np.random.choice(bandit.action_space)
    bandit.act(a)

x = bandit.sample() 

phi_XA = np.array(bandit.phi_XA)
R = np.array(bandit.R)
S = np.array(bandit.S)

beta_hat_R = bandit_learning.linear_regression(phi_XA, R)
beta_hat_S = bandit_learning.linear_regression(phi_XA, S)
plot_bandit(
    bandit, reward_param=beta_hat_R, safety_param=beta_hat_S, 
    title="Estimated bandit", baseline_policy=baseline_policy
)

for alg_label, alg in alg_dict.items():
    a = alg(x, bandit, alpha=0.1)
    print(f"{alg_label} selects {a}")