import numpy as np

import _BanditEnv as BanditEnv
import _utils as utils

bandit_constructor = utils.wrapped_partial(
    BanditEnv.get_high_dim_contextual_bandit, num_actions=5, p=1
)


bandit = bandit_constructor()

n = 100
bandit.reset(1, n)

X = bandit.sample()
A = np.random.choice(bandit.action_space, n)
bandit.act(A, None)
phi_XA = bandit.feature_vectorized(X, bandit.A[0])

r_mean = phi_XA @ bandit.reward_param
s_mean = phi_XA @ bandit.safety_param

r_sq_reward = np.sum((r_mean - r_mean.mean())**2) / np.sum((bandit.R - r_mean.mean())**2)
r_sq_safety = np.sum((s_mean - s_mean.mean())**2) / np.sum((bandit.S - s_mean.mean())**2)

print(f"R^2 reward: {r_sq_reward:0.3f}")
print(f"R^2 safety: {r_sq_safety:0.3f}")