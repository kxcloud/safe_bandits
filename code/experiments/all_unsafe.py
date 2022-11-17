import numpy as np

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _utils as utils

num_actions = 10
safety_means = np.zeros(num_actions)
safety_means[1:] = -1
reward_means = np.zeros(num_actions)
reward_means[1:] = 100

bandit_constructor = utils.wrapped_partial(
    BanditEnv.get_standard_bandit, 
    safety_means=safety_means,
    reward_means=reward_means,
    outcome_covariance=[[1,0], [0,10**2]]
)

safety_tol = 0
baseline_policy = lambda x: 0

evaluator = utils.wrapped_partial(
    bandit_learning.evaluate,
    experiment_name="All unsafe",
    bandit_constructor=bandit_constructor,
    baseline_policy=baseline_policy,
    num_instances=1,
    alpha=0.1,
    safety_tol=safety_tol
)
