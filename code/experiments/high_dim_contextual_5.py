import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _utils as utils

d = 5

bandit_constructor = utils.wrapped_partial(
    BanditEnv.get_high_dim_contextual_bandit, num_actions=5, p=d
)

EPSILON = lambda t: 0.1 / (t+1)**0.1
safety_tol = 0
baseline_policy = lambda x: -2

evaluator = utils.wrapped_partial(
    bandit_learning.evaluate,
    experiment_name=f"High-dim context (d={d})",
    bandit_constructor=bandit_constructor,
    baseline_policy=baseline_policy,
    num_random_timesteps=25,
    num_alg_timesteps=350,
    num_instances=1,
    alpha=0.1,
    safety_tol=safety_tol
)