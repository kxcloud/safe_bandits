import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _utils as utils

num_actions = 10

bandit_constructor = utils.wrapped_partial(
    BanditEnv.get_power_checker, num_actions=num_actions, effect_size=0.5
)

EPSILON = lambda t: 0.1 / (t+1)**0.1
safety_tol = 0
baseline_policy = lambda x: 0

evaluator = utils.wrapped_partial(
    bandit_learning.evaluate,
    experiment_name=f"Power checker ({num_actions} actions)",
    bandit_constructor=bandit_constructor,
    baseline_policy=baseline_policy,
    num_random_timesteps=100,
    num_alg_timesteps=300,
    num_instances=1,
    alpha=0.1,
    safety_tol=safety_tol
)