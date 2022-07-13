import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _utils as utils

p=2

bandit_constructor = utils.wrapped_partial(
    BanditEnv.get_random_polynomial_bandit, num_actions=6, p=p
)

safety_tol = 0
baseline_policy = lambda x: 0

evaluator = utils.wrapped_partial(
    bandit_learning.evaluate,
    experiment_name=f"Polynomial bandit (p={2})",
    bandit_constructor=bandit_constructor,
    baseline_policy=baseline_policy,
    num_instances=1,
    alpha=0.1,
    safety_tol=safety_tol
)
