import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _utils as utils

d_noise = 5

bandit_constructor = utils.wrapped_partial(
    BanditEnv.get_noisy_bandit_2, num_actions=5, p_noise=d_noise
)

safety_tol = 0
baseline_policy = lambda x: 0

evaluator = utils.wrapped_partial(
    bandit_learning.evaluate,
    experiment_name=f"Noisy bandit v2, (d_noise={d_noise})",
    bandit_constructor=bandit_constructor,
    baseline_policy=baseline_policy,
    num_instances=1,
    alpha=0.1,
    safety_tol=safety_tol
)