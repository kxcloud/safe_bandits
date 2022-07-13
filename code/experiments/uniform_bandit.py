import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _utils as utils

bandit_constructor = utils.wrapped_partial(
    BanditEnv.get_uniform_armed_bandit,
    means=[1, 1.5, 2], 
    prob_negative=[0, 0.15, 0.4]
)

safety_tol = 0.3
baseline_policy = lambda x: 0

evaluator = utils.wrapped_partial(
    bandit_learning.evaluate,
    experiment_name="Uniform-armed bandit",
    bandit_constructor=bandit_constructor,
    baseline_policy=baseline_policy,
    num_instances=1,
    alpha=0.1,
    safety_tol=safety_tol
)