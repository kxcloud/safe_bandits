import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _utils as utils

correlation = -0.5
d=1

bandit_constructor = utils.wrapped_partial(
    BanditEnv.get_contextual_bandit_by_correlation, num_actions=5, d=d, correlation=correlation
)

safety_tol = 0
baseline_policy = lambda x: 0


evaluator = utils.wrapped_partial(
    bandit_learning.evaluate,
    experiment_name=f"Contextual bandit (d={d}, Reward-safety corr={correlation})",
    bandit_constructor=bandit_constructor,
    baseline_policy=baseline_policy,
    num_instances=1,
    alpha=0.1,
    safety_tol=safety_tol
)

override_burn_in_samples_per_action = d*4