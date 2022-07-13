import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _utils as utils

outcome_correlation = 0.5
bandit_constructor = utils.wrapped_partial(
    BanditEnv.get_dosage_example, num_actions=20, param_count=10, outcome_correlation=outcome_correlation
)
safety_tol = 0.1
baseline_policy = lambda x: 0

evaluator = utils.wrapped_partial(
    bandit_learning.evaluate,
    experiment_name=f"Dosage bandit (corr={outcome_correlation})",
    bandit_constructor=bandit_constructor,
    baseline_policy=baseline_policy,
    num_instances=1,
    alpha=0.1,
    safety_tol=safety_tol
)