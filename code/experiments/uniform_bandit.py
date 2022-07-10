from functools import partial

import numpy as np

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _utils as utils

bandit_constructor = partial(
    BanditEnv.get_uniform_armed_bandit,
    means=[1, 1.5, 2], 
    prob_negative=[0, 0.15, 0.4]
)

EPSILON = lambda t: 0.1 / (t+1)**0.1
safety_tol = 0.3
baseline_policy = lambda x: 0

evaluator = partial(
    bandit_learning.evaluate,
    bandit_constructor=bandit_constructor,
    baseline_policy=baseline_policy,
    num_random_timesteps=20,
    num_alg_timesteps=380,
    num_instances=1,
    alpha=0.1,
    safety_tol=safety_tol
)

alg_dict = {
    "FWER pretest TS" : utils.wrapped_partial(
            bandit_learning.alg_fwer_pretest_ts, 
            baseline_policy=baseline_policy,
            epsilon=EPSILON
        ),
    "SPT" : utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=True, 
            use_out_of_sample_covariance=False,
            sample_overlap=0,
            thompson_sampling=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=EPSILON
        ),
    "SPT (FWER fallback) (safe)" : utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts_fwer_fallback,
            baseline_policy=baseline_policy, 
            correct_alpha=True, 
            num_actions_to_test=np.inf, 
            epsilon=EPSILON, 
        ),
    "SPT (FWER fallback) (unsafe)" : utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts_fwer_fallback,
            baseline_policy=baseline_policy, 
            correct_alpha=False, 
            num_actions_to_test=np.inf, 
            epsilon=EPSILON, 
        )
}
