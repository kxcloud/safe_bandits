import numpy as np

import _bandit_learning as bandit_learning
import _utils as utils

EPSILON = 0.1
safety_tol = 0.1
baseline_policy = lambda x: 0

alg_dict = {
    "SPT (random split) (smart explore)" : utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts_smart_explore, 
            random_split=True, 
            use_out_of_sample_covariance=False,
            sample_overlap=0,
            thompson_sampling=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=EPSILON
        ),
    "SPT TS (random split) (smart explore)" : utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts_smart_explore, 
            random_split=True, 
            use_out_of_sample_covariance=False,
            sample_overlap=0,
            thompson_sampling=True,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=EPSILON
        )
}