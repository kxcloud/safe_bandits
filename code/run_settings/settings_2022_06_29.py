import numpy as np

import _bandit_learning as bandit_learning
import _utils as utils

EPSILON = 0.1
safety_tol = 0.1
baseline_policy = lambda x: 0

alg_dict = {
    "FWER pretest" : utils.wrapped_partial(
            bandit_learning.alg_fwer_pretest_eps_greedy, 
            baseline_policy=baseline_policy,
            num_actions_to_test=np.inf,
            epsilon=EPSILON
        ),
    "SPT" : utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=False, 
            use_out_of_sample_covariance=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=EPSILON
        ),
    "SPT (smart explore)" : utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts_smart_explore, 
            random_split=False, 
            use_out_of_sample_covariance=False,
            baseline_policy=baseline_policy,
            objective_temperature=1,
            epsilon=EPSILON
        ),
}