from functools import partial

import numpy as np

import _BanditEnv as BanditEnv
import _bandit_learning as bandit_learning
import _utils as utils

# General settings
num_runs = 500
num_processes = 10

# Bandit settings
bandit_constructor = partial(
    BanditEnv.get_uniform_armed_bandit,
    means=[1, 1.5, 2], 
    prob_negative=[0, 0.15, 0.4]
)

# Alg settings
EPSILON = 0.1
safety_tol = 0.3
baseline_policy = lambda x: 0

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
            can_propose_baseline_action=False,
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
