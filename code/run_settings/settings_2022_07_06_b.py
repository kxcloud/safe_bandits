import numpy as np

import _bandit_learning as bandit_learning
import _utils as utils

EPSILON = 0.1
safety_tol = 0.1
baseline_policy = lambda x: 0

alg_dict = {}
for can_propose_baseline_action in [0,1]:
    for objective_temp in [2,4,8]:
        label = "SPT (r. split)"
        if not can_propose_baseline_action:
            label += " (no baseline)"
        label += f" (temp={objective_temp})"
        
        alg_dict[label] = utils.wrapped_partial(
            bandit_learning.alg_propose_test_ts, 
            random_split=True, 
            use_out_of_sample_covariance=False,
            sample_overlap=0,
            thompson_sampling=False,
            can_propose_baseline_action=can_propose_baseline_action,
            baseline_policy=baseline_policy,
            objective_temperature=objective_temp,
            epsilon=EPSILON
        )