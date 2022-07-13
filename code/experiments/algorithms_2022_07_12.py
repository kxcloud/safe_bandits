import _bandit_learning as bandit_learning
import _utils as utils

EPSILON = lambda t: 0.1 / (t+1)**0.1

def get_alg_dict(baseline_policy, safety_tol):
    alg_dict = {
        "Unsafe TS" : utils.wrapped_partial(
            bandit_learning.alg_unsafe_ts,
            epsilon=EPSILON
            ),
        "Oracle" : utils.wrapped_partial(
            bandit_learning.alg_oracle,
            baseline_policy=baseline_policy
        ),
        "Pretest all" : utils.wrapped_partial(
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
        "SPT (fallback) (safe)" : utils.wrapped_partial(
                bandit_learning.alg_propose_test_ts_fwer_fallback,
                baseline_policy=baseline_policy, 
                correct_alpha=True, 
                epsilon=EPSILON, 
            ),
        # "SPT (fallback) (unsafe)" : utils.wrapped_partial(
        #         bandit_learning.alg_propose_test_ts_fwer_fallback,
        #         baseline_policy=baseline_policy, 
        #         correct_alpha=False, 
        #         epsilon=EPSILON, 
        #     )
    }
    return alg_dict

burn_in_samples_per_action = 4
num_alg_timesteps = 350