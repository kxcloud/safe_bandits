import numpy as np
import matplotlib.pyplot as plt

import _utils as utils

class BanditEnv:
    
    def __init__(
            self, 
            x_dist,
            action_space,
            feature_vector,
            reward_param,
            safety_param,
            outcome_covariance,
        ):
        self.x_dist = x_dist
        self.action_space = action_space
        self.feature_vector = feature_vector
        self.reward_param = reward_param
        self.safety_param = safety_param
        self.outcome_covariance = outcome_covariance
        
        self.action_idx = {action:idx for idx, action in enumerate(action_space)}
        
    def reset(self, num_timesteps, num_instances):
        """
        Create arrays to store data from a run.
        """
        self.num_timesteps = num_timesteps
        self.num_instances = num_instances
        
        x = self.x_dist()
        a = self.action_space[0] 
        self.x_length = len(x) if hasattr(x, "__len__") else 1
        feature_length = len(self.feature_vector(x,a))
        
        self.X = np.zeros((num_timesteps, num_instances, self.x_length))
        self.phi_XA = np.zeros((num_timesteps, num_instances, feature_length))
        self.A = np.zeros_like(self.action_space, shape=(num_timesteps, num_instances))
        self.R = np.zeros((num_timesteps, num_instances))
        self.S = np.zeros((num_timesteps, num_instances))
        self.W = np.ones((num_timesteps, num_instances)) # sqrt importance weights
        self.U = np.random.uniform(size=(num_timesteps, num_instances)) # random seeds for alg
        
        self.R_mean = np.zeros((num_timesteps, num_instances))
        self.S_mean = np.zeros((num_timesteps, num_instances))
        
        self.indices_by_action = {a : [] for a in self.action_space}
        
        self.t = 0
        self.current_x = None
    
    def sample(self):
        self.current_x = np.array([self.x_dist() for _ in range(self.num_instances)])
        return self.current_x
    
    def _get_noise(self, a_batch):
        reward_noise, safety_noise = np.random.multivariate_normal(
            np.zeros(2), self.outcome_covariance, size=self.num_instances
        ).T
        return reward_noise, safety_noise
        
    def act(self, a_batch, a_prob_batch):
        phi_batch = self.feature_vectorized(self.current_x, a_batch)

        mean_reward = phi_batch @ self.reward_param
        mean_safety = phi_batch @ self.safety_param
                        
        reward_noise, safety_noise = self._get_noise(a_batch)
        
        r_batch, s_batch = mean_reward + reward_noise, mean_safety + safety_noise
        
        self.X[self.t] = self.current_x#[:, None]
        self.phi_XA[self.t] = phi_batch
        self.A[self.t] = a_batch
        self.R[self.t] = r_batch
        self.S[self.t] = s_batch
        # self.W[self.t] = np.sqrt(1/np.array(a_prob_batch))
        
        self.R_mean[self.t] = mean_reward
        self.S_mean[self.t] = mean_safety
        
        for instance_idx, a in enumerate(a_batch):
            self.indices_by_action[a].append((self.t, instance_idx))
        
        self.t += 1
    
    def get_phi_XA(self, flatten=True):
        if flatten:
            return self.phi_XA[:self.t].reshape((-1, self.phi_XA.shape[-1]))
        else:
            return self.phi_XA[:self.t]

    def get_X(self, flatten=True):
        if flatten:
            return self.X[:self.t].reshape((-1, self.X.shape[-1]))
        else:
            return self.X[:self.t]

    def get_A(self, flatten=True):
        if flatten:
            return self.A[:self.t].reshape(-1).squeeze()
        else:
            return self.A[:self.t]
        
    def get_R(self, flatten=True):
        if flatten:
            return self.R[:self.t].reshape(-1).squeeze()
        else:
            return self.R[:self.t]
    
    def get_S(self, flatten=True):
        if flatten:
            return self.S[:self.t].reshape(-1).squeeze()
        else:
            return self.S[:self.t]
        
    def get_W(self, flatten=True):
        if flatten:
            return self.W[:self.t].reshape(-1).squeeze()
        else:
            return self.W[:self.t]
        
    def get_U(self, flatten=True):
        if flatten:
            return self.U[:self.t].reshape(-1).squeeze()
        else:
            return self.U[:self.t]
    
    def feature_vectorized(self, x_batch, a_batch):
        x_is_batched = hasattr(x_batch[0], "__len__")
        a_is_batched = hasattr(a_batch, "__len__")
        
        if not x_is_batched and not a_is_batched:
            return self.feature_vector(x_batch, a_batch)
        
        if x_is_batched and not a_is_batched:
            a_to_broadcast = a_batch
            a_batch = [a_to_broadcast for x in x_batch]
        
        if not x_is_batched and a_is_batched:
            x_to_broadcast = x_batch
            x_batch = [x_to_broadcast for a in a_batch]
        
        phi_XA = []
        assert len(x_batch) == len(a_batch), "Contexts and actions must be same size"
        for x, a in zip(x_batch, a_batch):
            phi_XA.append(self.feature_vector(x, a))
        return np.array(phi_XA)
    
    def get_mean_rewards(self, a_batch):
        """ Convenience function """
        raise NotImplementedError()
        phi = self.feature_vector(self.current_x, a_batch)
        mean_reward = np.dot(phi, self.reward_param)
        mean_safety = np.dot(phi, self.safety_param)
        return mean_reward, mean_safety
    
    def plot(self, figsize=(8,4), title="", reward_param=None, safety_param=None, legend=True):
        reward_param = self.reward_param if reward_param is None else reward_param
        safety_param = self.safety_param if safety_param is None else safety_param
        targets = {"Mean reward" : reward_param, "Mean safety" : safety_param}

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        for idx, (label, param) in enumerate(targets.items()):
            ax = axes[idx]
            X_grid = np.linspace(0, 1, 70)
            for a_idx, a in enumerate(self.action_space):
                phi_a = self.feature_vectorized(X_grid, a)             
                ax.plot(X_grid, phi_a @ param, label=f"action={a}", lw=2)
            ax.set_title(label)
            ax.set_xlabel("Context (x)")
        
        if legend:
            axes[-1].legend()
        plt.suptitle(title)
        return fig, axes
    
    def plot_actions_at_context(
            self, context, figsize=(8,4), title="", reward_param=None, safety_param=None
        ):
        reward_param = self.reward_param if reward_param is None else reward_param
        safety_param = self.safety_param if safety_param is None else safety_param
        targets = {"Mean reward" : reward_param, "Mean safety" : safety_param}

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        for idx, (label, param) in enumerate(targets.items()):
            ax = axes[idx]
            targets = [self.feature_vector(context, a) @ param for a in self.action_space]
            ax.scatter(self.action_space, targets)
            ax.set_title(label)
            ax.set_xlabel("Action (a)")

        plt.suptitle(title)
        return fig, axes


# FEATURE VECTORS
# Apply on single (x,a) pairs -- no broadcasting
# "Standard bandit" means the context is ignored

def orthogonal_polynomial_feature(x, a, p, num_actions):
    """ Assumes action space is {0, 1, ... num_actions-1}. """
    polynomial_terms = p+1
    phi_xa = np.zeros(num_actions*polynomial_terms)
    for j in range(polynomial_terms):
        phi_xa[a*polynomial_terms + j] = x**j
    return phi_xa

def standard_bandit_feature(x, a, num_actions):
    """ Assumes action space is {0, 1, ... num_actions-1}. """
    phi_xa = np.zeros(num_actions)
    phi_xa[a] = 1
    return phi_xa

def standard_bandit_rbf_feature(x, a, param_count):
    """ Assumes action space is [0,1]. """
    param_center = a*param_count
    rbf_distances = np.abs(param_center - np.linspace(0, param_count, param_count))
    rbf_preweight = np.exp(-rbf_distances)
    phi = rbf_preweight / rbf_preweight.sum()
    return phi


# PRESET BANDIT ENVS

def get_polynomial_bandit(**kwargs):
    context_rng = kwargs.get("context_rng", np.random.default_rng())
    x_dist = utils.wrapped_partial(context_rng.uniform, size=1)
    
    theta_reward_0 = np.array([0, 2, 0, -2])*0
    theta_reward_1 = np.array([-1, 0.5, 2, 0])
    theta_reward = np.concatenate((theta_reward_0, theta_reward_1))
    
    theta_safety_0 = np.array([0, 0, 0, 0])
    theta_safety_1 = np.array([1, -1, 0, -1])
    theta_safety = np.concatenate((theta_safety_0, theta_safety_1))
    
    bandit = BanditEnv(
        x_dist=x_dist, 
        action_space=[0,1],
        feature_vector=utils.wrapped_partial(
            orthogonal_polynomial_feature, p=3, num_actions=2
        ),
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_covariance=[[2,0],[0,2]]
    )
    return bandit

def get_random_polynomial_bandit(num_actions, p=3, **kwargs):   
    param_rng = kwargs.get("param_rng", np.random.default_rng())
    context_rng = kwargs.get("context_rng", np.random.default_rng())
    
    x_dist = utils.wrapped_partial(context_rng.uniform, size=1)
    
    param_size = (p+1)*num_actions
    theta_reward = param_rng.normal(size=param_size)
    theta_safety = param_rng.normal(size=param_size)
    
    bandit = BanditEnv(
        x_dist=x_dist, 
        action_space=list(range(num_actions)),
        feature_vector=utils.wrapped_partial(
            orthogonal_polynomial_feature, p=p, num_actions=num_actions
        ),
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_covariance=[[1,0],[0,1]]
    )
    return bandit
    
def get_standard_bandit(safety_means, outcome_covariance, reward_means=None, **kwargs):
    param_rng = kwargs.get("param_rng", np.random.default_rng())
    
    num_actions = len(safety_means)
        
    theta_reward = param_rng.normal(size=len(safety_means)) if reward_means is None else reward_means
    theta_safety = safety_means
    
    action_space = list(range(num_actions))
    
    feature_vector = utils.wrapped_partial(
        standard_bandit_feature, num_actions = num_actions
    )
    
    bandit = BanditEnv(
        x_dist=lambda : np.array([0]), 
        action_space=action_space,
        feature_vector=feature_vector,
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_covariance=outcome_covariance
    )
    return bandit

def get_power_checker(num_actions, effect_size, **kwargs):
    """
    A non-contextual bandit with only one arm worth considering.
    """
    theta_reward = np.zeros(num_actions)
    theta_reward[1] = 100
    
    theta_safety = -np.ones(num_actions) * effect_size
    theta_safety[0] = 0
    theta_safety[1] = effect_size
    
    action_space = list(range(num_actions))
    
    feature_vector = utils.wrapped_partial(
        standard_bandit_feature, num_actions = num_actions
    )
                
    bandit = BanditEnv(
        x_dist=lambda : np.array([0]), 
        action_space=action_space,
        feature_vector=feature_vector,
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_covariance=[[1,0],[0,1]]
    )
    return bandit


def get_dosage_example(num_actions, param_count, outcome_correlation, **kwargs):
    """
    A non-contextual linear bandit where reward is increasing and safety is 
    decreasing in action ("dosage") level, and information is pooled across 
    nearby dosage levels using a radial basis function representation for 
    features.
    """
    action_space = np.linspace(0, 1, num_actions).round(3)
    
    dosage_reward = lambda a : 1 - 1/np.exp(5*a)
    dosage_safety = lambda a : 1 - 1/np.exp(-5*(a-1))
    
    rbf_feature_vector = utils.wrapped_partial(
        standard_bandit_rbf_feature, param_count=param_count    
    )
    
    ft_grid = np.zeros((num_actions, param_count))
    for a_idx, a in enumerate(action_space):
        ft_grid[a_idx,:] = rbf_feature_vector(0, a)
     
    rewards = [dosage_reward(a) for a in action_space]
    safetys = [dosage_safety(a) for a in action_space]
    
    theta_r = utils.linear_regression(ft_grid, rewards, None)
    theta_s = utils.linear_regression(ft_grid, safetys, None)
    
    var = 0.01
    cov = var*outcome_correlation
    
    bandit = BanditEnv(
        x_dist=lambda : np.array([0]), 
        action_space=action_space,
        feature_vector=rbf_feature_vector,
        reward_param=theta_r,
        safety_param=theta_s,
        outcome_covariance=[[var,cov],[cov,var]]
    )
    return bandit

def get_uniform_armed_bandit(means, prob_negative, **kwargs):
    """
    A non-contextual, standard bandit where arm i has reward distribution
    Uniform([l_i, u_i]) and safety is defined as reward being positive.
    """
    reward_param = np.array(means)
    safety_param = 1 - np.array(prob_negative)
    
    assert np.all(reward_param >= 0), "Means must be nonnegative"
    
    lower_bounds = 2*(1-safety_param)*reward_param / (1-2*safety_param)
    upper_bounds = 2*reward_param - lower_bounds
    
    action_space = list(range(len(means)))
    
    feature_vector = utils.wrapped_partial(
        standard_bandit_feature, num_actions = len(action_space)
    )
    
    bandit = BanditEnv(
        x_dist=lambda : np.array([0]), 
        action_space=list(range(len(means))),
        feature_vector=feature_vector,
        reward_param=reward_param,
        safety_param=safety_param,
        outcome_covariance=None,
    )
    
    def custom_noise(a_batch):
        rewards = np.random.uniform(lower_bounds[a_batch], upper_bounds[a_batch])
        reward_noise = rewards - reward_param[a_batch]
        safety = rewards >= 0
        safety_noise = safety - safety_param[a_batch]
        return reward_noise, safety_noise
    
    bandit._get_noise = custom_noise
        
    return bandit

def get_high_dim_contextual_bandit(p, num_actions, **kwargs):
    param_rng = kwargs.get("param_rng", np.random.default_rng())
    context_rng = kwargs.get("context_rng", np.random.default_rng())
    
    x_dim = 5
    x_and_a_dim = 2*x_dim
    
    x_dist = utils.wrapped_partial(context_rng.normal, size=x_dim)
    action_space = np.linspace(-2, 2, num_actions)
    
    reward_param = param_rng.normal(size=p, scale=10)
    reward_param[p//2:] = 0 # half of contexts don't matter
    safety_param = param_rng.normal(size=p, scale=10)
    safety_param[p//2:] = 0 # half of contexts don't matter

    knots = param_rng.normal(size=(x_and_a_dim, p))
    
    def feature_vector(x, a):
        distances = np.linalg.norm(np.append(x,[a]*x_dim)[:,None] - knots, 2, axis=0)
        return np.exp(-distances)
    
    bandit = BanditEnv(
        x_dist=x_dist, 
        action_space=action_space,
        feature_vector=feature_vector,
        reward_param=reward_param,
        safety_param=safety_param,
        outcome_covariance=[[1e-1,0],[0,5e-3]]
    )
    return bandit


def get_noisy_bandit_2(p_noise, num_actions, **kwargs):
    context_rng = kwargs.get("context_rng", np.random.default_rng())
    
    """ A standard bandit but with """
    p_total = num_actions+p_noise
    
    def standard_bandit_feature_w_noise_features(x, a):
        """ Assumes action space is {0, 1, ... num_actions-1}. """
        phi_xa = np.zeros(p_total)
        phi_xa[a] = 1
        phi_xa[num_actions:] = x
        return phi_xa

    reward_param = np.zeros(p_total)
    reward_param[:num_actions] = np.linspace(0, 40, num_actions)
    reward_param[num_actions:] = 0
    
    safety_param = np.zeros(p_total) #rng.normal(size=p_total)
    safety_param[1:num_actions] = np.linspace(40, 20, num_actions-1)
    safety_param[1:num_actions:2] = -10
    safety_param[num_actions:] = 0

    x_dist = utils.wrapped_partial(context_rng.normal, size=p_noise)

    bandit = BanditEnv(
        x_dist=x_dist, 
        action_space=list(range(num_actions)),
        feature_vector=standard_bandit_feature_w_noise_features,
        reward_param=reward_param,
        safety_param=safety_param,
        outcome_covariance=[[60**2,0],[0,20**2]]
    )
    return bandit


def get_contextual_bandit(reward_betas, safety_betas, context_rng, **kwargs):
    """ 
    Multivariate normal context; unique beta for each reward and safety,
    passed as matrix of shape (num_actions, d). Betas are orthogonal
    by action but beta^r[a] can be non-orthogonal to beta^s[a]
    """
    assert reward_betas.shape == safety_betas.shape
    
    num_actions, d = reward_betas.shape
    d_total = d * num_actions
        
    def contextual_bandit_feature(x, a):
        phi_xa = np.zeros(d_total)
        start_idx_reward = d*a
        phi_xa[start_idx_reward:start_idx_reward+d] = x
        return phi_xa

    # Need to make these overlap
    reward_param = reward_betas.flatten()
    safety_param = safety_betas.flatten()
    
    x_dist = utils.wrapped_partial(context_rng.normal, size=d)

    bandit = BanditEnv(
        x_dist=x_dist, 
        action_space=list(range(num_actions)),
        feature_vector=contextual_bandit_feature,
        reward_param=reward_param,
        safety_param=safety_param,
        outcome_covariance=[[4**2,0],[0,1e-3]]
    )
    return bandit  

def get_contextual_bandit_by_correlation(num_actions, d, correlation, **kwargs):
    param_rng = kwargs.get("param_rng", np.random.default_rng())
    context_rng = kwargs.get("context_rng", np.random.default_rng())

    reward_betas = np.zeros((num_actions, 2*d))
    safety_betas = np.zeros((num_actions, 2*d))

    for a in range(num_actions):
        reward_betas[a][:d] = param_rng.normal(size=d)
        safety_betas[a][:d] = correlation * reward_betas[a][:d]
        safety_betas[a][d:] = (1-correlation) * param_rng.normal(size=d)

    return get_contextual_bandit(reward_betas, safety_betas, context_rng=context_rng)


if __name__ == "__main__":
    num_instances = 1
    
    num_actions = 5
    
    bandit = get_contextual_bandit_by_correlation(num_actions, correlation=0, d=1)
    bandit.reset(10, 1)
    x = bandit.sample()
