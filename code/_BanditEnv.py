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
        
        self.X = []
        self.phi_XA = []
        self.A = []
        self.R = []
        self.S = []
        
        self.R_mean = []
        self.S_mean = []
        
        self.t = 0
        self.current_x = None
        
    def sample(self):
        self.current_x = self.x_dist()
        return self.current_x
    
    def _get_noise(self, a):
        reward_noise, safety_noise = np.random.multivariate_normal(
            np.zeros(2), self.outcome_covariance
        )
        return reward_noise, safety_noise
    
    def act(self, a):
        phi = self.feature_vector(self.current_x, a)
        mean_reward = np.dot(phi, self.reward_param)
        mean_safety = np.dot(phi, self.safety_param)
                        
        reward_noise, safety_noise = self._get_noise(a)
        
        r, s = mean_reward + reward_noise, mean_safety + safety_noise
        
        self.X.append(self.current_x)
        self.phi_XA.append(phi)
        self.A.append(a)
        self.R.append(r)
        self.S.append(s)
        
        self.R_mean.append(mean_reward)
        self.S_mean.append(mean_safety)
        
        self.t += 1
    
    def get_mean_rewards(self, a):
        """ Convenience function """
        phi = self.feature_vector(self.current_x, a)
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
                phi_a = self.feature_vector(X_grid, a)             
                ax.plot(X_grid, phi_a @ param, label=f"action={a}")
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
# "Standard bandit" means the context is ignored

def orthogonal_polynomial_feature(x, a, p, num_actions):
    x_vec = np.array([x**j for j in range(p+1)])
    phi_xa = np.concatenate([(a==k)*x_vec for k in range(num_actions)]).T
    return phi_xa

def standard_bandit_feature(x, a, num_actions):
    if type(x) is np.ndarray:
        phi_xa = np.zeros((len(x), num_actions))
        phi_xa[:, a] = 1
    else:
        phi_xa = np.zeros(num_actions)
        phi_xa[a] = 1
    return phi_xa

def standard_bandit_rbf_feature(x, a, param_count):
    """
    Assumes action space is [0,1].
    """
    param_center = a*param_count
    rbf_distances = np.abs(param_center - np.linspace(0, param_count, param_count))
    rbf_preweight = np.exp(-rbf_distances)
    phi = rbf_preweight / rbf_preweight.sum()
    
    if type(x) is np.ndarray:
        phi = np.tile(phi, (len(x),1))
    
    return phi

# PRESET BANDIT ENVS

def get_polynomial_bandit():
    theta_reward_0 = np.array([0, 2, 0, -2])*0
    theta_reward_1 = np.array([-1, 0.5, 2, 0])
    theta_reward = np.concatenate((theta_reward_0, theta_reward_1))
    
    theta_safety_0 = np.array([0, 0, 0, 0])
    theta_safety_1 = np.array([1, -1, 0, -1])
    theta_safety = np.concatenate((theta_safety_0, theta_safety_1))
    
    bandit = BanditEnv(
        x_dist=np.random.uniform, 
        action_space=[0,1],
        feature_vector=utils.wrapped_partial(
            orthogonal_polynomial_feature, p=3, num_actions=2
        ),
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_covariance=[[2,0],[0,2]]
    )
    return bandit

def get_random_polynomial_bandit(
        num_actions, p=3, seed=None
    ):   
    rng = np.random.default_rng(seed=seed)
    
    param_size = (p+1)*num_actions
    theta_reward = rng.normal(size=param_size)
    theta_safety = rng.normal(size=param_size)  
    
    bandit = BanditEnv(
        x_dist=np.random.uniform, 
        action_space=range(num_actions),
        feature_vector=utils.wrapped_partial(
            orthogonal_polynomial_feature, p=p, num_actions=num_actions
        ),
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_covariance=[[2,0],[0,2]]
    )
    return bandit
    
def get_standard_bandit(safety_means, outcome_std_dev):
    num_actions = len(safety_means)
    
    theta_reward = np.random.normal(size=num_actions)
    theta_safety = safety_means
    
    action_space = range(num_actions)
    
    feature_vector = utils.wrapped_partial(
        standard_bandit_feature, num_actions = num_actions
    )
    
    bandit = BanditEnv(
        x_dist=lambda : 0, 
        action_space=action_space,
        feature_vector=feature_vector,
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_covariance=[[outcome_std_dev**2,0],[0,outcome_std_dev**2]]
    )
    return bandit

def get_power_checker(num_actions, effect_size):
    """
    A bandit with only one arm worth considering.
    """
    theta_reward = np.zeros(num_actions)
    theta_reward[1:] = 100
    
    theta_safety = -np.ones(num_actions) * effect_size
    theta_safety[0] = 0
    theta_safety[1] = effect_size
    
    action_space = range(num_actions)
    
    feature_vector = utils.wrapped_partial(
        standard_bandit_feature, num_actions = num_actions
    )
                
    bandit = BanditEnv(
        x_dist=lambda : 0, 
        action_space=action_space,
        feature_vector=feature_vector,
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_covariance=[[1,0],[0,1]]
    )
    return bandit


def get_dosage_example(num_actions, param_count):
    dosage_reward = lambda x : 1 - 1/np.exp(5*x)
    dosage_safety = lambda x : 1 - 1/np.exp(-5*(x-1))
    
    action_space = np.linspace(0, 1, num_actions).round(3)
    
    rbf_feature_vector = utils.wrapped_partial(
        standard_bandit_rbf_feature, param_count=param_count    
    )
    
    ft_grid = np.zeros((num_actions, param_count))
    for a_idx, a in enumerate(action_space):
        ft_grid[a_idx,:] = rbf_feature_vector(0, a)
     
    rewards = [dosage_reward(a) for a in action_space]
    safetys = [dosage_safety(a) for a in action_space]
    
    theta_r = utils.linear_regression(ft_grid, rewards)
    theta_s = utils.linear_regression(ft_grid, safetys)
    
    bandit = BanditEnv(
        x_dist=lambda : 0, 
        action_space=action_space,
        feature_vector=rbf_feature_vector,
        reward_param=theta_r,
        safety_param=theta_s,
        outcome_covariance=[[0.1,0],[0,0.1]]
    )
    return bandit
