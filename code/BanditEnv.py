from functools import partial

import numpy as np
import matplotlib.pyplot as plt

class BanditEnv:
    
    def __init__(
            self, 
            x_dist,
            action_space,
            feature_vector,
            reward_param,
            safety_param,
            outcome_std_dev,
            outcome_correlation
        ):
        self.x_dist = x_dist
        self.action_space = action_space
        self.feature_vector = feature_vector
        self.reward_param = reward_param
        self.safety_param = safety_param
        self.outcome_covariance = outcome_std_dev**2 * np.array(
            [[1, outcome_correlation], [outcome_correlation, 1]]
        )
        
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
       
def polynomial_feature(x, a, p, num_actions):
    x_vec = np.array([x**j for j in range(p+1)])
    phi_xa = np.concatenate([(a==k)*x_vec for k in range(num_actions)]).T
    return phi_xa

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
        feature_vector=partial(polynomial_feature, p=3, num_actions=2),
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_std_dev=2,
        outcome_correlation=0.8
    )
    return bandit

def get_random_polynomial_bandit(num_actions, outcome_correlation, p=3, seed=None):   
    rng = np.random.default_rng(seed=seed)
    
    param_size = (p+1)*num_actions
    theta_reward = rng.normal(size=param_size)
    theta_safety = rng.normal(size=param_size)  
    
    bandit = BanditEnv(
        x_dist=np.random.uniform, 
        action_space=range(num_actions),
        feature_vector=partial(polynomial_feature, p=p, num_actions=num_actions),
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_std_dev=2,
        outcome_correlation=outcome_correlation
    )
    return bandit
    
def sinusoidal_feature(x, a):
    one = np.ones_like(x)
    phi_xa = np.array(
        [one, (a!=0)*one, (a!=0)*x, a, (a!=0)*np.sin(x*5 + a), 
         x, x**2, np.sin(x*5), x*a]
    ).T
    return phi_xa

def get_sinusoidal_bandit():
    theta_reward = np.array([0, 0, 2,    1, 1, 0, 0, 0, 0])
    theta_safety = np.array([0, 1, 2, -0.5, 0, 0, 0, 0, 0])
   
    bandit = BanditEnv(
        x_dist=np.random.uniform, 
        action_space=range(6),
        feature_vector=sinusoidal_feature,
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_std_dev=2,
        outcome_correlation=0.8
    )
    return bandit

def get_random_action_bandit(num_actions, outcome_correlation, p=3, seed=None):
    """ A non-contextual bandit. """
    rng = np.random.default_rng(seed=seed)
    
    param_size = (p+1)
    theta_reward = rng.normal(size=param_size)
    theta_safety = rng.normal(size=param_size)
    
    def polynomial_action(x, a):
        return polynomial_feature(np.full_like(x, a, dtype=float), 0, p=p, num_actions=1)
    
    action_space = np.round(np.linspace(0, 1, num=num_actions),2)
    
    bandit = BanditEnv(
        x_dist=lambda : 0, 
        action_space=action_space,
        feature_vector=polynomial_action,
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_std_dev=2,
        outcome_correlation=outcome_correlation
    )
    return bandit

def get_example_bandit(num_actions, p=2, seed=None):
    """ A non-contextual bandit. """
    
    param_size = (p+1)
    theta_reward = np.zeros(param_size)
    theta_reward[1] = 1
    theta_safety = -0.5*theta_reward
    
    def polynomial_action(x, a):
        return polynomial_feature(np.full_like(x, a, dtype=float), 0, p=p, num_actions=1)
    
    action_space = np.round(np.linspace(0, 1, num=num_actions),2)
    
    bandit = BanditEnv(
        x_dist=lambda : 0, 
        action_space=action_space,
        feature_vector=polynomial_action,
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_std_dev=1,
        outcome_correlation=0
    )
    return bandit

def get_standard_bandit(safety_means, outcome_std_dev):
    num_actions = len(safety_means)
    
    theta_reward = np.random.normal(size=num_actions)
    theta_safety = safety_means
    
    action_space = range(num_actions)
    
    def feature_vector(x, a):
        if type(x) is np.ndarray:
            phi_xa = np.zeros((len(x), num_actions))
            phi_xa[:, a] = 1
        else:
            phi_xa = np.zeros(num_actions)
            phi_xa[a] =1
        return phi_xa
                
    bandit = BanditEnv(
        x_dist=lambda : 0, 
        action_space=action_space,
        feature_vector=feature_vector,
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_std_dev=outcome_std_dev,
        outcome_correlation=0
    )
    return bandit

def get_pretest_punisher(outcome_std_dev):
    """
    A bandit with many obviously-low-reward actions.
    """
    num_actions = 50
    num_good_actions = 1
    
    theta_reward = np.zeros(num_actions)
    theta_reward[0] = -1
    theta_reward[1:num_good_actions+1] = 5
    
    theta_safety = -np.ones(num_actions)
    theta_safety[1:num_good_actions+1] = 0.25
    theta_safety[0] = 0
    
    action_space = range(num_actions)
    
    def feature_vector(x, a):
        if type(x) is np.ndarray:
            phi_xa = np.zeros((len(x), num_actions))
            phi_xa[:, a] = 1
        else:
            phi_xa = np.zeros(num_actions)
            phi_xa[a] =1
        return phi_xa
                
    bandit = BanditEnv(
        x_dist=lambda : 0, 
        action_space=action_space,
        feature_vector=feature_vector,
        reward_param=theta_reward,
        safety_param=theta_safety,
        outcome_std_dev=outcome_std_dev,
        outcome_correlation=0
    )
    return bandit

if __name__ == "__main__":
    def linear_regression(x_mat, y, penalty=0.01):
        return np.linalg.solve(x_mat.T @ x_mat + penalty * np.identity(x_mat.shape[1]), x_mat.T @ y)
    
    num_actions = 50
    bandit = get_example_bandit(num_actions)

    for _ in range(60):
        x = bandit.sample()
        a = np.random.choice(bandit.action_space)
        bandit.act(a)
    
    phi_XA = np.array(bandit.phi_XA)
    reward_param_est = linear_regression(phi_XA, np.array(bandit.R), penalty=0.1)
    safety_param_est = linear_regression(phi_XA, np.array(bandit.S), penalty=0.1)
    
    bandit.plot(title=f"Example of random polynomial bandit (num_actions={num_actions})", legend=False)
    bandit.plot(reward_param = reward_param_est, safety_param = safety_param_est, title="estimates", legend=False)