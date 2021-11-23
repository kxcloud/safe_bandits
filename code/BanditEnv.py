import numpy as np
import matplotlib.pyplot as plt

class BanditEnv:
    
    def __init__(
            self, 
            x_dist,
            action_space,
            feature_vector,
            reward_param,
            safety_param
        ):
        self.x_dist = x_dist
        self.action_space = action_space
        self.feature_vector = feature_vector
        self.reward_param = reward_param
        self.safety_param = safety_param
        
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
        reward_noise = np.random.normal()
        safety_noise = np.random.normal()
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
    
    def plot(self, figsize=(8,4)):
        targets = {"reward" : self.reward_param, "safety" : self.safety_param}

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        for idx, (label, param) in enumerate(targets.items()):
            ax = axes[idx]
            X_grid = np.linspace(0, 1, 70)
            for a in self.action_space:
                phi_a = self.feature_vector(X_grid, a)
                ax.plot(X_grid, phi_a @ param, label=f"action={a}")
            ax.set_title(label)
            ax.legend()
        return fig, axes
    
def polynomial_feature(x, a, p=3, num_actions=2):
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
        feature_vector=polynomial_feature,
        reward_param=theta_reward,
        safety_param=theta_safety
    )
    return bandit

def sinusoidal_feature(x, a):
    one = np.ones_like(x)
    phi_xa = np.array([one, (a!=0)*one, (a!=0)*x, a, (a!=0)*np.sin(x*5 + a)]).T
    return phi_xa

def get_sinusoidal_bandit():
    theta_reward = np.array([0, 0, 2, 1, 1])
    theta_safety = np.array([0, 1, 2, -0.5, 0])
   
    bandit = BanditEnv(
        x_dist=np.random.uniform, 
        action_space=range(6),
        feature_vector=sinusoidal_feature,
        reward_param=theta_reward,
        safety_param=theta_safety
    )
    return bandit
   
if __name__ == "__main__":
    bandit = get_sinusoidal_bandit()
    bandit.plot()