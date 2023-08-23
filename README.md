# Safe bandits
A framework for online learning in the constrained linear contextual bandit setting. The code is structured as follows:
* **Core functionality** is implemented in scripts prefixed by an underscore:
  * `_bandit_learning.py` - defines a variety of "safe" and "unsafe" bandit learning algorithms, as well as the `evaluate()` function for running algorithm-environment interactions and recording the results;
  * `_BanditEnv.py` - defines the `BanditEnv` class for representing contextual bandit problems; includes getter functions to construct bandit problems with different properties;
  * `_utils.py` - defines short, resuable functions like `linear_regression()`;
  * `_visualize_results.py` - tools for producing plots, and serializing/deserializing experiment data.
* **Experiment configuration** is done by two kinds of files in the `experiments` folder:
  *  Learning algorithm configuration (scripts prefixed by `algorithms_`)
  *  Bandit environment configuration (all other scripts)
* **To run an experiment,** execute `experiment_driver.py`, which will use the specified experiment configurations and call the `experiment_worker.py` to make repeated calls to `_bandit_learning.evaluate()`.
* **One-off scripts** with no dependencies live in the `standalone` folder.

The framework was used for simulation experiments in chapter 4 of [my thesis](https://repository.lib.ncsu.edu/handle/1840.20/40248). I described the problem setting as:
>... a constrained reinforcement learning problem, where in addition to reward maximization, a decision maker must also select actions according to a constraint on their “safety.” Constraint satisfaction, like the underlying reward signal, is estimated from noisy data and thus requires careful handling of uncertainty.
