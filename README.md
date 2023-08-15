# Safe bandits 
This repo corresponds to the third chapter of Alex Cloud's [thesis](https://repository.lib.ncsu.edu/handle/1840.20/40248), on "Safety-constrained online learning in contextual bandits." The code is structured as follows:
* **Core functionality** is implemented in scripts prefixed by an underscore:
  * `_bandit_learning.py` - defines a variety of safe and unsafe bandit learning algorithms, as well as the `evaluate()` function for running the algorithm-environment interaction for an experiment;
  * `_BanditEnv.py` - defines the `BanditEnv` class for representing contextual bandit problems; includes getter functions to construct bandit problems with different properties;
  * `_utils.py` - contains short, resuable functions like `linear_regression()`;
  * `_visualize_results.py` - tools for producing plots, and serializing/deserializing experiment data.
* **Experiment configuration** given by two kinds of files in the `experiments` folder:
  *  Learning algorithm configuration (scripts prefixed by `algorithms_`)
  *  Bandit environment configuration (all other scripts)
* **To run an experiment,** execute `experiment_driver.py`, which will use the specified experiment configurations and call the `experiment_worker.py` to make repeated calls to `_bandit_learning.evaluate()`.
* **One-off scripts** with no dependencies live in the `standalone` folder.
