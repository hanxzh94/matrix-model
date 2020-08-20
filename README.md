# Deep Quantum Geometry of Matrices
Here is the implementation of the variational quantum Monte Carlo for low-energy states in mini-BMN matrix models. The wavefunction ansatz consists of generative flows for bosons and superposition of free states for fermions. For more details please see arXiv:1906.08781.

## Setup
I recommend Anaconda (https://www.anaconda.com/) for environment setup. Please install Anaconda from the website and create an environment as follows:

  1. Run the command `conda create -n tf tensorflow=1.13 tensorflow-probability=0.6.0 matplotlib` to create an environment named `tf` (feel free to use other names);
  2. Activate the environment by `conda activate tf`;
  3. There are several bugs in implementations of gradients for complex matrices in TensorFlow 1.13. As a quick fix, run `pip show tensorflow` to see where the TensorFlow packages are located, and replace `tensorflow/python/ops/linalg_grad.py` in the TensorFlow directory by the copy in this folder;
  4. I recommend running the tests `python tests.py` to make sure that the environment is setup properly. All tests should pass (in ~1h) and you will see `OK` after the command finishes. 
  
## Demo
Demonstration code is in `demo.py`; please check out the source code for its arguments and usage. As an example, the command `python demo.py maf 2 1.0 -f 2` will search for the ground state of the `N = 2` and `nu = 1.0` mini-BMN with `-f 2` fermions, using Masked Autoregressive Flows.

## Structure of the repository
* `algebra.py` is utility code for Lie groups and algebras
* `bent_identity.py` includes an implementation of the bent-identity nonlinearity
* `demo.py` is for demonstration
* `dist.py` includes implementations of probability distributions
* `ent.py` is used to compute entanglement of the wavefunction
* `linalg_grad.py` should be replaced into the TensorFlow library
* `obs.py` includes observables and hamiltonians
* `tests.py` is for unit tests
* `train.py` includes procedures for training neural networks
* `wavefunc.py` implements wavefunction ansatz
* `fig/` includes temporary files of plots for probability distributions from tests
* `results/` is where trained parameters are stored in files
* `data/` is for additional data files (spin matrices etc)
