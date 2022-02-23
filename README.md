# alternate_pg

Code for the paper An Alternate Policy Gradient Estimator for Softmax Policies (https://arxiv.org/abs/2112.11622) published at AISTATS 2022.

Different settings have different codes (all require Numpy, Scipy, matplotlib):
- bandits (3 armed bandit testbed with normal noise; also contains code for plotting the policy update directions on the policy simplex)
- tabular (linear chain with REINFORCE; involves exact gradients)
- linear (online AC with linear function approximation (+ tilecoding) with softmax and escort transform; also entropy regularization; requires additional files for running the environments and tilecode --- look up the help file in the folder)
- neural (online AC with neural networks; also contains the DotReacher environment; requires PyTorch)