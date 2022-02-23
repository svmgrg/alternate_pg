import numpy as np
import pdb

from mdp_experiments import LinearChain, Agent, run_experiment
from environments import P, r, pi, tabular_features, inverted_features, \
    dependent_features, single_features, P_mess, r_mess

start_state = 3
terminal_states = [0, 6]
reward_noise = 1

env1 = LinearChain(P, r, start_state, terminal_states)
env2 = LinearChain(P_mess, r_mess, start_state, terminal_states)

pi0 = np.array([[0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1]])

pi_random = np.array([[0.5, 0.5],  # s_0
                      [0.5, 0.5],  # s_1
                      [0.5, 0.5],  # s_2
                      [0.5, 0.5],  # s_3
                      [0.5, 0.5],  # s_4
                      [0.5, 0.5],  # s_5
                      [0.5, 0.5]]) # s_6
    
pi1 = np.array([[0.5, 0.5],  # s_0
                [0.75, 0.25],  # s_1
                [0.75, 0.25],  # s_2
                [0.75, 0.25],  # s_3
                [0.75, 0.25],  # s_4
                [0.75, 0.25],  # s_5
                [0.5, 0.5]]) # s_6

pi2 = np.array([[0.25, 0.25, 0.25, 0.25],  # s_0
                [0.25, 0.25, 0.25, 0.25],  # s_1
                [0.25, 0.25, 0.25, 0.25],  # s_2
                [0.25, 0.25, 0.25, 0.25],  # s_3
                [0.25, 0.25, 0.25, 0.25],  # s_4
                [0.25, 0.25, 0.25, 0.25],  # s_5
                [0.25, 0.25, 0.25, 0.25]]) # s_6

pi3 = np.array([[0.5, 0.5],  # s_0
                [0.9, 0.1],  # s_1
                [0.4, 0.6],  # s_2
                [0.01, 0.99],  # s_3
                [0.1, 0.9],  # s_4
                [0.75, 0.25],  # s_5
                [0.5, 0.5]]) # s_6

pi4 = np.array([[0.25, 0.25, 0.25, 0.25],  # s_0
                [0.2, 0.2, 0.5, 0.1],  # s_1
                [0.0, 0.2, 0.2, 0.6],  # s_2
                [0.01/3, 0.01/3, 0.01/3, 0.99],  # s_3
                [0.0, 0.05, 0.05, 0.9],  # s_4
                [0.25, 0.25, 0.25, 0.25],  # s_5
                [0.25, 0.25, 0.25, 0.25]]) # s_6

reward = np.array([0, 1])
baseline = +4
pi_star_s5 = (1 / (reward - baseline)) / (1 / (reward - baseline)).sum()
pi_star = 0.5 * np.ones((7, 2))
pi_star[5] = pi_star_s5
print(pi_star)

gamma = 0.9

env1.calc_v_pi(pi_star, gamma)
