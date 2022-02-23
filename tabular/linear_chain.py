import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import pdb

from mdp_experiments import LinearChain, Agent, run_experiment
from environments import P, r, pi, tabular_features, inverted_features, \
    dependent_features, single_features, P_mess, r_mess, r_dense


num_runs = 150
num_episodes = 100

policy_weight_init_left = 0
policy_weight_init_right = 0
value_weight_init = +4

grad_bias = None
grad_noise = None

start_state = 3
terminal_states = [0, 6]
reward_noise = 1

num_actions = 2
policy_features = tabular_features
gamma = 0.9
episode_cutoff_length = 100

stepsize_list = [2**i for i in range(-6, 2)]
critic_stepsize_list = [2**i for i in range(-4, 2)]

nstep = 'inf'

FLAG_BASELINE = True

# folder details
folder_name = 'mdp_data_biased/linearChain__thetaInit_{}_{}__valueWeightInit_{}'\
    '__numRuns_{}__numEpisodes_{}__startState_{}__rewardNoise_{}__gamma_{}'\
    '__episodeCutoff_{}__gradBias_{}__gradNoise_{}'.format(
        policy_weight_init_left, policy_weight_init_right, value_weight_init,
        num_runs, num_episodes, start_state, reward_noise, gamma,
        episode_cutoff_length, grad_bias, grad_noise)
print(folder_name)

# folder details
os.makedirs(folder_name, exist_ok='True')

#----------------------------------------------------------------------
# Using true value function
#----------------------------------------------------------------------
FLAG_LEARN_VPI = False
value_features = None
tic = time.time()
for FLAG_PG_TYPE in ['expected', 'regular', 'alternate']:
    for policy_stepsize in stepsize_list:
        critic_stepsize = None
        dat = run_experiment(num_runs=num_runs, num_episodes=num_episodes,
                             P=P, r=r, start_state=start_state,
                             terminal_states=terminal_states,
                             num_actions=num_actions,
                             policy_features=policy_features,
                             value_features=value_features,
                             policy_stepsize=policy_stepsize,
                             critic_stepsize=critic_stepsize,
                             nstep=nstep, gamma=gamma,
                             FLAG_BASELINE=FLAG_BASELINE,
                             FLAG_LEARN_VPI=FLAG_LEARN_VPI,
                             FLAG_PG_TYPE=FLAG_PG_TYPE,
                             reward_noise=reward_noise, vpi_bias=0,
                             policy_weight_init_left=policy_weight_init_left,
                             policy_weight_init_right=policy_weight_init_right,
                             episode_cutoff_length=episode_cutoff_length,
                             value_weight_init=value_weight_init,
                             grad_bias=grad_bias, grad_noise=grad_noise)
        filename='{}/pg_{}__pol_{}__val_{}__learnVpi_{}'.format(
            folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
            FLAG_LEARN_VPI)
        with open(filename, 'w') as fp:
            json.dump(dat, fp)
    print('True Vpi', FLAG_PG_TYPE, time.time() - tic)

#----------------------------------------------------------------------
# Using a learned value function
#----------------------------------------------------------------------
FLAG_LEARN_VPI = True
value_features = tabular_features
for FLAG_PG_TYPE in ['regular', 'alternate']:
    for policy_stepsize in stepsize_list:
        for critic_stepsize in critic_stepsize_list:
            dat = run_experiment(
                num_runs=num_runs, num_episodes=num_episodes,
                P=P, r=r, start_state=start_state,
                terminal_states=terminal_states,
                num_actions=num_actions,
                policy_features=policy_features,
                value_features=value_features,
                policy_stepsize=policy_stepsize,
                critic_stepsize=critic_stepsize,
                nstep=nstep, gamma=gamma,
                FLAG_BASELINE=FLAG_BASELINE,
                FLAG_LEARN_VPI=FLAG_LEARN_VPI,
                FLAG_PG_TYPE=FLAG_PG_TYPE,
                reward_noise=reward_noise, vpi_bias=0,
                policy_weight_init_left=policy_weight_init_left,
                policy_weight_init_right=policy_weight_init_right,
                episode_cutoff_length=episode_cutoff_length,
                value_weight_init=value_weight_init,
                grad_bias=grad_bias, grad_noise=grad_noise)
            filename='{}/pg_{}__pol_{}__val_{}__learnVpi_{}'.format(
                folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
                FLAG_LEARN_VPI)
            with open(filename, 'w') as fp:
                json.dump(dat, fp)
    print('Learned Vpi', FLAG_PG_TYPE, time.time() - tic)


# #----------------------------------------------------------------------
# # Using a fixed value function
# #----------------------------------------------------------------------
# tic = time.time()
# FLAG_LEARN_VPI = False
# critic_stepsize = None
# value_features = tabular_features
# for FLAG_PG_TYPE in ['regular', 'alternate']:
#     for policy_stepsize in stepsize_list:
#         dat = run_experiment(num_runs=num_runs, num_episodes=num_episodes,
#                              P=P, r=r, start_state=start_state,
#                              terminal_states=terminal_states,
#                              num_actions=num_actions,
#                              policy_features=policy_features,
#                              value_features=value_features,
#                              policy_stepsize=policy_stepsize,
#                              critic_stepsize=critic_stepsize,
#                              nstep=nstep, gamma=gamma,
#                              FLAG_BASELINE=FLAG_BASELINE,
#                              FLAG_LEARN_VPI=FLAG_LEARN_VPI,
#                              FLAG_PG_TYPE=FLAG_PG_TYPE,
#                              reward_noise=reward_noise, vpi_bias=0,
#                              policy_weight_init_left=policy_weight_init_left,
#                              policy_weight_init_right=policy_weight_init_right,
#                              episode_cutoff_length=episode_cutoff_length,
#                              value_weight_init=value_weight_init,
#                              grad_bias=grad_bias, grad_noise=grad_noise,
#                              FLAG_FIXED_VPI_OPTIM=True)
#         filename='{}/pg_{}__pol_{}__val_{}__learnVpi_{}_biased'.format(
#             folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
#             FLAG_LEARN_VPI)
#         with open(filename, 'w') as fp:
#             json.dump(dat, fp)
#     print('Fixed Vpi', FLAG_PG_TYPE, time.time() - tic)

    
#======================================================================
# Extra Code (primarily for testing)
#======================================================================
# policy_stepsize = 0.5
# critic_stepsize = None
# nstep = 'inf'
# FLAG_PG_TYPE = 'alternate'

# dat = run_experiment(num_runs, num_episodes,
#                      P, r, start_state, terminal_states,
#                      num_actions, policy_features,
#                      value_features,
#                      policy_stepsize, critic_stepsize,
#                      nstep, gamma, FLAG_BASELINE, FLAG_LEARN_VPI,
#                      FLAG_PG_TYPE, reward_noise=reward_noise)

# def plot_single_run(dat):
#     return_across_runs = np.array(dat['returns'])
#     mean_perf = return_across_runs.mean(0)
#     std_err = return_across_runs.std(0) / np.sqrt(num_runs)
#     plt.fill_between(range(num_episodes), mean_perf - std_err,
#                      mean_perf + std_err, alpha=0.5)
#     plt.plot(mean_perf)
#     plt.show()

# plot_single_run(dat)

# pdb.set_trace()




#----------------------------------------------------------------------
# env = LinearChain(P, r, start_state, terminal_states, reward_noise)


# policy_features = tabular_features
# policy_weight = np.zeros((policy_features.shape[1],
#                           num_actions))

# pi = np.array([[0.5, 0.5],  # s_0
#                [0.9, 0.1],  # s_1
#                [0.3, 0.7],  # s_2
#                [0.4, 0.6],  # s_3
#                [0.7, 0.3],  # s_4
#                [0.1, 0.9],  # s_5
#                [0.5, 0.5]]) # s_6

# d_gamma = env.calc_d_gamma(pi, gamma)[start_state]
# # P_pi_control: SxA -> SxA
# P_pi_control = np.concatenate([pi[:, a] * np.concatenate(P)
#                                for a in range(pi.shape[1])], 1)

# sa_visitation = np.linalg.inv(np.eye(P_pi_control[0].shape[0]) \
#                         - gamma * P_pi_control)
# r_sa = r.reshape(-1, 1, order='F')
# q_pi = np.matmul(sa_visitation, r_sa).reshape(-1, 2, order='F')
# v_pi = env.calc_v_pi(pi, gamma)

# def calc_grad_pi(state, action, FLAG_PG_TYPE='regular'):
#     x = policy_features[state].reshape(-1, 1)
#     # theta = np.matmul(x.T, policy_weight)
#     # pi = Agent.softmax(theta).T

#     I_action = np.zeros((num_actions, 1))
#     I_action[action] = 1

#     one_vec = np.ones((1, num_actions))

#     if FLAG_PG_TYPE == 'regular':
#         # grad = pi[action] * np.matmul(x, one_vec) * (I_action - pi).T
#         grad = pi[state][action] \
#             * np.matmul(x, (I_action - pi[state].reshape(-1, 1)).T) 
#     elif FLAG_PG_TYPE == 'alternate':
#         # grad = pi[action] * np.matmul(x, one_vec) * (I_action).T
#         grad = pi[state][action] \
#             * np.matmul(x, I_action.T) # equivalent

#     return grad

# grad_expected = np.zeros((policy_features.shape[1], num_actions))
# # theta = np.matmul(policy_features, policy_weight)
# # pi = Agent.softmax(theta)
# for s in range(P[0].shape[0]):
#     grad_expected += (1 - gamma) * d_gamma[s] * np.matmul(
#         policy_features[s].reshape(-1, 1),
#         (pi[s] * (q_pi[s] - v_pi[s])).reshape(1, -1))

# grad_regular = np.zeros((policy_features.shape[1], num_actions))
# for s in range(P[0].shape[0]):
#     tmp = np.zeros((policy_features.shape[1], num_actions))
#     for a in range(num_actions):
#         tmp += calc_grad_pi(s, a, 'regular') * (q_pi[s, a] - v_pi[s])
#     grad_regular += (1 - gamma) * d_gamma[s] * tmp

# grad_alternate = np.zeros((policy_features.shape[1], num_actions))
# for s in range(P[0].shape[0]):
#     tmp = np.zeros((policy_features.shape[1], num_actions))
#     for a in range(num_actions):
#         tmp += calc_grad_pi(s, a, 'alternate') * (q_pi[s, a] - v_pi[s])
#     grad_alternate += (1 - gamma) * d_gamma[s] * tmp

# # for s in range(P[0].shape[0]):
# #     x = policy_features[s].reshape(-1, 1)

# #     expected = np.matmul(x, (pi[s] * (q_pi[s] - v_pi[s])).reshape(1, -1))
# #     regular = np.zeros((policy_features.shape[1], num_actions))
# #     alternate = np.zeros((policy_features.shape[1], num_actions))
# #     for a in range(num_actions):
# #         regular += calc_grad_pi(s, a, 'regular') * (q_pi[s, a] - v_pi[s])
# #         alternate += calc_grad_pi(s, a, 'alternate') * (q_pi[s, a] - v_pi[s])

# #     print('expected\n', expected,
# #       '\nregular\n', regular,
# #       '\nalternate\n', alternate)


# print('expected\n', grad_expected,
#       '\nregular\n', grad_regular,
#       '\nalternate\n', grad_alternate)


# agent = Agent(num_actions, policy_features, value_features,
#               policy_stepsize=0, critic_stepsize=0, nstep='inf', gamma=0.99,
#               FLAG_BASELINE=True, FLAG_PG_TYPE='expected')

# expected_pg = agent.calc_expected_pg(env, start_state, method='expected')
# regular_pg = agent.calc_expected_pg(env, start_state, method='regular')
# alternate_pg = agent.calc_expected_pg(env, start_state, method='alternate')

# print('-------------------------------------------------------------\n')
# print('expected\n', expected_pg,
#       '\nregular\n', regular_pg,
#       '\nalternate\n', alternate_pg)

# pdb.set_trace()

# #----------------------------------------------------------------------
