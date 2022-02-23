import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import pdb
import time
import os
import pickle

import bandit
from tools import run_exp

#----------------------------------------------------------------------
# hyperparameters
#----------------------------------------------------------------------

num_arms = 3
num_runs = 150
num_iter = 1000
true_reward = [1, 2, 3]
action_pref_init = [20, 0, 0]
baseline_init = +4

reward_noise = 1
grad_bias = None
grad_noise = None
alpha_list = [2**i for i in range(-6, 2)]
beta_list = [2**i for i in range(-4, 1)]

grad_noise_list = [0.0, 0.1, 0.5, 1.0, 2.0]

# folder details
folder_name = 'bandit_numArms{}_data/reward_{}_{}_{}__actionPrefInit_{}_{}_{}'\
    '__baselineInit_{}__numRuns{}__numIter{}__rewardNoise_{}'\
    '__gradBias_{}__gradNoise_{}'.format(
        num_arms, true_reward[0], true_reward[1], true_reward[2],
        action_pref_init[0], action_pref_init[1], action_pref_init[2],
        baseline_init, num_runs, num_iter, reward_noise, grad_bias, grad_noise)
os.makedirs(folder_name, exist_ok='True')

print('Expmt -- reward:{} | theta_init:{}'.format(true_reward, action_pref_init))
#----------------------------------------------------------------------
# main experiment
#----------------------------------------------------------------------
bandit_env = bandit.BatchBandit(
    num_actions=num_arms, mean=None, std=None, reward_noise=reward_noise,
    random_walk_std=0, batch_size=num_runs, true_reward_list=true_reward)

tic0 = tic = time.time()

#----------------------------------------------------------------------
# expected pg
#----------------------------------------------------------------------
# for alpha in alpha_list:
#     agent = bandit.BatchGradientBanditAgent(
#         num_actions=num_arms, alpha=alpha, beta=None,
#         action_pref_init=action_pref_init,
#         TYPE='expected_pg', eta_flag=None, baseline_flag=None,
#         batch_size=num_runs, true_reward=bandit_env.true_reward,
#         baseline_init=baseline_init, grad_bias=None, grad_noise=None)

#     dat = run_exp(bandit_env, agent, num_iter, num_capture=num_runs)
#     pickle_filename = '{}/expectedPG__alpha_{}'.format(
#         folder_name, alpha)
#     with open(pickle_filename, 'wb') as handle:
#         pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)

#----------------------------------------------------------------------
# stochastic pg: (regular, alternate) + (rpi, learned)
#----------------------------------------------------------------------
# for eta_flag in ['pi', 'zero']:
#     for alpha in alpha_list:
#         baseline_flag = 'learned'
#         for beta in beta_list:
#             agent = bandit.BatchGradientBanditAgent(
#                 num_actions=num_arms, alpha=alpha, beta=beta,
#                 action_pref_init=action_pref_init,                    
#                 TYPE='sample_based', eta_flag=eta_flag,
#                 baseline_flag=baseline_flag, batch_size=num_runs,
#                 true_reward=bandit_env.true_reward,
#                 baseline_init=baseline_init,
#                 grad_bias=grad_bias, grad_noise=grad_noise)

#             dat = run_exp(bandit_env, agent, num_iter, num_capture=num_runs)
#             filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'.format(
#                 folder_name, eta_flag, baseline_flag, alpha, beta)
#             with open(filename, 'wb') as handle:
#                 pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
#         baseline_flag = 'rpi'
#         beta = None
#         agent = bandit.BatchGradientBanditAgent(
#             num_actions=num_arms, alpha=alpha, beta=beta,
#             action_pref_init=action_pref_init,                    
#             TYPE='sample_based', eta_flag=eta_flag,
#             baseline_flag=baseline_flag, batch_size=num_runs,
#             true_reward=bandit_env.true_reward,
#             baseline_init=baseline_init,
#             grad_bias=grad_bias, grad_noise=grad_noise)

#         dat = run_exp(bandit_env, agent, num_iter, num_capture=num_runs)
#         filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'.format(
#             folder_name, eta_flag, baseline_flag, alpha, beta)
#         with open(filename, 'wb') as handle:
#             pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     print('------------------------------------------')
#     print(alpha, time.time() - tic)
#     tic = time.time()
# print('\ntotal time taken:', time.time() - tic0)


#----------------------------------------------------------------------
# stochastic pg: (regular, alternate) + (fixed)
#----------------------------------------------------------------------
# for eta_flag in ['pi', 'zero']:
#     for alpha in alpha_list:
#         baseline_flag = 'fixed'
#         beta = None
#         agent = bandit.BatchGradientBanditAgent(
#             num_actions=num_arms, alpha=alpha, beta=beta,
#             action_pref_init=action_pref_init,                    
#             TYPE='sample_based', eta_flag=eta_flag,
#             baseline_flag=baseline_flag, batch_size=num_runs,
#             true_reward=bandit_env.true_reward,
#             baseline_init=baseline_init,
#             grad_bias=grad_bias, grad_noise=grad_noise)
        
#         dat = run_exp(bandit_env, agent, num_iter, num_capture=num_runs)
#         filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'.format(
#             folder_name, eta_flag, baseline_flag, alpha, beta)
#         with open(filename, 'wb') as handle:
#             pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     print('------------------------------------------')
#     print(alpha, time.time() - tic)
#     tic = time.time()
# print('\ntotal time taken:', time.time() - tic0)


#----------------------------------------------------------------------
# additional gradient noise
#----------------------------------------------------------------------
grad_bias = 0
for grad_noise in grad_noise_list:
    for alpha in alpha_list:
        agent = bandit.BatchGradientBanditAgent(
            num_actions=num_arms, alpha=alpha, beta=None,
            action_pref_init=action_pref_init,
            TYPE='expected_pg', eta_flag=None, baseline_flag=None,
            batch_size=num_runs, true_reward=bandit_env.true_reward,
            baseline_init=baseline_init,
            grad_bias=grad_bias, grad_noise=grad_noise)

        dat = run_exp(bandit_env, agent, num_iter, num_capture=num_runs)
        pickle_filename = '{}/expectedPG__alpha_{}__gradBias_{}'\
            '__gradNoise_{}'.format(folder_name, alpha, grad_bias, grad_noise)
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for eta_flag in ['pi', 'zero']:
        for alpha in alpha_list:
            baseline_flag = 'learned'
            for beta in beta_list:
                agent = bandit.BatchGradientBanditAgent(
                    num_actions=num_arms, alpha=alpha, beta=beta,
                    action_pref_init=action_pref_init,                    
                    TYPE='sample_based', eta_flag=eta_flag,
                    baseline_flag=baseline_flag, batch_size=num_runs,
                    true_reward=bandit_env.true_reward,
                    baseline_init=baseline_init,
                    grad_bias=grad_bias, grad_noise=grad_noise)

                dat = run_exp(bandit_env, agent, num_iter, num_capture=num_runs)
                filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'\
                    '__gradBias_{}__gradNoise_{}'.format(
                        folder_name, eta_flag, baseline_flag, alpha, beta,
                        grad_bias, grad_noise)
                with open(filename, 'wb') as handle:
                    pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)

            baseline_flag = 'rpi'
            beta = None
            agent = bandit.BatchGradientBanditAgent(
                num_actions=num_arms, alpha=alpha, beta=beta,
                action_pref_init=action_pref_init,                    
                TYPE='sample_based', eta_flag=eta_flag,
                baseline_flag=baseline_flag, batch_size=num_runs,
                true_reward=bandit_env.true_reward,
                baseline_init=baseline_init,
                grad_bias=grad_bias, grad_noise=grad_noise)

            dat = run_exp(bandit_env, agent, num_iter, num_capture=num_runs)
            filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'\
                '__gradBias_{}__gradNoise_{}'.format(
                    folder_name, eta_flag, baseline_flag, alpha, beta,
                    grad_bias, grad_noise)
            with open(filename, 'wb') as handle:
                pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('------------------------------------------')
        print(alpha, time.time() - tic)
        tic = time.time()
    print('\ntotal time taken:', time.time() - tic0)
