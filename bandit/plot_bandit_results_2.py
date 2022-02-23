import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb
import pickle

import tools

#----------------------------------------------------------------------
# plotting_details
#----------------------------------------------------------------------
num_arms = 3
num_runs = 150
num_iter = 1000
true_reward = [1, 2, 3]
action_pref_init = [0, 0, 0]
baseline_init = 0

reward_noise = 1
grad_bias = None
grad_noise = None
alpha_list = [2**i for i in range(-6, 2)]
beta_list = [2**i for i in range(-4, 1)]

action_pref_init_list = [[0, 0, 0],
                         [5, 0, 0],
                         [10, 0, 0],
                         [50, 0, 0]]

grad_noise_list = [0.0, 0.1, 0.5, 1.0, 2.0]

end_plotting_iter = 1000
start_plotting_iter = end_plotting_iter - 50

print('Plot -- reward:{} | theta_init:{}'.format(true_reward, action_pref_init))

# folder details
folder_name = 'bandit_numArms{}_data/reward_{}_{}_{}__actionPrefInit_{}_{}_{}'\
    '__baselineInit_{}__numRuns{}__numIter{}__rewardNoise_{}'\
    '__gradBias_{}__gradNoise_{}'.format(
        num_arms, true_reward[0], true_reward[1], true_reward[2],
        action_pref_init[0], action_pref_init[1], action_pref_init[2],
        baseline_init, num_runs, num_iter, reward_noise,
        grad_bias, grad_noise)

#----------------------------------------------------------------------
# Regular (no baseline) plots
#----------------------------------------------------------------------
# fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharey=True)
# for idx, action_pref_init in zip(range(len(action_pref_init_list)),
#                                  action_pref_init_list):
#     # folder details
#     folder_name = 'bandit_numArms{}_data/reward_{}_{}_{}'\
#         '__actionPrefInit_{}_{}_{}__baselineInit_{}__numRuns{}'\
#         '__numIter{}__rewardNoise_{}__gradBias_{}__gradNoise_{}'.format(
#             num_arms, true_reward[0], true_reward[1], true_reward[2],
#             action_pref_init[0], action_pref_init[1], action_pref_init[2],
#             baseline_init, num_runs, num_iter, reward_noise,
#             grad_bias, grad_noise)
#     # stochastic PG
#     for eta_flag in ['pi', 'zero']:
#         baseline_flag = 'fixed'
#         plt_color = 'red' if eta_flag == 'pi' else 'blue'
#         ax = axs[0] if eta_flag == 'pi' else axs[1]

#         c1 = np.array(mpl.colors.to_rgb(plt_color))
#         c2 = np.array(mpl.colors.to_rgb('white'))
#         mix = (idx + 1) / (len(action_pref_init_list) + 2)
#         plt_color = mpl.colors.to_hex((1 - mix) * c1 + mix * c2)
        
#         tools.plot_alpha_sensitivity_2(ax=ax, folder_name=folder_name,
#                                        alpha_list=alpha_list,
#                                        pg_type='stochastic_pg',
#                                        eta_flag=eta_flag,
#                                        baseline_flag=baseline_flag, beta=None,
#                                        plt_color=plt_color, linewidth=1,
#                                        start_plotting_iter=start_plotting_iter,
#                                        end_plotting_iter=end_plotting_iter,
#                                        action_pref_init=action_pref_init)
 
# axs[0].spines['right'].set_visible(False)
# axs[0].spines['top'].set_visible(False)
# axs[1].spines['right'].set_visible(False)
# axs[1].spines['top'].set_visible(False)

# axs[0].legend()
# axs[1].legend()
# axs[0].set_ylabel(r'$\mathcal{J}_\pi$' + ' (PG objective)')
# axs[1].set_xlabel(r'$2^\alpha$')

# plt.savefig('{}_{}__No_baseline___fixed_sensitivity.pdf'.format(
#     folder_name, end_plotting_iter))
# plt.close()


##----------------------------------------------------------------------
## Sensitivity plots for alpha (gradient noise plots)
##----------------------------------------------------------------------
fig, axs = plt.subplots(1, 6, figsize=(20, 3), sharey=True)

grad_bias = 0
for grad_noise, tmp in zip(grad_noise_list, range(len(grad_noise_list))):
    # expected PG
    c1 = np.array(mpl.colors.to_rgb('black'))
    c2 = np.array(mpl.colors.to_rgb('white'))
    mix = (tmp + 1) / (len(grad_noise_list) + 2)
    plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)
    tools.plot_alpha_sensitivity_3(ax=axs[0], folder_name=folder_name,
                                   alpha_list=alpha_list,
                                   pg_type='expected_pg', eta_flag=None,
                                   baseline_flag=None, beta=None,
                                   plt_color=plt_color, linewidth=1,
                                   start_plotting_iter=start_plotting_iter,
                                   end_plotting_iter=end_plotting_iter,
                                   grad_bias=grad_bias, grad_noise=grad_noise)
    # stochastic PG
    for eta_flag in ['pi', 'zero']:
        baseline_flag = 'rpi'
        plt_color = 'tab:red' if eta_flag == 'pi' else 'tab:blue'
        c1 = np.array(mpl.colors.to_rgb(plt_color))
        c2 = np.array(mpl.colors.to_rgb('white'))
        mix = (tmp + 1) / (len(grad_noise_list) + 2)
        plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)
        tools.plot_alpha_sensitivity_3(ax=axs[0], folder_name=folder_name,
                                       alpha_list=alpha_list,
                                       pg_type='stochastic_pg',
                                       eta_flag=eta_flag,
                                       baseline_flag=baseline_flag, beta=None,
                                       plt_color=plt_color, linewidth=1,
                                       start_plotting_iter=start_plotting_iter,
                                       end_plotting_iter=end_plotting_iter,
                                       grad_bias=grad_bias,
                                       grad_noise=grad_noise)
    
        baseline_flag = 'learned'
        for beta, tmp_small in zip(beta_list, range(len(beta_list))):
            c1 = 'red' if eta_flag == 'pi' else 'blue'
            c1 = np.array(mpl.colors.to_rgb(c1))
            c2 = np.array(mpl.colors.to_rgb('white'))
            mix = (tmp_small + 1) / (len(beta_list) + 2)
            plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)
            tools.plot_alpha_sensitivity_3(
                ax=axs[tmp + 1], folder_name=folder_name, alpha_list=alpha_list,
                pg_type='stochastic_pg', eta_flag=eta_flag,
                baseline_flag=baseline_flag, beta=beta, plt_color=plt_color,
                linewidth=1, start_plotting_iter=start_plotting_iter,
                end_plotting_iter=end_plotting_iter,
                grad_bias=grad_bias, grad_noise=grad_noise)

for i in range(6):
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
# axs[0].legend()
# axs[1].legend()

axs[0].set_ylabel(r'$\mathcal{J}_\pi$' + ' (PG objective)')
axs[1].set_xlabel(r'$2^\alpha$')

plt.savefig('{}_{}__noisyyyyy.pdf'.format(folder_name, end_plotting_iter))
plt.close()


##----------------------------------------------------------------------
## Alternate (biased) = fixed baseline with optimistic / pessimistic
##----------------------------------------------------------------------
# fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharey=True)
# for idx, action_pref_init in zip(range(len(action_pref_init_list)),
#                                  action_pref_init_list):
#     # folder details
#     folder_name = 'bandit_numArms{}_data/reward_{}_{}_{}'\
#         '__actionPrefInit_{}_{}_{}__baselineInit_{}__numRuns{}'\
#         '__numIter{}__rewardNoise_{}__gradBias_{}__gradNoise_{}'.format(
#             num_arms, true_reward[0], true_reward[1], true_reward[2],
#             action_pref_init[0], action_pref_init[1], action_pref_init[2],
#             baseline_init, num_runs, num_iter, reward_noise,
#             grad_bias, grad_noise)
#     # stochastic PG
#     for eta_flag in ['pi', 'zero']:
#         baseline_flag = 'fixed'
#         plt_color = 'red' if eta_flag == 'pi' else 'blue'
        
#         c1 = np.array(mpl.colors.to_rgb(plt_color))
#         c2 = np.array(mpl.colors.to_rgb('white'))
#         mix = (idx + 1) / (len(action_pref_init_list) + 2)
#         plt_color = mpl.colors.to_hex((1 - mix) * c1 + mix * c2)
        
#         tools.plot_alpha_sensitivity_2(ax=ax, folder_name=folder_name,
#                                        alpha_list=alpha_list,
#                                        pg_type='stochastic_pg',
#                                        eta_flag=eta_flag,
#                                        baseline_flag=baseline_flag, beta=None,
#                                        plt_color=plt_color, linewidth=1,
#                                        start_plotting_iter=start_plotting_iter,
#                                        end_plotting_iter=end_plotting_iter,
#                                        action_pref_init=action_pref_init)
 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# ax.legend()
# ax.set_ylabel(r'$\mathcal{J}_\pi$' + ' (PG objective)')
# ax.set_xlabel(r'$2^\alpha$')

# plt.savefig('{}_{}__baselineInit_{}___fixed_sensitivity.pdf'.format(
#     folder_name, end_plotting_iter, baseline_init))

# plt.close()
