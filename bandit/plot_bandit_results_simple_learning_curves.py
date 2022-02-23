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
true_reward = [0, 0, 1]
action_pref_init = [10, 0, 0]
baseline_init = 0

reward_noise = 1
grad_bias = None
grad_noise = None
alpha_list = [2**i for i in range(-6, 2)]
beta_list = [2**i for i in range(-4, 1)]

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
# Find the best hyper-parameter configuration
#----------------------------------------------------------------------
best_param_dict = dict()

# expected PG
best_param_dict['expected_pg'] = None
max_return_alpha = -1 * np.inf
for alpha in alpha_list:
    filename = '{}/expectedPG__alpha_{}'.format(folder_name, alpha)
    with open(filename, 'rb') as handle:
            dat = pickle.load(handle)
    return_alpha = dat['list_rpi'][
        start_plotting_iter:end_plotting_iter].mean()
    
    if return_alpha > max_return_alpha:
        max_return_alpha = return_alpha
        best_param_dict['expected_pg'] = alpha

# stochastic PG
for eta_flag in ['pi', 'zero']:
    best_param_dict[eta_flag] = dict()

    baseline_flag = 'rpi'
    best_param_dict[eta_flag][baseline_flag] = None
    max_return_alpha = -1 * np.inf
    for alpha in alpha_list:
        beta = None
        filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'.format(
            folder_name, eta_flag, baseline_flag, alpha, beta)
        with open(filename, 'rb') as handle:
            dat = pickle.load(handle)
        return_alpha = dat['list_rpi'][
            start_plotting_iter:end_plotting_iter].mean()
        
        if return_alpha > max_return_alpha:
            max_return_alpha = return_alpha
            best_param_dict[eta_flag][baseline_flag] = alpha

    baseline_flag = 'learned'
    best_param_dict[eta_flag][baseline_flag] = dict()
    best_param_dict[eta_flag][baseline_flag]['optimal_alpha'] = None
    max_return_alpha = -1 * np.inf
    for alpha in alpha_list:
        best_param_dict[eta_flag][baseline_flag][alpha] = None
        max_return_beta = -1 * np.inf
        for beta in beta_list:
            filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'.format(
                folder_name, eta_flag, baseline_flag, alpha, beta)
            with open(filename, 'rb') as handle:
                dat = pickle.load(handle)
            return_beta = dat['list_rpi'][
                start_plotting_iter:end_plotting_iter].mean()
                
            if return_beta > max_return_beta:
                max_return_beta = return_beta
                best_param_dict[eta_flag][baseline_flag][alpha] = beta
                    
        if max_return_beta > max_return_alpha:
            max_return_alpha = max_return_beta
            best_param_dict[eta_flag][baseline_flag]['optimal_alpha'] = alpha

#----------------------------------------------------------------------
# Sensitivity plots for alpha
#----------------------------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharey=True)

# expected PG
tools.plot_alpha_sensitivity(ax=axs[0], folder_name=folder_name,
                             alpha_list=alpha_list,
                             pg_type='expected_pg', eta_flag=None,
                             baseline_flag=None, beta=None,
                             plt_color='black', linewidth=1,
                             start_plotting_iter=start_plotting_iter,
                             end_plotting_iter=end_plotting_iter)
# stochastic PG
for eta_flag in ['pi', 'zero']:
    baseline_flag = 'rpi'
    plt_color = 'tab:red' if eta_flag == 'pi' else 'tab:blue'
    tools.plot_alpha_sensitivity(ax=axs[0], folder_name=folder_name,
                                 alpha_list=alpha_list,
                                 pg_type='stochastic_pg', eta_flag=eta_flag,
                                 baseline_flag=baseline_flag, beta=None,
                                 plt_color=plt_color, linewidth=1,
                                 start_plotting_iter=start_plotting_iter,
                                 end_plotting_iter=end_plotting_iter)
    baseline_flag = 'learned'
    for beta, tmp in zip(beta_list, range(len(beta_list))):
        c1 = 'red' if eta_flag == 'pi' else 'blue'
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb('white'))
        mix = (tmp + 1) / (len(beta_list) + 2)
        plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)
        tools.plot_alpha_sensitivity(ax=axs[1], folder_name=folder_name,
                                     alpha_list=alpha_list,
                                     pg_type='stochastic_pg', eta_flag=eta_flag,
                                     baseline_flag=baseline_flag, beta=beta,
                                     plt_color=plt_color, linewidth=1,
                                     start_plotting_iter=start_plotting_iter,
                                     end_plotting_iter=end_plotting_iter)

axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)

# axs[0].legend()
# axs[1].legend()
# axs[0].set_ylabel(r'$\mathcal{J}_\pi$' + ' (PG objective)')
# axs[1].set_xlabel(r'$\log(\alpha)$')

plt.savefig('{}_{}__rpi_sensitivity.pdf'.format(folder_name, end_plotting_iter))
plt.close()
#----------------------------------------------------------------------
# Learning Curves (rpi, baseline, action preferences)
#----------------------------------------------------------------------
fig, axs = plt.subplots(1, 1, figsize=(5, 4), sharex=True)

# expected PG
alpha = best_param_dict['expected_pg']
tools.plot_learning_curves_simple(ax1=axs, folder_name=folder_name,
                                  pg_type='expected_pg', eta_flag=None,
                                  baseline_flag=None, alpha=alpha, beta=None)
# stochastic PG
for eta_flag in ['pi', 'zero']:
    baseline_flag = 'rpi'
    alpha = best_param_dict[eta_flag][baseline_flag]
    idx = 1 if eta_flag == 'pi' else 2
    tools.plot_learning_curves_simple(ax1=axs, folder_name=folder_name,
                                      pg_type='stochastic_pg', eta_flag=eta_flag,
                                      baseline_flag=baseline_flag,
                                      alpha=alpha, beta=None)
    baseline_flag = 'learned'
    alpha = best_param_dict[eta_flag][baseline_flag]['optimal_alpha']
    beta = best_param_dict[eta_flag][baseline_flag][alpha]
    idx = 3 if eta_flag == 'pi' else 4
    tools.plot_learning_curves_simple(ax1=axs, folder_name=folder_name,
                                      pg_type='stochastic_pg', eta_flag=eta_flag,
                                      baseline_flag=baseline_flag,
                                      alpha=alpha, beta=beta)

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
plt.legend()

# axs[0, 0].set_ylabel('Reward')
# axs[1, 0].set_ylabel('Action Preferences')
# axs[1, 2].set_xlabel('Timesteps')

plt.savefig('{}__learning_curves_simple.pdf'.format(folder_name))
plt.close()
exit()
#----------------------------------------------------------------------
# Trajectory plots for Expected PG
#----------------------------------------------------------------------
fig, axs = plt.subplots(2, 5, figsize=(18, 8))

i = 0
theta_alpha = best_param_dict['expected_pg']
pickle_filename = '{}/expected_pg_theta_{}'.format(
    folder_name, theta_alpha)
with open(pickle_filename, 'rb') as handle:
    dat = pickle.load(handle)
tools.plot_traj(axs[0][i], dat['list_prob'])
axs[0][i].set_title(
    'expected_pg \n theta_alpha:{}' \
    ' \n % optimal:{:.3f}'.format(
        theta_alpha,
        best_param_dict['expected_pg_optimal']))

list_theta = dat['list_theta']
axs[1][i].plot(list_theta[:, 0, :50], color='c',
               linewidth=1, alpha=0.2)
axs[1][i].plot(list_theta[:, 1, :50], color='k',
               linewidth=1, alpha=0.2)
axs[1][i].plot(list_theta[:, 2, :50], color='m',
               linewidth=1, alpha=0.2)


#----------------------------------------------------------------------
# Trajectory plots for Stochastic PGs
#----------------------------------------------------------------------
i = 1
for eta_flag in ['pi', 'zero']:
    for rpi_flag in ['learned', 'true']:
        theta_alpha = best_param_dict[eta_flag][rpi_flag]['theta_alpha']
        rpi_alpha = best_param_dict[eta_flag][rpi_flag][theta_alpha]
        pickle_filename = '{}/rpi_{}_eta_{}_rpi_{}_theta_{}'.format(
            folder_name, rpi_flag, eta_flag, rpi_alpha, theta_alpha)
        with open(pickle_filename, 'rb') as handle:
            dat = pickle.load(handle)
        tools.plot_traj(axs[0][i], dat['list_prob'])

        axs[0][i].set_title(
            'eta_flag:{} | rpi_flag:{} \n theta_alpha:{}' \
            ' \n rpi_alpha:{} \n % optimal:{:.3f}'.format(
                eta_flag, rpi_flag, theta_alpha, rpi_alpha,
                best_param_dict[eta_flag][rpi_flag]['optimal']))

        list_theta = dat['list_theta']
        axs[1][i].plot(list_theta[:, 0, :50], color='c',
                       linewidth=1, alpha=0.2)
        axs[1][i].plot(list_theta[:, 1, :50], color='k',
                       linewidth=1, alpha=0.2)
        axs[1][i].plot(list_theta[:, 2, :50], color='m',
                       linewidth=1, alpha=0.2)
        
        i = i + 1

axs[0][0].set_ylabel('Policy prob. trajectory')
axs[1][0].set_ylabel('Action pref. values')
plt.savefig('{}_{}_trajectory.pdf'.format(folder_name,
                                          end_plotting_iter), dpi=300)
