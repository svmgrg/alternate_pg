import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

num_runs = 150
num_episodes = 100

policy_weight_init_left = 3
policy_weight_init_right = 0
value_weight_init = 4

grad_bias = None
grad_noise = None

start_state = 3
terminal_states = [0, 6]
reward_noise = 1

num_actions = 2
gamma = 0.9
episode_cutoff_length = 100

policy_stepsize_list = [2**i for i in range(-6, 2)]
critic_stepsize_list = [2**i for i in range(-4, 2)]

end_plotting_episode = 10
start_plotting_episode = end_plotting_episode - 5

folder_name = 'mdp_data_dense_1/'\
    'linearChain__thetaInit_{}_{}__valueWeightInit_{}'\
    '__numRuns_{}__numEpisodes_{}__startState_{}__rewardNoise_{}__gamma_{}'\
    '__episodeCutoff_{}__gradBias_{}__gradNoise_{}'.format(
        policy_weight_init_left, policy_weight_init_right, value_weight_init,
        num_runs, num_episodes, start_state, reward_noise, gamma,
        episode_cutoff_length, grad_bias, grad_noise)
print(folder_name)

def plot_alpha_sensitivity(ax, folder_name, policy_stepsize_list, FLAG_PG_TYPE,
                           critic_stepsize, plt_color, linewidth,
                           start_plotting_epsiode, end_plotting_episode,
                           FLAG_LEARN_VPI):
    perf_mean_list = []
    perf_stderr_list = []
    for policy_stepsize in policy_stepsize_list:
        filename='{}/pg_{}__pol_{}__val_{}__learnVpi_{}'.format(
            folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
            FLAG_LEARN_VPI)
        with open(filename, 'r') as fp:
            dat = json.load(fp)
        dat_return = np.array(dat['vpi_s0'])

        num_runs = dat_return.shape[0]
        ep_len = dat_return.shape[1]
        final_perf = dat_return[
            :, start_plotting_episode:end_plotting_episode].mean(1)

        perf_mean_list.append(final_perf.mean())
        perf_stderr_list.append(final_perf.std() / np.sqrt(num_runs))

    ax.errorbar(np.log(policy_stepsize_list) / np.log(2), perf_mean_list,
                yerr=perf_stderr_list, color=plt_color,
                label='{}_{}'.format(FLAG_PG_TYPE, critic_stepsize),
                linewidth=linewidth)
    ax.scatter(0, 0.85, s=0) # add an invisible point to make scaling from 0 -> 1

def plot_learning_curves(ax, folder_name, FLAG_PG_TYPE, policy_stepsize,
                         critic_stepsize, FLAG_LEARN_VPI, plt_color):
    filename='{}/pg_{}__pol_{}__val_{}__learnVpi_{}'.format(
        folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
        FLAG_LEARN_VPI)
    with open(filename, 'r') as fp:
        dat = json.load(fp)

    dat_return = np.array(dat['vpi_s0'])[:, :100]
    dat_mean = dat_return.mean(0)
    dat_err = dat_return.std(0) / np.sqrt(dat_return.shape[0])

    ax.plot(np.arange(dat_mean.shape[0]), dat_mean,
            linewidth=1.5, color=plt_color,
            label='{}_{}_{}'.format(
                FLAG_PG_TYPE, policy_stepsize, critic_stepsize))
    ax.fill_between(np.arange(dat_mean.shape[0]),
                    dat_mean - dat_err, dat_mean + dat_err,
                    alpha=0.2, facecolor=plt_color)
    # ax.plot(np.arange(dat_mean.shape[0]), dat_return.T, linewidth=0.1,
    #         alpha=0.1, color=plt_color)
    ax.scatter(0, 0.85, s=0) # add an invisible point to make scaling from 0 -> 1
        

#----------------------------------------------------------------------
# Find the best hyper-parameter configuration
#----------------------------------------------------------------------
best_param_dict = dict()
# expected PG
FLAG_PG_TYPE = 'expected'
best_param_dict[FLAG_PG_TYPE] = None
max_return_alpha = -1 * np.inf
critic_stepsize = None
FLAG_LEARN_VPI = False
for policy_stepsize in policy_stepsize_list:
    filename='{}/pg_{}__pol_{}__val_{}__learnVpi_{}'.format(
        folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
        FLAG_LEARN_VPI)
    with open(filename, 'r') as fp:
        dat = json.load(fp)
    dat_return = np.array(dat['vpi_s0'])
    final_performance = dat_return[
        :, start_plotting_episode:end_plotting_episode].mean()
    if final_performance > max_return_alpha:
        best_param_dict['expected'] = policy_stepsize
        max_return_alpha = final_performance

# stochastic PG
for FLAG_PG_TYPE in ['regular', 'alternate']:
    best_param_dict[FLAG_PG_TYPE] = dict()
    best_param_dict[FLAG_PG_TYPE]['true_vpi'] = None
    max_return_alpha = -1 * np.inf
    critic_stepsize = None
    FLAG_LEARN_VPI = False
    for policy_stepsize in policy_stepsize_list:
        filename='{}/pg_{}__pol_{}__val_{}__learnVpi_{}'.format(
            folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
            FLAG_LEARN_VPI)
        with open(filename, 'r') as fp:
            dat = json.load(fp)
        dat_return = np.array(dat['vpi_s0'])
        final_performance = dat_return[
            :, start_plotting_episode:end_plotting_episode].mean()
        if final_performance > max_return_alpha:
            best_param_dict[FLAG_PG_TYPE]['true_vpi'] = policy_stepsize
            max_return_alpha = final_performance

    best_param_dict[FLAG_PG_TYPE]['learned_vpi'] = dict()
    best_param_dict[FLAG_PG_TYPE]['learned_vpi']['optimal_policy_stepsize'] \
        = None
    max_return_alpha = -1 * np.inf
    FLAG_LEARN_VPI = True
    for policy_stepsize in policy_stepsize_list:
        best_param_dict[FLAG_PG_TYPE]['learned_vpi'][policy_stepsize] = None
        max_return_beta = -1 * np.inf
        for critic_stepsize in critic_stepsize_list:
            filename='{}/pg_{}__pol_{}__val_{}__learnVpi_{}'.format(
                folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
                FLAG_LEARN_VPI)
            with open(filename, 'r') as fp:
                dat = json.load(fp)
                dat_return = np.array(dat['vpi_s0'])
                final_performance = dat_return[
                    :, start_plotting_episode:end_plotting_episode].mean()
            if final_performance > max_return_beta:
                max_return_beta = final_performance
                best_param_dict[FLAG_PG_TYPE]['learned_vpi'][policy_stepsize] \
                    = critic_stepsize
        if max_return_beta > max_return_alpha:
            max_return_alpha = max_return_beta
            best_param_dict[FLAG_PG_TYPE]['learned_vpi'][
                'optimal_policy_stepsize'] \
                = policy_stepsize

#----------------------------------------------------------------------
# Sensitivity plots for alpha
#----------------------------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(5, 8))

critic_stepsize = None
FLAG_LEARN_VPI = False
for FLAG_PG_TYPE, plt_color in zip(['regular', 'alternate', 'expected'],
                                   ['tab:red', 'tab:blue', 'black']):
    plot_alpha_sensitivity(ax=axs[0], folder_name=folder_name,
                           policy_stepsize_list=policy_stepsize_list,
                           FLAG_PG_TYPE=FLAG_PG_TYPE,
                           critic_stepsize=critic_stepsize,
                           plt_color=plt_color, linewidth=1,
                           start_plotting_epsiode=start_plotting_episode,
                           end_plotting_episode=end_plotting_episode,
                           FLAG_LEARN_VPI=FLAG_LEARN_VPI)

FLAG_LEARN_VPI = True
for FLAG_PG_TYPE in ['regular', 'alternate']:
    for critic_stepsize, tmp in zip(critic_stepsize_list,
                                    range(len(critic_stepsize_list))):
        c1 = 'red' if FLAG_PG_TYPE == 'regular' else 'blue'
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb('white'))
        mix = (tmp + 1) / (len(critic_stepsize_list) + 2)
        plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)

        plot_alpha_sensitivity(ax=axs[1], folder_name=folder_name,
                               policy_stepsize_list=policy_stepsize_list,
                               FLAG_PG_TYPE=FLAG_PG_TYPE,
                               critic_stepsize=critic_stepsize,
                               plt_color=plt_color, linewidth=1,
                               start_plotting_epsiode=start_plotting_episode,
                               end_plotting_episode=end_plotting_episode,
                               FLAG_LEARN_VPI=FLAG_LEARN_VPI)

axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)

# axs[0].legend()
# axs[1].legend()
# axs[0].set_ylabel(r'$\mathcal{J}_\pi$' + ' (PG objective)')
# axs[1].set_xlabel(r'$\log(\alpha)$')

plt.savefig('{}__rpi_sensitivity.pdf'.format(folder_name))
plt.close()

#----------------------------------------------------------------------
# Learning Curves
#----------------------------------------------------------------------
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True)

# expected PG
policy_stepsize = best_param_dict['expected']
idx = 0
plt_color = 'black'
# axs.set_title('expected_{}'.format(policy_stepsize))
plot_learning_curves(ax=axs, folder_name=folder_name,
                     FLAG_PG_TYPE='expected',
                     policy_stepsize=policy_stepsize, critic_stepsize=None,
                     FLAG_LEARN_VPI=False, plt_color=plt_color)

# stochastic PG
for FLAG_PG_TYPE in ['regular', 'alternate']:
    critic_stepsize = None
    policy_stepsize = best_param_dict[FLAG_PG_TYPE]['true_vpi']
    idx = 0 # idx = 1 if FLAG_PG_TYPE == 'regular' else 2
    plt_color = 'tab:red' if FLAG_PG_TYPE == 'regular' else 'tab:blue'
    # axs.set_title('{}_{}'.format(FLAG_PG_TYPE, policy_stepsize))
    plot_learning_curves(ax=axs, folder_name=folder_name,
                         FLAG_PG_TYPE=FLAG_PG_TYPE,
                         policy_stepsize=policy_stepsize, critic_stepsize=None,
                         FLAG_LEARN_VPI=False, plt_color=plt_color)

    policy_stepsize = best_param_dict[
        FLAG_PG_TYPE]['learned_vpi']['optimal_policy_stepsize']
    print(FLAG_PG_TYPE, policy_stepsize)
    critic_stepsize = best_param_dict[
        FLAG_PG_TYPE]['learned_vpi'][policy_stepsize]
    idx = 0 # idx = 3 if FLAG_PG_TYPE == 'regular' else 4
    plt_color = 'red' if FLAG_PG_TYPE == 'regular' else 'blue'
    # axs.set_title('{}_{}_{}'.format(
    #     FLAG_PG_TYPE, policy_stepsize, critic_stepsize))
    plot_learning_curves(ax=axs, folder_name=folder_name,
                         FLAG_PG_TYPE=FLAG_PG_TYPE,
                         policy_stepsize=policy_stepsize,
                         critic_stepsize=critic_stepsize,
                         FLAG_LEARN_VPI=True, plt_color=plt_color)

axs.legend()
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
# axs.set_xlabel('Timestep')

# plt.show()
plt.savefig('{}__learning_curves.pdf'.format(folder_name))
