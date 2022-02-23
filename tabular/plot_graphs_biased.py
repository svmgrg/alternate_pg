import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

num_runs = 150
num_episodes = 100

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

end_plotting_episode = 100
start_plotting_episode = end_plotting_episode - 10



def plot_alpha_sensitivity(ax, folder_name, policy_stepsize_list, FLAG_PG_TYPE,
                           critic_stepsize, plt_color, linewidth,
                           start_plotting_epsiode, end_plotting_episode,
                           FLAG_LEARN_VPI, label):
    perf_mean_list = []
    perf_stderr_list = []
    for policy_stepsize in policy_stepsize_list:
        filename='{}/pg_{}__pol_{}__val_{}__learnVpi_{}_biased'.format(
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
                label='{}'.format(label),
                linewidth=linewidth)
    ax.scatter(0, 0.45, s=0) # add an invisible point to make scaling from 0 -> 1

#----------------------------------------------------------------------
# Sensitivity plots for alpha
#----------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

critic_stepsize = None
FLAG_LEARN_VPI = False
for value_weight_init, axs_idx in zip([+4, -4], [0, 1]):
    for FLAG_PG_TYPE, tmp_color in zip(['regular', 'alternate'],
                                       ['red', 'blue']):
        policy_weight_init_right = 0
        for policy_weight_init_left, tmp in zip([0, 1, 2, 3], range(4)):
            folder_name = 'mdp_data_biased/linearChain__thetaInit_{}_{}'\
                '__valueWeightInit_{}__numRuns_{}__numEpisodes_{}'\
                '__startState_{}__rewardNoise_{}__gamma_{}'\
                '__episodeCutoff_{}__gradBias_{}__gradNoise_{}'.format(
                    policy_weight_init_left, policy_weight_init_right,
                    value_weight_init, num_runs, num_episodes, start_state,
                    reward_noise, gamma, episode_cutoff_length,
                    grad_bias, grad_noise)

            c1 = np.array(mpl.colors.to_rgb(tmp_color))
            c2 = np.array(mpl.colors.to_rgb('white'))
            mix = (tmp + 1) / (4 + 2)
            plt_color = mpl.colors.to_hex((1 - mix) * c1 + mix * c2)
            plot_alpha_sensitivity(ax=axs[axs_idx], folder_name=folder_name,
                                   policy_stepsize_list=policy_stepsize_list,
                                   FLAG_PG_TYPE=FLAG_PG_TYPE,
                                   critic_stepsize=critic_stepsize,
                                   plt_color=plt_color, linewidth=1,
                                   start_plotting_epsiode=start_plotting_episode,
                                   end_plotting_episode=end_plotting_episode,
                                   FLAG_LEARN_VPI=FLAG_LEARN_VPI,
                                   label=str([policy_weight_init_left,
                                              policy_weight_init_right]))

axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)

# axs[0].legend()
# axs[1].legend()
# axs[0].set_ylabel(r'$\mathcal{J}_\pi$' + ' (PG objective)')
# axs[1].set_xlabel(r'$\log(\alpha)$')

plt.savefig('{}__biased.pdf'.format(folder_name))
plt.close()
