import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import argparse

# $ python plot_graphs.py --exp_config '0,0,0' --num_runs 2 --num_total_timesteps 50000 --target_move_timestep 0 --hidden_layer_size 10 --episode_cutoff_length 1000

# read command line arguments to get parameter configurations
parser = argparse.ArgumentParser()
parser.add_argument('-ex', '--exp_config', required=True, type=str)
parser.add_argument('-nr', '--num_runs', required=True, type=int)
parser.add_argument('-nts', '--num_total_timesteps', required=True, type=int)
parser.add_argument('-mtf', '--target_move_timestep', required=True, type=int)
parser.add_argument('-hls', '--hidden_layer_size', required=True, type=int)
parser.add_argument('-ecl', '--episode_cutoff_length', required=True, type=int)

args = parser.parse_args()

exp_config = args.exp_config.split(',')
theta_init_other_actions = float(exp_config[0])
theta_init_no_action = float(exp_config[1])
value_weight_init = float(exp_config[2])

num_runs = args.num_runs
num_total_timesteps = args.num_total_timesteps
target_move_timestep = args.target_move_timestep
hidden_layer_size = args.hidden_layer_size
episode_cutoff_length = args.episode_cutoff_length

episode_cutoff_length = 1000
plotting_bin_size = 5000

# policy_stepsize_list = [2**i for i in range(-21, -2, 2)]
# policy_stepsize_list = [2**i for i in range(-19, 0, 2)]
# critic_stepsize_list = [2**i for i in range(-17, -6, 2)]

policy_stepsize_list = [2**i for i in range(-17, -2, 2)]
critic_stepsize_list = [2**i for i in range(-17, -6, 2)]

FLAG_PG_TYPE_list = ['regular', 'alternate']

# folder details
folder_name = 'nn_data/DotReacher__theta_{}_{}__valueWeightInit_{}__numRuns_{}'\
    '__numTotalSteps_{}__episodeCutoff_{}__targetmoveTimestep_{}'\
    '__hiddenLayerSize_{}'.format(
        theta_init_other_actions, theta_init_no_action, value_weight_init,
        num_runs, num_total_timesteps, episode_cutoff_length,
        target_move_timestep, hidden_layer_size)
print(folder_name)

def calc_final_perf(dat, idx=-1):
    assert num_runs == np.array(dat['returns']).shape[0]
    dat_mean = np.array(dat['returns']).mean(0)
    dat_stderr = np.array(dat['returns']).std(0) / np.sqrt(num_runs)

    # tmp = int(np.ceil(np.array(dat['returns']).shape[1] / 2))
    # final_perf_mean = dat_mean[tmp]
    # final_perf_stderr = dat_stderr[tmp]

    final_perf_mean = dat_mean[idx]
    final_perf_stderr = dat_stderr[idx]
    return final_perf_mean, final_perf_stderr

def process_data(folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
                 plotting_bin_size):
    processed_dat = {}
    processed_dat['returns'] = []
    processed_dat['vpi_s0'] = []
    processed_dat['entropy'] = []
    for seed_number in range(num_runs):
        # print('-----------------------------------------------------')
        returns_single_run = []
        vpi_s0_single_run = []
        entropy_single_run = []

        filename = '{}/pg_{}__pol_{}__val_{}__seed_{}'.format(
            folder_name, FLAG_PG_TYPE, float(policy_stepsize),
            float(critic_stepsize),seed_number)

        with open(filename, 'r') as fp:
            dat = json.load(fp)

        FLAG_CONTINUE_ITERATING = True
        FLAG_SAVE_CURRENT_BIN_ITEMS = False
        total_timesteps_cumulated = 0
        current_bin_max_length = plotting_bin_size
        returns_across_bin = []
        vpi_s0_across_bin = []
        entropy_across_bin = []
        
        i = 0
        while FLAG_CONTINUE_ITERATING:
            ep_len = dat['ep_len'][i]
            ret = dat['returns'][i]
            vpi = dat['vpi_s0'][i]
            entropy = dat['entropy'][i]
            
            # add items to the current bin and continue to the next item
            returns_across_bin.append(ret)
            vpi_s0_across_bin.append(vpi)
            entropy_across_bin.append(entropy)
            total_timesteps_cumulated += ep_len
            i = i + 1

            # if this bin is full
            if total_timesteps_cumulated > current_bin_max_length:
                FLAG_SAVE_CURRENT_BIN_ITEMS = True # save the items           
                current_bin_max_length += plotting_bin_size # create a new bin

            # even if the bin is not full, we ran out of items to put in.
            if i >= len(dat['ep_len']):
                FLAG_SAVE_CURRENT_BIN_ITEMS = True # therefore save the items
                FLAG_CONTINUE_ITERATING = False # don't continue shopping

            # if for some reason we need to save the items, do that!
            if FLAG_SAVE_CURRENT_BIN_ITEMS:
                # print(returns_across_bin, sum(returns_across_bin))
                returns_single_run.append(np.mean(returns_across_bin))
                vpi_s0_single_run.append(np.mean(vpi_s0_across_bin))
                entropy_single_run.append(np.mean(entropy_across_bin))
                    
                returns_across_bin = []
                vpi_s0_across_bin = []
                entropy_across_bin = []
                FLAG_SAVE_CURRENT_BIN_ITEMS = False

        # print(len(returns_single_run))
        processed_dat['returns'].append(returns_single_run)
        processed_dat['vpi_s0'].append(vpi_s0_single_run)
        processed_dat['entropy'].append(entropy_single_run)
        
    # if len(dat['returns']) != 100:
    # plt.plot(np.array(processed_dat['returns']).T,
    #          color='red', linewidth=0.5)
    # plt.plot(np.array(processed_dat['vpi_s0']).T,
    #          color='blue', linewidth=0.5)
    # pdb.set_trace()
    return processed_dat

def plot_alpha_sensitivity(ax, folder_name, policy_stepsize_list, FLAG_PG_TYPE,
                           critic_stepsize, plt_color, linewidth,
                           plotting_bin_size):
    final_perf_mean_list = []
    final_perf_stderr_list = []
    for policy_stepsize in policy_stepsize_list:
        dat = process_data(folder_name=folder_name,
                           FLAG_PG_TYPE=FLAG_PG_TYPE,
                           policy_stepsize=policy_stepsize,
                           critic_stepsize=critic_stepsize,
                           plotting_bin_size=plotting_bin_size)
        # pdb.set_trace()
        final_perf_mean, final_perf_stderr = calc_final_perf(dat)
        
        final_perf_mean_list.append(final_perf_mean)
        final_perf_stderr_list.append(final_perf_stderr)

    ax.errorbar(np.log(policy_stepsize_list) / np.log(2), final_perf_mean_list,
                yerr=final_perf_stderr_list, color=plt_color,
                label='{}_{}'.format(FLAG_PG_TYPE, critic_stepsize),
                linewidth=linewidth)
    ax.scatter(-10, 0, s=0) # add invisible point to make scaling from 0 -> 1

def plot_learning_curves(ax, folder_name, FLAG_PG_TYPE, policy_stepsize,
                         critic_stepsize, plt_color, plotting_bin_size,
                         ax_v=None, ax_ent=None):
    dat = process_data(folder_name=folder_name, FLAG_PG_TYPE=FLAG_PG_TYPE,
                       policy_stepsize=policy_stepsize,
                       critic_stepsize=critic_stepsize,
                       plotting_bin_size=plotting_bin_size)
    dat_mean = np.array(dat['returns']).mean(0)
    dat_stderr = np.array(dat['returns']).std(0) / np.sqrt(num_runs)

    # average performance across runs
    ax.plot(np.arange(dat_mean.shape[0]), dat_mean,
            linewidth=1.5, color=plt_color,
            label='{}_{}_{}'.format(
                FLAG_PG_TYPE, policy_stepsize, critic_stepsize))
    ax.fill_between(np.arange(dat_mean.shape[0]),
                    dat_mean - dat_stderr, dat_mean + dat_stderr,
                    alpha=0.2, facecolor=plt_color)
    
    if ax_v is not None: # plot learned value function
        plt_color_v = 'red' if FLAG_PG_TYPE == 'regular' else 'blue'
        dat_mean_v = np.array(dat['vpi_s0']).mean(0)
        dat_stderr_v = np.array(dat['vpi_s0']).std(0) / np.sqrt(num_runs)
        ax_v.plot(np.arange(dat_mean.shape[0]), dat_mean_v,
                linewidth=1.5, color=plt_color_v)
        ax_v.fill_between(np.arange(dat_mean_v.shape[0]),
                          dat_mean_v - dat_stderr_v, dat_mean_v + dat_stderr_v,
                          alpha=0.2, facecolor=plt_color_v)

    if ax_ent is not None: # plot the average policy entropy
        plt_color_ent = 'red' if FLAG_PG_TYPE == 'regular' else 'blue'
        dat_mean_ent = np.array(dat['entropy']).mean(0)
        dat_stderr_ent = np.array(dat['entropy']).std(0) / np.sqrt(num_runs)
        ax_ent.plot(np.arange(dat_mean.shape[0]), dat_mean_ent,
                    linewidth=1.5, color=plt_color_ent)
        ax_ent.fill_between(np.arange(dat_mean_ent.shape[0]),
                            dat_mean_ent - dat_stderr_ent,
                            dat_mean_ent + dat_stderr_ent,
                            alpha=0.2, facecolor=plt_color_ent)

    # individual runs
    # ax.plot(np.array(dat['returns']).T,
    #         linewidth=0.5, alpha=0.5, color=plt_color)

    # # real individual runs (unprocessed raw data)
    # for seed_number in range(num_runs):
    #     filename = '{}/pg_{}__pol_{}__val_{}__seed_{}'.format(
    #         folder_name, FLAG_PG_TYPE, float(policy_stepsize),
    #         float(critic_stepsize), seed_number)
    #     with open(filename, 'r') as fp:
    #         dat = json.load(fp)
            
    #     x_data = np.cumsum(np.array(dat['ep_len'])) / plotting_bin_size
    #     y_data = np.array(dat['returns'])
    #     tmp_color = 'green' if FLAG_PG_TYPE == 'regular' else 'black'
    #     ax.plot(x_data, y_data, linewidth=0.5, alpha=0.02, color=plt_color)
    
    ax.scatter(0, 0, s=0) # add an invisible point to make scaling from 0 -> 1


#----------------------------------------------------------------------
# Find the best hyper-parameter configuration
#----------------------------------------------------------------------
best_param_dict = dict()
for FLAG_PG_TYPE in FLAG_PG_TYPE_list:
    best_param_dict[FLAG_PG_TYPE] = dict()
    best_param_dict[FLAG_PG_TYPE]['optimal_policy_stepsize'] = None
    max_return_alpha = -1 * np.inf
    
    for policy_stepsize in policy_stepsize_list:
        best_param_dict[FLAG_PG_TYPE][policy_stepsize] = None
        max_return_beta = -1 * np.inf
        
        for critic_stepsize in critic_stepsize_list:
            # read and process the data
            dat = process_data(folder_name=folder_name,
                               FLAG_PG_TYPE=FLAG_PG_TYPE,
                               policy_stepsize=policy_stepsize,
                               critic_stepsize=critic_stepsize,
                               plotting_bin_size=plotting_bin_size)
            final_perf_mean, _ = calc_final_perf(dat)

            if final_perf_mean > max_return_beta:
                max_return_beta = final_perf_mean
                best_param_dict[FLAG_PG_TYPE][policy_stepsize] = critic_stepsize
                        
        if max_return_beta > max_return_alpha:
            max_return_alpha = max_return_beta
            best_param_dict[FLAG_PG_TYPE]['optimal_policy_stepsize'] \
                = policy_stepsize

#----------------------------------------------------------------------
# Sensitivity plots for alpha
#----------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(5, 4))

for FLAG_PG_TYPE in FLAG_PG_TYPE_list:
    for critic_stepsize, tmp in zip(critic_stepsize_list,
                                    range(len(critic_stepsize_list))):
        c1 = 'red' if FLAG_PG_TYPE == 'regular' else 'blue'
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb('white'))
        mix = (tmp + 1) / (len(critic_stepsize_list) + 2)
        plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)

        plot_alpha_sensitivity(ax=ax, folder_name=folder_name,
                               policy_stepsize_list=policy_stepsize_list,
                               FLAG_PG_TYPE=FLAG_PG_TYPE,
                               critic_stepsize=critic_stepsize,
                               plt_color=plt_color, linewidth=1,
                               plotting_bin_size=plotting_bin_size)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.legend()
ax.set_ylabel(r'$\mathcal{J}_\pi$' + ' (PG objective)')
ax.set_xlabel(r'$\log(\alpha)$')

plt.savefig('{}_sensitivity.pdf'.format(folder_name))
plt.close()

#----------------------------------------------------------------------
# Learning Curves
#----------------------------------------------------------------------
num_fig_cols = 3 
fig, axs = plt.subplots(num_fig_cols, 1, figsize=(5, 12), sharex=True)

# stochastic PG
for FLAG_PG_TYPE in FLAG_PG_TYPE_list:
    policy_stepsize = best_param_dict[FLAG_PG_TYPE]['optimal_policy_stepsize']
    print(FLAG_PG_TYPE, policy_stepsize)
    critic_stepsize = best_param_dict[FLAG_PG_TYPE][policy_stepsize]
    plt_color = 'red' if FLAG_PG_TYPE == 'regular' else 'blue'
    # axs.set_title('{}_{}_{}'.format(
    #     FLAG_PG_TYPE, policy_stepsize, critic_stepsize))
    ax_ent = axs[2]
    plot_learning_curves(ax=axs[0], folder_name=folder_name,
                         FLAG_PG_TYPE=FLAG_PG_TYPE,
                         policy_stepsize=policy_stepsize,
                         critic_stepsize=critic_stepsize,
                         plt_color=plt_color,
                         plotting_bin_size=plotting_bin_size,
                         ax_v=axs[1], ax_ent=axs[2])

axs[0].legend()
for i in range(2):
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
axs[1].set_xlabel('Timesteps')

# plt.show()
plt.savefig('{}_learning_curves.pdf'.format(folder_name))
