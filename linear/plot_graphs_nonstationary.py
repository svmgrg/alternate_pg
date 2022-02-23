import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import argparse
import time

# python plot_graphs_nonstationary.py --env_name 'GymMountainCar' --exp_config '0,0,0,0,None,None' --num_runs 50 --policy_mapping 'softmax'

# read command line arguments to get parameter configurations
parser = argparse.ArgumentParser()
parser.add_argument('-en', '--env_name', required=True, type=str)
parser.add_argument('-ex', '--exp_config', required=True, type=str)
parser.add_argument('-nr', '--num_runs', required=True, type=int)
parser.add_argument('-pm', '--policy_mapping', required=True, type=str)

FLAG_SWITCH_ACTIONS = True
FLAG_STATE_AGGREGATION = False
FLAG_CAPTURE_ENTROPY = True

args = parser.parse_args()

env_name = args.env_name
exp_config = args.exp_config.split(',')
theta_init_0 = float(exp_config[0])
theta_init_1 = float(exp_config[1])
theta_init_2 = float(exp_config[2])
value_weight_init = float(exp_config[3])
grad_bias = None if exp_config[4] == 'None' else float(exp_config[4])
grad_noise = None if exp_config[5] == 'None' else float(exp_config[5])
num_runs = args.num_runs
FLAG_POLICY_MAPPING = args.policy_mapping

# env_name = 'GymMountainCar'
# theta_init_0 = 50
# theta_init_1 = 0
# theta_init_2 = 0
# value_weight_init = -200
# grad_bias = None
# grad_noise = None
# num_runs = 8

if env_name == 'GymMountainCar':
    num_total_timesteps = 200000
elif env_name == 'Acrobot':
    num_total_timesteps = 400000
else:
    raise NotImplementedError()

gamma = 1
episode_cutoff_length = 1000
plotting_bin_size = 5000
idx_list = [int(num_total_timesteps / plotting_bin_size / 2) - 1, -1]

# if env_name == 'Acrobot':
#     policy_stepsize_list=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 16, 32)
#     critic_stepsize_list=(0.001, 0.05, 0.01, 0.05, 0.1, 0.5, 1, 2, 4)
# elif env_name == 'MountainCar' or env_name == 'GymMountainCar':
#     policy_stepsize_list = [2**i for i in range(-15, 9, 2)]
#     critic_stepsize_list = [0.01, 0.05, 0.1, 0.5, 1, 2]
#     # policy_stepsize_list=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 16,
#     #                       32, 64, 128, 256, 512, 1024)
#     # critic_stepsize_list=(0.001, 0.05, 0.01, 0.05, 0.1, 0.5, 1, 2, 4)
# else:
#     raise NotImplementedError()

policy_stepsize_list = [2**i for i in range(-13, 6, 2)]
critic_stepsize_list = [0.01, 0.05, 0.1, 0.5, 1, 2]
escort_p_list = [1.0, 2.0, 3.0, 5.0, 10.0]
ent_tau_list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

if FLAG_POLICY_MAPPING == 'softmax':
    FLAG_PG_TYPE_list = ['regular', 'alternate']
elif FLAG_POLICY_MAPPING == 'escort':
    FLAG_PG_TYPE_list = ['regular']
else:
    raise NotImplementedError()

FLAG_BASELINE = True
FLAG_POPULAR_PG = True
FLAG_LEARN_VPI = True

# folder details
# folder details
folder_name = 'mdp_data/{}_{}__Tiles__theta_{}_{}_{}__valueWeightInit_{}'\
    '__numRuns_{}__numTotalSteps_{}__gamma_{}__episodeCutoff_{}__gradBias_{}'\
    '__gradNoise_{}__flagBaseline_{}__flagPopularPG_{}__flagLearnVpi'.format(
        env_name, FLAG_POLICY_MAPPING, 
        float(theta_init_0), float(theta_init_1), float(theta_init_2),
        value_weight_init,
        num_runs, num_total_timesteps, float(gamma), episode_cutoff_length,
        grad_bias, grad_noise, FLAG_BASELINE, FLAG_POPULAR_PG, FLAG_LEARN_VPI)
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
                 escort_p, ent_tau, plotting_bin_size):
    processed_dat = {}
    processed_dat['returns'] = []
    processed_dat['vpi_s0'] = []
    if FLAG_CAPTURE_ENTROPY:
        processed_dat['entropy'] = []
    for seed_number in range(num_runs):
        returns_single_run = []
        vpi_s0_single_run = []
        if FLAG_CAPTURE_ENTROPY:
            entropy_single_run = []

        if (not FLAG_SWITCH_ACTIONS) \
           and (not FLAG_STATE_AGGREGATION) \
           and (not FLAG_CAPTURE_ENTROPY):
            filename='{}/pg_{}__pol_{}__val_{}__seed_{}'.format(
                folder_name, FLAG_PG_TYPE, float(policy_stepsize),
                float(critic_stepsize), seed_number)
        else:
            filename='{}/pg_{}__pol_{}__val_{}__escort_p_{}__ent_tau_{}__'\
                '__seed_{}__switchActions_{}'\
                '__stateAggregation_{}__captureEntropy_{}'.format(
                    folder_name, FLAG_PG_TYPE, float(policy_stepsize),
                    float(critic_stepsize), float(escort_p), float(ent_tau),
                    seed_number, FLAG_SWITCH_ACTIONS,
                    FLAG_STATE_AGGREGATION, FLAG_CAPTURE_ENTROPY)

        with open(filename, 'r') as fp:
            dat = json.load(fp)

        FLAG_CONTINUE_ITERATING = True
        FLAG_SAVE_CURRENT_BIN_ITEMS = False
        total_timesteps_cumulated = 0
        current_bin_max_length = plotting_bin_size
        returns_across_bin = []
        vpi_s0_across_bin = []
        if FLAG_CAPTURE_ENTROPY:
            entropy_across_bin = []
        i = 0
        while FLAG_CONTINUE_ITERATING:
            ep_len = dat['ep_len'][i]
            ret = dat['returns'][i]
            vpi = dat['vpi_s0'][i]
            if FLAG_CAPTURE_ENTROPY:
                entropy = dat['entropy'][i]
            
            # add items to the current bin and continue to the next item
            returns_across_bin.append(ret)
            vpi_s0_across_bin.append(vpi)
            if FLAG_CAPTURE_ENTROPY:
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
                if FLAG_CAPTURE_ENTROPY:
                    entropy_single_run.append(np.mean(entropy_across_bin))
                    
                returns_across_bin = []
                vpi_s0_across_bin = []
                if FLAG_CAPTURE_ENTROPY:
                    entropy_across_bin = []
                FLAG_SAVE_CURRENT_BIN_ITEMS = False

        # print(len(returns_single_run))
        processed_dat['returns'].append(returns_single_run)
        processed_dat['vpi_s0'].append(vpi_s0_single_run)
        if FLAG_CAPTURE_ENTROPY:
            processed_dat['entropy'].append(entropy_single_run)
        
    # if len(dat['returns']) != 100:
    # plt.plot(np.array(processed_dat['returns']).T,
    #          color='red', linewidth=0.5)
    # plt.plot(np.array(processed_dat['vpi_s0']).T,
    #          color='blue', linewidth=0.5)
    # pdb.set_trace()
    return processed_dat

def plot_alpha_sensitivity(ax, folder_name, policy_stepsize_list, FLAG_PG_TYPE,
                           critic_stepsize, escort_p, ent_tau,
                           plt_color, linewidth,
                           plotting_bin_size, idx=-1):
    final_perf_mean_list = []
    final_perf_stderr_list = []
    for policy_stepsize in policy_stepsize_list:
        dat = process_data(folder_name=folder_name,
                           FLAG_PG_TYPE=FLAG_PG_TYPE,
                           policy_stepsize=policy_stepsize,
                           critic_stepsize=critic_stepsize,
                           escort_p=escort_p, ent_tau=ent_tau,
                           plotting_bin_size=plotting_bin_size)
        final_perf_mean, final_perf_stderr = calc_final_perf(dat, idx)
        
        final_perf_mean_list.append(final_perf_mean)
        final_perf_stderr_list.append(final_perf_stderr)

    ax.errorbar(np.log(policy_stepsize_list) / np.log(2), final_perf_mean_list,
                yerr=final_perf_stderr_list, color=plt_color,
                label='{}_{}'.format(FLAG_PG_TYPE, critic_stepsize),
                linewidth=linewidth)
    # ax.plot(np.log(policy_stepsize_list) / np.log(2), final_perf_mean_list,
    #         color=plt_color,
    #         label='{}_{}'.format(FLAG_PG_TYPE, critic_stepsize),
    #         linewidth=linewidth)
    # # ax.scatter(0, 0, s=0) # add invisible point to make scaling from 0 -> 1
    ax.set_ylim([-1000, -100])

def plot_learning_curves(ax, folder_name, FLAG_PG_TYPE,
                         policy_stepsize, critic_stepsize,
                         escort_p, ent_tau,
                         plt_color, plotting_bin_size,
                         ax_v=None, ax_ent=None):
    dat = process_data(folder_name=folder_name, FLAG_PG_TYPE=FLAG_PG_TYPE,
                       policy_stepsize=policy_stepsize,
                       critic_stepsize=critic_stepsize,
                       escort_p=escort_p, ent_tau=ent_tau, 
                       plotting_bin_size=plotting_bin_size)
    dat_mean = np.array(dat['returns']).mean(0)
    dat_stderr = np.array(dat['returns']).std(0) / np.sqrt(num_runs)

    # average performance across runs
    ax.plot(np.arange(dat_mean.shape[0]), dat_mean,
            linewidth=1.5, color=plt_color,
            label='{}_p{}_tau{}_{}_{}'.format(
                FLAG_PG_TYPE, escort_p, ent_tau, policy_stepsize, critic_stepsize))
    ax.fill_between(np.arange(dat_mean.shape[0]),
                    dat_mean - dat_stderr, dat_mean + dat_stderr,
                    alpha=0.2, facecolor=plt_color)
    ax.set_ylim([-1100, -100])               
    
    if ax_v is not None: # plot learned value function
        plt_color_v = plt_color
        dat_mean_v = np.array(dat['vpi_s0']).mean(0)
        dat_stderr_v = np.array(dat['vpi_s0']).std(0) / np.sqrt(num_runs)
        ax_v.plot(np.arange(dat_mean.shape[0]), dat_mean_v,
                linewidth=1.5, color=plt_color_v)
        ax_v.fill_between(np.arange(dat_mean_v.shape[0]),
                          dat_mean_v - dat_stderr_v, dat_mean_v + dat_stderr_v,
                          alpha=0.2, facecolor=plt_color_v)

    if ax_ent is not None: # plot the average policy entropy
        plt_color_ent = plt_color
        dat_mean_ent = np.array(dat['entropy']).mean(0)
        dat_stderr_ent = np.array(dat['entropy']).std(0) / np.sqrt(num_runs)
        ax_ent.plot(np.arange(dat_mean.shape[0]), dat_mean_ent,
                    linewidth=1.5, color=plt_color_ent)
        ax_ent.fill_between(np.arange(dat_mean_ent.shape[0]),
                            dat_mean_ent - dat_stderr_ent,
                            dat_mean_ent + dat_stderr_ent,
                            alpha=0.2, facecolor=plt_color_ent)
        ax_ent.set_ylim([0, 1.2])

    # individual runs
    # ax.plot(np.array(dat['returns']).T,
    #         linewidth=0.5, alpha=0.5, color=plt_color)

    # # real individual runs (not processed; raw data)
    # for seed_number in range(num_runs):
    #     if (not FLAG_SWITCH_ACTIONS) \
    #        and (not FLAG_STATE_AGGREGATION) \
    #        and (not FLAG_CAPTURE_ENTROPY):
    #         filename='{}/pg_{}__pol_{}__val_{}__seed_{}'.format(
    #             folder_name, FLAG_PG_TYPE, float(policy_stepsize),
    #             float(critic_stepsize), seed_number)
    #     else:
    #         filename='{}/pg_{}__pol_{}__val_{}__seed_{}__switchActions_{}'\
    #             '__stateAggregation_{}__captureEntropy_{}'.format(
    #                 folder_name, FLAG_PG_TYPE, float(policy_stepsize),
    #                 float(critic_stepsize), seed_number, FLAG_SWITCH_ACTIONS,
    #                 FLAG_STATE_AGGREGATION, FLAG_CAPTURE_ENTROPY)

    #     with open(filename, 'r') as fp:
    #         dat = json.load(fp)
    #     x_data = np.cumsum(np.array(dat['ep_len'])) / plotting_bin_size
    #     y_data = np.array(dat['returns'])
    #     tmp_color = 'blueviolet' if FLAG_PG_TYPE == 'regular' else 'black'
    #     ax.plot(x_data, y_data, linewidth=0.5, alpha=0.02, color=plt_color)
    
    # ax.scatter(0, 0, s=0) # add an invisible point to make scaling from 0 -> 1


#----------------------------------------------------------------------
# Find the best hyper-parameter configuration
#----------------------------------------------------------------------
if FLAG_POLICY_MAPPING == 'softmax':
    p_or_tau_list = ent_tau_list
    third_param_name = 'ent_tau'
elif FLAG_POLICY_MAPPING == 'escort':
    p_or_tau_list = escort_p_list
    third_param_name = 'escort_p'
else:
    raise NotImplementedError()

def find_best_hyperparams(FLAG_PG_TYPE_list, idx):
    bmd = dict()
    
    for FLAG_PG_TYPE in FLAG_PG_TYPE_list: # iterate in ['regular', 'alternate']
        bmd[FLAG_PG_TYPE] = dict()
        bmd[FLAG_PG_TYPE]['optimal_p_or_tau'] = None
        max_return = -1 * np.inf
        
        for p_or_tau in p_or_tau_list: # this is either the p_list or tau_list
            bmd[FLAG_PG_TYPE][p_or_tau] = dict()
            bmd[FLAG_PG_TYPE][p_or_tau]['optimal_policy_stepsize'] = None
            max_return_alpha = -1 * np.inf

            if FLAG_POLICY_MAPPING == 'softmax':
                ent_tau = p_or_tau
                escort_p = 0.0
            elif FLAG_POLICY_MAPPING == 'escort':
                ent_tau = 0.0
                escort_p = p_or_tau
            else:
                raise NotImplementedError()
            
            for policy_stepsize in policy_stepsize_list:
                bmd[FLAG_PG_TYPE][p_or_tau][policy_stepsize] = None
                max_return_beta = -1 * np.inf

                for critic_stepsize in critic_stepsize_list:
                    # read and process the data
                    dat = process_data(folder_name=folder_name,
                                       FLAG_PG_TYPE=FLAG_PG_TYPE,
                                       policy_stepsize=policy_stepsize,
                                       critic_stepsize=critic_stepsize,
                                       escort_p=escort_p, ent_tau=ent_tau,
                                       plotting_bin_size=plotting_bin_size)
                    final_perf_mean, _ = calc_final_perf(dat, idx)

                    if final_perf_mean > max_return_beta:
                        max_return_beta = final_perf_mean
                        bmd[FLAG_PG_TYPE][p_or_tau][policy_stepsize] \
                            = critic_stepsize

                if max_return_beta > max_return_alpha:
                    max_return_alpha = max_return_beta
                    bmd[FLAG_PG_TYPE][p_or_tau]['optimal_policy_stepsize'] \
                        = policy_stepsize
                    
            if max_return_alpha > max_return:
                bmd[FLAG_PG_TYPE]['optimal_p_or_tau'] = p_or_tau
                max_return = max_return_alpha
                
    return bmd

tic = time.time()
#----------------------------------------------------------------------
# Sensitivity plots for alpha
#----------------------------------------------------------------------
num_rows = len(p_or_tau_list)

fig, axs = plt.subplots(num_rows, 2, figsize=(14, 4 * num_rows), sharey=True)

for row, p_or_tau in enumerate(p_or_tau_list):
    print(row)
    if FLAG_POLICY_MAPPING == 'softmax':
        ent_tau = p_or_tau
        escort_p = 0.0
    elif FLAG_POLICY_MAPPING == 'escort':
        ent_tau = 0.0
        escort_p = p_or_tau
    else:
        raise NotImplementedError()
    
    for col, idx in enumerate(idx_list):
        for FLAG_PG_TYPE in FLAG_PG_TYPE_list:
            if idx == idx_list[0]: # half_point
                if FLAG_POLICY_MAPPING == 'softmax':
                    c1 = 'red' if FLAG_PG_TYPE == 'regular' else 'blue'
                else:
                    c1 = 'green'
            else: # end of experiment
                if FLAG_POLICY_MAPPING == 'softmax':
                    c1 = 'tomato' if FLAG_PG_TYPE == 'regular' else 'blueviolet'
                else:
                    c1 = 'deeppink'
            c1 = np.array(mpl.colors.to_rgb(c1))

            for tmp, critic_stepsize in enumerate(critic_stepsize_list):
                c2 = np.array(mpl.colors.to_rgb('white'))
                mix = (tmp + 1) / (len(critic_stepsize_list) + 2)
                plt_color = mpl.colors.to_hex(mix * c1 + (1 - mix) * c2)

                plot_alpha_sensitivity(
                    ax=axs[row][col],
                    folder_name=folder_name,
                    policy_stepsize_list=policy_stepsize_list,
                    FLAG_PG_TYPE=FLAG_PG_TYPE,
                    critic_stepsize=critic_stepsize,
                    escort_p=escort_p, ent_tau=ent_tau,
                    plt_color=plt_color, linewidth=1,
                    plotting_bin_size=plotting_bin_size, idx=idx)

for row, p_or_tau in enumerate(p_or_tau_list):
    for col in range(2):
        axs[row][col].spines['right'].set_visible(False)
        axs[row][col].spines['top'].set_visible(False)
    axs[row][0].set_ylabel('{}: {}'.format(third_param_name, p_or_tau))
axs[0][0].legend()
axs[0][1].legend()

plt.savefig('{}_sensitivity.pdf'.format(folder_name))
plt.close()
print('sensitivity done | time taken:{}'.format(time.time() - tic))

#----------------------------------------------------------------------
# Learning Curves
#----------------------------------------------------------------------
num_fig_cols = 3 if FLAG_CAPTURE_ENTROPY else 2
fig, axs = plt.subplots(num_fig_cols, 1, figsize=(10, 12), sharex=True)

# stochastic PG
for idx in idx_list:
    bmd = find_best_hyperparams(FLAG_PG_TYPE_list, idx)
    with open('{}_{}.json'.format(folder_name[9:], idx), 'w') as fp:
        json.dump(bmd, fp)
        
    for FLAG_PG_TYPE in FLAG_PG_TYPE_list:
        if idx == idx_list[0]: # half_point
            if FLAG_POLICY_MAPPING == 'softmax':
                plt_color = 'red' if FLAG_PG_TYPE == 'regular' else 'blue'
            else:
                plt_color = 'green'
        else: # end of experiment
            if FLAG_POLICY_MAPPING == 'softmax':
                plt_color = 'tomato' if FLAG_PG_TYPE == 'regular' \
                    else 'blueviolet'
            else:
                plt_color = 'deeppink'

        p_or_tau = bmd[FLAG_PG_TYPE]['optimal_p_or_tau']
        if FLAG_POLICY_MAPPING == 'softmax':
            ent_tau = p_or_tau
            escort_p = 0.0
        elif FLAG_POLICY_MAPPING == 'escort':
            ent_tau = 0.0
            escort_p = p_or_tau
        else:
            raise NotImplementedError()
        
        policy_stepsize \
            = bmd[FLAG_PG_TYPE][p_or_tau]['optimal_policy_stepsize']
        
        critic_stepsize = bmd[FLAG_PG_TYPE][p_or_tau][policy_stepsize]

        print(FLAG_PG_TYPE, FLAG_POLICY_MAPPING,
              'third_param_name', p_or_tau,
              'policy_stepsize', policy_stepsize,
              'critic_stepsize', critic_stepsize)

        # axs.set_title('{}_{}_{}'.format(
        #     FLAG_PG_TYPE, policy_stepsize, critic_stepsize))
        ax_ent = axs[2] if FLAG_CAPTURE_ENTROPY else None
        
        plot_learning_curves(ax=axs[0], folder_name=folder_name,
                             FLAG_PG_TYPE=FLAG_PG_TYPE,
                             policy_stepsize=policy_stepsize,
                             critic_stepsize=critic_stepsize,
                             escort_p=escort_p, ent_tau=ent_tau,
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
print('learning curves done | total time taken:{}'.format(time.time() - tic))
