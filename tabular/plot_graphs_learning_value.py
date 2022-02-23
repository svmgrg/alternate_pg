import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
        
stepsize_list = np.arange(0.0, 3.1, 0.1)
critic_factor_list = [0.1, 0.5, 1.0, 2.0]
alg_list = ['regular', 'alternate']

weight_init_left = 3
weight_init_right = 0

# folder details
folder_name = 'linear_chain_l{}_r{}'.format(
    weight_init_left, weight_init_right)

fig, axs = plt.subplots(2, 3, figsize=(16, 4))

plot_episode_list = [10, 50, 200]

for alg_i in range(2):
    FLAG_PG_TYPE = alg_list[alg_i]
    for critic_factor in critic_factor_list:
        curve_mean = [[] for i in range(3)]
        curve_stderr = [[] for i in range(3)]

        for policy_stepsize in stepsize_list:
            value_stepsize = critic_factor * policy_stepsize
            
            filename='{}/pg_{}__pol_{}__val_{}'.format(
                folder_name, FLAG_PG_TYPE, policy_stepsize, critic_factor)

            with open(filename, 'r') as fp:
                dat = json.load(fp)

            if FLAG_PG_TYPE == 'expected':
                dat_return = np.array(dat['vpi_s0'])
            else:
                dat_return = np.array(dat['returns'])

            num_runs = dat_return.shape[0]
            ep_len = dat_return.shape[1]

            for i in range(3):
                start = plot_episode_list[i]-10
                end = plot_episode_list[i]
                if dat_return[:, start:end].mean(1).mean() < 0:
                    curve_mean[i].append(0)
                else:
                    curve_mean[i].append(
                        dat_return[:, start:end].mean(1).mean())
                curve_stderr[i].append(0)
                # curve_stderr[i].append(
                #     dat_return[:, start:end].mean(1).std()/np.sqrt(num_runs))

        for i in range(3):
            axs[alg_i][i].errorbar(
                stepsize_list, curve_mean[i], yerr=curve_stderr[i],
                label='{}_{}'.format(FLAG_PG_TYPE, critic_factor))    

    axs[alg_i][0].set_ylabel('{} Returns'.format(FLAG_PG_TYPE))
    
for i in range(3):
    axs[0][i].legend()
    axs[1][i].legend()
    axs[0][i].set_title('End of {} episodes'.format(plot_episode_list[i]))
    axs[1][i].set_xlabel('Stepsize')

plt.show()
# plt.savefig('{}_value_learn.png'.format(folder_name), dpi=150)
