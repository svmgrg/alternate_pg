import torch
import pdb
import time
import argparse
import os
import json

from nn_experiment import DotReacher, OnlineAC_NN, run_experiment

# $ python online_ac.py --exp_config '0,0,0' --policy_stepsize 0.0003 --critic_stepsize 0.003 --seed_number 0 --num_runs 1 --num_total_timesteps 100000 --target_move_timestep 0 --hidden_layer_size 10 --episode_cutoff_length 1000

# read command line arguments to get parameter configurations
parser = argparse.ArgumentParser()
parser.add_argument('-ex', '--exp_config', required=True, type=str)
parser.add_argument('-al', '--policy_stepsize', required=True, type=float)
parser.add_argument('-be', '--critic_stepsize', required=True, type=float)
parser.add_argument('-nr', '--num_runs', required=True, type=int)
parser.add_argument('-sn', '--seed_number', required=True, type=int)
parser.add_argument('-nts', '--num_total_timesteps', required=True, type=int)
parser.add_argument('-mtf', '--target_move_timestep', required=True, type=int)
parser.add_argument('-hls', '--hidden_layer_size', required=True, type=int)
parser.add_argument('-ecl', '--episode_cutoff_length', required=True, type=int)

args = parser.parse_args()

policy_stepsize = args.policy_stepsize
critic_stepsize = args.critic_stepsize
num_runs = args.num_runs
seed_number = args.seed_number
num_total_timesteps = args.num_total_timesteps
target_move_timestep = args.target_move_timestep
hidden_layer_size = args.hidden_layer_size
episode_cutoff_length = args.episode_cutoff_length

exp_config = args.exp_config.split(',')
theta_init_other_actions = float(exp_config[0])
theta_init_no_action = float(exp_config[1])
value_weight_init = float(exp_config[2])

theta_init = theta_init_other_actions * torch.ones(9)
theta_init[4] = theta_init_no_action

# folder details
folder_name = 'nn_data/DotReacher__theta_{}_{}__valueWeightInit_{}__numRuns_{}'\
    '__numTotalSteps_{}__episodeCutoff_{}__targetmoveTimestep_{}'\
    '__hiddenLayerSize_{}'.format(
        theta_init_other_actions, theta_init_no_action, value_weight_init,
        num_runs, num_total_timesteps, episode_cutoff_length,
        target_move_timestep, hidden_layer_size)
print(folder_name)
os.makedirs(folder_name, exist_ok='True')

#----------------------------------------------------------------------
# Using a learned value function
#----------------------------------------------------------------------
tic = time.time()
for FLAG_PG_TYPE in ['regular', 'alternate']:
    dat = run_experiment(seed_number=seed_number,
                         num_total_timesteps=num_total_timesteps,
                         policy_stepsize=policy_stepsize,
                         critic_stepsize=critic_stepsize,
                         FLAG_PG_TYPE=FLAG_PG_TYPE, theta_init=theta_init,
                         value_weight_init=value_weight_init,
                         hidden_layer_size=hidden_layer_size,
                         episode_cutoff_length=episode_cutoff_length,
                         target_move_timestep=target_move_timestep)

    filename='{}/pg_{}__pol_{}__val_{}__seed_{}'.format(
        folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
        seed_number)
    with open(filename, 'w') as fp:
        json.dump(dat, fp)

    print('PG:{} | alpha:{} | beta:{} | seed:{} | time:{}'.format(
        FLAG_PG_TYPE, policy_stepsize, critic_stepsize, seed_number,
        time.time() - tic))
    print('Total time: ', time.time() - tic)
