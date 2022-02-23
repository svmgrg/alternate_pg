import math
import json
import time
import os
import pdb
import argparse

from mdp_experiments import Agent, run_experiment
from PyFixedReps import TileCoder

# $ python run_exp.py --env_name 'GymMountainCar' --exp_config '0,0,0,0,None,None' --policy_stepsize 0.5 --critic_stepsize 0.1 --escort_p 0 --ent_tau 0 --num_runs 1 --seed_number 0 --gamma 1 --num_total_timesteps 100000 --state_aggregation 0 --switch_actions 1 --capture_entropy 1 --pg_type 'alternate' --policy_mapping 'softmax'

# read command line arguments to get parameter configurations
parser = argparse.ArgumentParser()
parser.add_argument('-en', '--env_name', required=True, type=str)
parser.add_argument('-ex', '--exp_config', required=True, type=str)
parser.add_argument('-al', '--policy_stepsize', required=True, type=float)
parser.add_argument('-be', '--critic_stepsize', required=True, type=float)
parser.add_argument('-ep', '--escort_p', required=True, type=float)
parser.add_argument('-et', '--ent_tau', required=True, type=float)
parser.add_argument('-nr', '--num_runs', required=True, type=int)
parser.add_argument('-s', '--seed_number', required=True, type=int)
parser.add_argument('-g', '--gamma', required=True, type=float)
parser.add_argument('-nts', '--num_total_timesteps', required=True, type=int)

parser.add_argument('-sagg', '--state_aggregation', required=True, type=int)
parser.add_argument('-sact', '--switch_actions', required=True, type=int)
parser.add_argument('-cent', '--capture_entropy', required=True, type=int)

parser.add_argument('-pgt', '--pg_type', required=True, type=str)
parser.add_argument('-pm', '--policy_mapping', required=True, type=str)

args = parser.parse_args()

env_name = args.env_name

exp_config = args.exp_config.split(',')
theta_init_0 = float(exp_config[0])
theta_init_1 = float(exp_config[1])
theta_init_2 = float(exp_config[2])
value_weight_init = float(exp_config[3])
grad_bias = None if exp_config[4] == 'None' else float(exp_config[4])
grad_noise = None if exp_config[5] == 'None' else float(exp_config[5])
gamma = args.gamma

if ((args.state_aggregation not in [0, 1]) \
    or (args.switch_actions not in [0, 1]) \
    or (args.capture_entropy not in [0, 1])):
    raise IncorrectArgumentsPassedError()
    
FLAG_STATE_AGGREGATION = True if args.state_aggregation == 1 else False
FLAG_SWITCH_ACTIONS = True if args.switch_actions == 1 else False
FLAG_CAPTURE_ENTROPY = True if args.capture_entropy == 1 else False 

policy_stepsize = args.policy_stepsize
critic_stepsize = args.critic_stepsize
escort_p = args.escort_p
ent_tau = args.ent_tau
num_runs = args.num_runs
seed_number = args.seed_number

FLAG_PG_TYPE = args.pg_type
FLAG_POLICY_MAPPING = args.policy_mapping

# print('env_name ', env_name, type(env_name), '\n',
#       'theta_init_0 ', theta_init_0, type(theta_init_0), '\n',
#       'theta_init_1 ', theta_init_1, type(theta_init_1), '\n',
#       'theta_init_2 ', theta_init_2, type(theta_init_2), '\n',
#       'value_weight_init ', value_weight_init, type(value_weight_init), '\n',
#       'grad_bias ', grad_bias, type(grad_bias), '\n',
#       'grad_noise ', grad_noise, type(grad_noise), '\n',
#       'policy_stepsize ', policy_stepsize, type(policy_stepsize), '\n',
#       'critic_stepsize ', critic_stepsize, type(critic_stepsize), '\n',
#       'num_runs ', num_runs, type(num_runs), '\n',)

num_total_timesteps = args.num_total_timesteps
episode_cutoff_length = 1000

FLAG_BASELINE = True
FLAG_POPULAR_PG = True
FLAG_LEARN_VPI = True

# folder details
folder_name = 'mdp_data/{}_{}__Tiles__theta_{}_{}_{}__valueWeightInit_{}'\
    '__numRuns_{}__numTotalSteps_{}__gamma_{}__episodeCutoff_{}__gradBias_{}'\
    '__gradNoise_{}__flagBaseline_{}__flagPopularPG_{}__flagLearnVpi'.format(
        env_name, FLAG_POLICY_MAPPING, 
        theta_init_0, theta_init_1, theta_init_2, value_weight_init,
        num_runs, num_total_timesteps, gamma, episode_cutoff_length,
        grad_bias, grad_noise, FLAG_BASELINE, FLAG_POPULAR_PG, FLAG_LEARN_VPI)
print(folder_name)
os.makedirs(folder_name, exist_ok='True')

# create the environment
if env_name == 'MountainCar' or env_name == 'GymMountainCar':
    tilecoder = TileCoder({'tiles': 4, 'tilings': 8, 'dims': 2,
                           'random_offset': True,
                           'input_ranges': [(-1.2, 0.5), (-0.07, 0.07)]})
elif env_name == 'Acrobot':
    ma_vel1 = 4 * math.pi
    ma_vel2 = 9 * math.pi
    tilecoder = TileCoder({'tiles': 4, 'tilings': 8, 'dims': 6,
                           'random_offset': True,
                           'input_ranges': [(-1, 1), (-1, 1), (-1, 1), (-1, 1),
                                            (-ma_vel1, ma_vel1),
                                            (-ma_vel2, ma_vel2)]})
elif env_name == 'Cartpole':
    ma_vel1 = 4 * math.pi
    ma_vel2 = 9 * math.pi
    theta_thresh = 12 * 2 * math.pi / 360
    tilecoder = TileCoder({'tiles': 4, 'tilings': 8, 'dims': 4,
                           'random_offset': True,
                           'input_ranges': [(-2.4, 2.4), (-10, 10),
                                            (-theta_thresh, theta_thresh),
                                            (-10, 10)]})
else:
    raise NotImplementedError()

if FLAG_STATE_AGGREGATION:
    if env_name == 'MountainCar' or env_name == 'GymMountainCar':
        tilecoder = TileCoder({'tiles': 10, 'tilings': 1, 'dims': 2,
                               'random_offset': True,
                               'input_ranges': [(-1.2, 0.5), (-0.07, 0.07)]})
    elif env_name == 'Acrobot':
        ma_vel1 = 4 * math.pi
        ma_vel2 = 9 * math.pi
        tilecoder = TileCoder({'tiles': 10, 'tilings': 1, 'dims': 6,
                               'random_offset': True,
                               'input_ranges': [(-1, 1), (-1, 1),
                                                (-1, 1), (-1, 1),
                                                (-ma_vel1, ma_vel1),
                                                (-ma_vel2, ma_vel2)]})
    else:
        raise NotImplementedError()

#----------------------------------------------------------------------
# Using a learned value function
#----------------------------------------------------------------------
tic = time.time()

dat = run_experiment(
    env_name=env_name, seed_number=seed_number,
    num_total_timesteps=num_total_timesteps, tilecoder=tilecoder,
    policy_stepsize=policy_stepsize, critic_stepsize=critic_stepsize,
    gamma=gamma, FLAG_BASELINE=FLAG_BASELINE,
    FLAG_LEARN_VPI=FLAG_LEARN_VPI,
    FLAG_PG_TYPE=FLAG_PG_TYPE, FLAG_POLICY_MAPPING=FLAG_POLICY_MAPPING,
    FLAG_POPULAR_PG=FLAG_POPULAR_PG, theta_init_0=theta_init_0,
    theta_init_1=theta_init_1, theta_init_2=theta_init_2,
    value_weight_init=value_weight_init,
    episode_cutoff_length=episode_cutoff_length,
    escort_p=escort_p, ent_tau=ent_tau,
    grad_bias=grad_bias, grad_noise=grad_noise,
    FLAG_SWITCH_ACTIONS=FLAG_SWITCH_ACTIONS,
    FLAG_CAPTURE_ENTROPY=FLAG_CAPTURE_ENTROPY)

if (not FLAG_SWITCH_ACTIONS) \
   and (not FLAG_STATE_AGGREGATION) \
   and (not FLAG_CAPTURE_ENTROPY):
    filename='{}/pg_{}__pol_{}__val_{}__seed_{}'.format(
        folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
        seed_number)
else:
    filename='{}/pg_{}__pol_{}__val_{}__escort_p_{}__ent_tau_{}__'\
        '__seed_{}__switchActions_{}'\
        '__stateAggregation_{}__captureEntropy_{}'.format(
            folder_name, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
            escort_p, ent_tau,
            seed_number, FLAG_SWITCH_ACTIONS, FLAG_STATE_AGGREGATION,
            FLAG_CAPTURE_ENTROPY) 
with open(filename, 'w') as fp:
    json.dump(dat, fp)

print('map:{} | PG:{} | alpha:{} | beta:{} | tau:{} | p:{} | seed:{}'.format(
    FLAG_POLICY_MAPPING, FLAG_PG_TYPE, policy_stepsize, critic_stepsize,
    escort_p, ent_tau, seed_number))
print('Total time: ', time.time() - tic)
