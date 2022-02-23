#!/bin/bash
# SLURM submission script for submitting multiple serial jobs on Niagara
#
# taken from here: https://github.com/sinaghiassian/OffpolicyAlgorithms/blob/master/Job/SubmitJobsTemplates.SL
#
#SBATCH --account=def-XXX
#SBATCH --time=01:00:00
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=80
#SBATCH --job-name OnlineAC

env_name_list=('Acrobot')

exp_config_list=('0,0,0,0,None,None')

policy_stepsize_list=(0.0001220703125 0.00048828125 0.001953125 0.0078125 0.03125 0.125 0.5 2 8 32)
critic_stepsize_list=(0.01 0.05 0.1 0.5 1 2)
ent_tau_list=(0.0)# 0.1 0.3 0.5 0.7 0.9)

gamma=(1)

num_runs=(50)
seed_number_list=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49)

num_total_timesteps=(400000)

state_aggregation=(0)
switch_actions=(1)
capture_entropy=(1)

module load NiaEnv/2019b
module load gnu-parallel
module load python/3.8.5
source ANDY_STUFF/bin/activate

cd $SLURM_SUBMIT_DIR || exit
export OMP_NUM_THREADS=1

echo "Current working directory is $(pwd)"
echo "Running on hostname $(hostname)"
echo "Starting run at: $(date)"

HOSTS=$(scontrol show hostnames $SLURM_NODELIST | tr '\n' ,)
NCORES=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

parallel --env OMP_NUM_THREADS,PATH,LD_LIBRARY_PATH --joblog slurm-$SLURM_JOBID.log -j $NCORES -S $HOSTS --wd $PWD \
	 python run_exp.py ::: \
	 --env_name ::: ${env_name_list[@]} ::: \
	 --exp_config ::: ${exp_config_list[@]} ::: \
	 --policy_stepsize ::: ${policy_stepsize_list[@]} ::: \
	 --critic_stepsize ::: ${critic_stepsize_list[@]} ::: \
	 --ent_tau ::: ${ent_tau_list[@]} ::: \
	 --num_runs ::: ${num_runs} ::: \
	 --gamma ::: ${gamma} ::: \
	 --seed_number ::: ${seed_number_list[@]} ::: \
	 --num_total_timesteps ::: ${num_total_timesteps} ::: \
	 --state_aggregation ::: ${state_aggregation[@]} ::: \
	 --switch_actions ::: ${switch_actions[@]} ::: \
	 --capture_entropy ::: ${capture_entropy[@]}

echo "Program test finished with exit code $? at: $(date)"

# $ sbatch job_submit_acrobot.sh

