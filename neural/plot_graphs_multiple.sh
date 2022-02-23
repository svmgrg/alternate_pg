env_name=('Acrobot')

exp_config_list=('0,0,0' '0,5,0' '0,10,0' '0,20,0' '0,0,25' '0,5,25' '0,10,25' '0,20,25' '0,0,-25' '0,5,-25' '0,10,-25' '0,20,-25')
#('0,10,0' '0,15,0' '0,20,0' '0,0,10' '0,5,10' '0,10,10' '0,15,10' '0,20,10' '0,0,-10' '0,5,-10' '0,10,-10' '0,15,-10' '0,20,-10' '0,0,20' '0,5,20' '0,10,20' '0,15,20' '0,20,20' '0,0,-20' '0,5,-20' '0,10,-20' '0,15,-20' '0,20,-20')

num_runs=(50)

for exp_config in ${exp_config_list[@]}
do
    python plot_graphs.py --exp_config ${exp_config} --num_runs ${num_runs} --num_total_timesteps 50000 --target_move_timestep 0 --hidden_layer_size 10 --episode_cutoff_length 1000
done
