env_name=('MountainCar')

exp_config_list=('0,0,0,0,None,None' '0,0,0,+500,None,None' '0,0,0,-500,None,None')

# exp_config_list=('0,0,0,0,None,None' '0,0,0,+500,None,None' '0,0,0,-500,None,None' '0,5,0,0,None,None' '0,5,0,+500,None,None' '0,5,0,-500,None,None' '0,10,0,0,None,None' '0,10,0,+500,None,None' '0,10,0,-500,None,None' '0,100,0,0,None,None' '0,100,0,+500,None,None' '0,100,0,-500,None,None')

num_runs=(3)

for exp_config in ${exp_config_list[@]}
do
    python plot_graphs.py --env_name ${env_name} --exp_config ${exp_config} --num_runs ${num_runs}
done
