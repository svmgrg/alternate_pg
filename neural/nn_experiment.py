import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt

class DotReacher():
    def __init__(self, target_state=torch.zeros(2), episode_cutoff_length=1000):
        self.num_actions = 9
        self.dim_states = 2
        self.LB = torch.tensor([-1, -1], dtype=torch.float32)
        self.UB = torch.tensor([+1, +1], dtype=torch.float32)
        self.action_values = 0.03 * torch.tensor([[-1, -1], [+0, +1], [+1, +1],
                                                  [-1, +0], [+0, +0], [+1, +0],
                                                  [-1, -1], [+0, -1], [+1, -1]],
                                                 dtype=torch.float32)
        self.target_state = target_state
        self.episode_cutoff_length = episode_cutoff_length
        
        self.state= None
        self.t = 0

    def reset(self):
        self.state = self.LB \
            + torch.rand((1, self.dim_states)) * (self.UB - self.LB)
        self.t = 0
        return self.state

    def step(self, action):
        noise = torch.rand(self.dim_states) * 0.06 - 0.03
        self.state = torch.clamp(self.state + self.action_values[action] \
                                 + 0 * noise, self.LB, self.UB)
        reward = - 0.01
        self.t += 1
        
        if torch.allclose(self.state, self.target_state, atol=0.1):
            done = 'terminal'
        elif self.t > self.episode_cutoff_length:
            done = 'cutoff'
        else:
            done = False
            
        return self.state, reward, done

class OnlineAC_NN():
    def __init__(self, num_actions, dim_states, policy_stepsize,
                 critic_stepsize, hidden_layer_size, FLAG_PG_TYPE,
                 theta_init, value_weight_init):
        self.num_actions = num_actions
        self.dim_states = dim_states
        self.policy_stepsize = policy_stepsize
        self.critic_stepsize = critic_stepsize
        self.hidden_layer_size = hidden_layer_size
        self.FLAG_PG_TYPE = FLAG_PG_TYPE
        self.theta_init = theta_init
        self.value_weight_init = value_weight_init

        self.actor_body = torch.nn.Sequential(
            torch.nn.Linear(self.dim_states, self.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            torch.nn.ReLU())

        self.actor_pref = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_layer_size, self.num_actions))
        self.actor_pref[-1].weight.data[:] = 0
        self.actor_pref[-1].bias.data[:] = self.theta_init

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(self.dim_states, self.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_size, 1))
        self.critic[-1].weight.data[:] = 0
        self.critic[-1].bias.data[:] = self.value_weight_init
        
        self.policy_optim = torch.optim.Adam(
            list(self.actor_body.parameters()) \
            + list(self.actor_pref.parameters()),
            lr=self.policy_stepsize)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=self.critic_stepsize)

    def take_action(self, state):
        with torch.no_grad():
            state_feature = self.actor_body(state)
            action_preferences = self.actor_pref(state_feature)
            pi = torch.distributions.Categorical(logits=action_preferences)
            action = pi.sample()
            entropy = pi.entropy().item()
        return action, entropy

    def predict_value(self, state):
        with torch.no_grad():
            value_pred_state = self.critic(state)
        return value_pred_state

    def update(self, state, action, reward, next_state, done):
        #------------------------------------------------------------
        # critic loss
        #------------------------------------------------------------
        value_pred_state = self.critic(state)
        if done == 'terminal':
            value_pred_next_state = 0
        elif done == 'cutoff' or done == False:
            with torch.no_grad():
                value_pred_next_state = self.critic(next_state)
        else:
            raise NotImplementedError()
        target = reward + value_pred_next_state
        critic_loss = (target - value_pred_state)**2

        #------------------------------------------------------------
        # policy loss
        #------------------------------------------------------------
        state_feature = self.actor_body(state)
        action_preferences = self.actor_pref(state_feature)
        pi = torch.distributions.Categorical(logits=action_preferences)
        log_prob = pi.log_prob(action)        
        delta = reward + value_pred_next_state - value_pred_state.detach()
        if self.FLAG_PG_TYPE == 'regular':
            policy_objective = log_prob * delta
        elif self.FLAG_PG_TYPE == 'alternate':
            policy_objective = action_preferences.flatten()[action] * delta
        else:
            raise NotImplementedError()
        policy_loss = -1 * policy_objective

        #------------------------------------------------------------
        # update the neural networks
        #------------------------------------------------------------
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
    def any_param_nan(self):
        if torch.isinf(self.actor_pref[-1].bias.data[:].sum()) \
           or torch.isinf(self.critic[-1].bias.data[:].sum()):
            return True
        else:
            return False

def run_experiment(seed_number, num_total_timesteps, policy_stepsize,
                   critic_stepsize, FLAG_PG_TYPE, theta_init,
                   value_weight_init, hidden_layer_size,
                   episode_cutoff_length, target_move_timestep):

    torch.manual_seed(seed_number)

    if target_move_timestep == 0:
        env_first = DotReacher(
            target_state=torch.tensor([+0, +0], dtype=torch.float32),
            episode_cutoff_length=episode_cutoff_length)
    else:
        env_first = DotReacher(
            target_state=torch.tensor([-1, -1], dtype=torch.float32),
            episode_cutoff_length=episode_cutoff_length)
        env_second = DotReacher(
            target_state=torch.tensor([+1, +1], dtype=torch.float32),
            episode_cutoff_length=episode_cutoff_length)
    
    agent = OnlineAC_NN(num_actions=env_first.num_actions,
                        dim_states=env_first.dim_states,
                        policy_stepsize=policy_stepsize,
                        critic_stepsize=critic_stepsize,
                        hidden_layer_size=hidden_layer_size,
                        FLAG_PG_TYPE=FLAG_PG_TYPE,
                        theta_init=theta_init,
                        value_weight_init=value_weight_init)
    
    return_across_episodes = []
    ep_len_across_episodes = []
    vpi_s0_across_episodes = []
    entropy_across_episodes = []

    episode_i = 0
    total_timesteps = 0

    FLAG_NAN_ENCOUNTERED_ABORT = False

    while total_timesteps < num_total_timesteps:
        if target_move_timestep > 0 and total_timesteps > target_move_timestep:
            env = env_second
        else:
            env = env_first
            
        episode_i += 1

        # sample a trajectory by following the current policy
        done = False
        state = env.reset()
        start_state = state.clone() # keep the start state for plotting vpi_pred

        time_per_episode = 0
        undiscounted_return = 0
        entropy_for_one_episode = []
        while done == False:
            action, entropy_this_step = agent.take_action(state)
            next_state, reward, done = env.step(action)

            # update the policy and the value function
            agent.update(state, action, reward, next_state, done)

            state = next_state
            time_per_episode += 1
            undiscounted_return += reward
            entropy_for_one_episode.append(entropy_this_step)

            # if policy is infinite, kill the training and give bad return
            # assume that the agent will be unable to solve the task in rest of
            # the episodes, and therefore calculate the remaining episodes it
            # will go through based on episode_cutoff_length
            if agent.any_param_nan():
                print('\n\n=============================================\n\n')
                remaining_ep_count = (num_total_timesteps - total_timesteps) \
                    / episode_cutoff_length
                remaining_ep_count = math.ceil(remaining_ep_count)
                max_negative_return = -1 / (1 - gamma) if gamma < 1 else -1e8
                return_across_episodes += \
                    [max_negative_return] * remaining_ep_count
                vpi_s0_across_episodes += \
                    [max_negative_return] * remaining_ep_count
                ep_len_across_episodes += \
                    [episode_cutoff_length] * remaining_ep_count
                entropy_across_episodes += [0] * remaining_ep_count
                FLAG_NAN_ENCOUNTERED_ABORT = True
                break
                
        if FLAG_NAN_ENCOUNTERED_ABORT:
            FLAG_NAN_ENCOUNTERED_ABORT = False
            break
        else:
            # save the returns, ep_len, vpi_s0 for this episode
            return_across_episodes.append(undiscounted_return)
            ep_len_across_episodes.append(time_per_episode)
            vpi_s0_across_episodes.append(
                agent.predict_value(start_state).item())
            entropy_across_episodes.append(np.mean(entropy_for_one_episode))
            print(episode_i, time_per_episode)

        total_timesteps += time_per_episode

    dat = {'returns': return_across_episodes,
           'ep_len': ep_len_across_episodes,
           'vpi_s0': vpi_s0_across_episodes,
           'entropy': entropy_across_episodes}

    return dat
