# Actor critic agent
import torch
import matplotlib.pyplot as plt

# Problem
torch.manual_seed(1)
LB = torch.tensor([-1, -1], dtype=torch.float32)
UB = torch.tensor([+1, +1], dtype=torch.float32)

# Agent
num_actions = 9
dim_states = 2
action_values = 0.03 * torch.tensor([[-1, -1], [+0, +1], [+1, +1],
                                     [-1, +0], [+0, +0], [+1, +0],
                                     [-1, -1], [+0, -1], [+1, -1]],
                                    dtype=torch.float32)
hidden_layer_size = 10
alpha = 0.0003

actor_body = torch.nn.Sequential(
    torch.nn.Linear(dim_states, hidden_layer_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer_size, hidden_layer_size),
    torch.nn.ReLU())

actor_pref = torch.nn.Sequential(
    torch.nn.Linear(hidden_layer_size, num_actions))
actor_pref[-1].weight.data[:] = 0
actor_pref[-1].bias.data[:] = 0

critic = torch.nn.Sequential(
    torch.nn.Linear(dim_states, hidden_layer_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer_size, hidden_layer_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer_size, 1))

policy_optim = torch.optim.Adam(
    list(actor_body.parameters()) + list(actor_pref.parameters()),
    lr=alpha)
critic_optim = torch.optim.Adam(critic.parameters(),
                                lr=10*alpha)

# Experiment
num_episodes = 600
list_return = []
Slogs = []
i = 0
for episode_i in range(num_episodes):
    Slogs.append([])
    state = torch.rand((1, dim_states))*(UB-LB) + LB
    Slogs[-1].append(state)
    total_return = 0
    
    while True:
        # Take action
        state_feature = actor_body(state)
        action_pref = actor_pref(state_feature)
        
        try:
            pi_state = torch.distributions.Categorical(logits=action_pref)
        except:
            print("E", action_pref)
            
        action = pi_state.sample()

        # Receive reward and next state
        noise = torch.rand(dim_states) * 0.06 - 0.03
        next_state = torch.clamp(state + action_values[action] + 0 * noise,
                                 LB, UB)
        R = - 0.01
        done = torch.allclose(next_state, torch.zeros(dim_states), atol=0.2)

        # Learning
        value_pred_state = critic(state)
        value_pred_next_state = critic(next_state)
        pobj = pi_state.log_prob(action) \
            * (R + (1 - done) * value_pred_next_state \
               - value_pred_state).detach()
        policy_loss = -pobj
        critic_loss = (R + (1 - done) * value_pred_next_state.detach() \
                       - value_pred_state)**2
        
        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()
        
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        # Log
        Slogs[-1].append(next_state)
        total_return += R

        # Termination
        if done:
            print(value_pred_next_state)
            list_return.append(total_return)
            i += 1
            print(i, len(Slogs[-1]))
            break
        state = next_state

# Plotting
plt.plot(-100 * torch.tensor(list_return))
plt.figure()

color_list = ['tab:blue', 'tab:green', 'tab:orange',
              'tab:purple', 'tab:red', 'tab:brown']
for i in range(-min(30, num_episodes), 0):
    Slog = torch.cat(Slogs[i])
    for i in range(Slog.shape[0]-1):
        plt.plot(Slog[i:i+2, 0], Slog[i:i+2, 1],
                 alpha=(i + 1) / Slog.shape[0],
                 color=color_list[i % len(color_list)])
        
plt.xlim([LB[0], UB[0]])
plt.ylim([LB[1], UB[1]])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.show()
