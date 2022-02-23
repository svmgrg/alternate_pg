import numpy as np
import pdb

class LinearChain():
    def __init__(self, P, r, start_state, terminal_states, reward_noise=0.3,
                 episode_cutoff_length=1000):
        self.P = P
        self.r = r
        self.reward_noise = reward_noise
        self.n = P.shape[-1]
        self.start_state = start_state
        self.terminal_states = terminal_states

        self.observation_space = self.n
        self.action_space = P.shape[0]
        self.state = None

        self.t = 0
        self.episode_cutoff_length = episode_cutoff_length

    def reset(self):
        self.state = self.start_state
        self.t = 0
        return self.state

    def step(self, action):
        if self.state is None:
            raise Exception('step() used before calling reset()')
        assert action in range(self.P.shape[0])

        reward = self.r[self.state, action] \
            + np.random.normal(loc=0, scale=self.reward_noise)
        self.state = np.random.choice(a=self.n, p=self.P[action, self.state])
        self.t = self.t + 1

        done = False
        if (self.state in self.terminal_states
            or self.t > self.episode_cutoff_length):
            done = True

        return self.state, reward, done, {}

    def calc_v_pi(self, pi, gamma):
        # calculate P_pi from the transition matrix P and the policy pi
        P_pi = np.zeros(self.P[0].shape)
        for a in range(pi.shape[1]):
            P_pi += self.P[a] * pi[:, a].reshape(-1, 1)

        # calculate the vector r_pi
        r_pi = (self.r * pi).sum(1).reshape(-1, 1)

        # calculate v_pi using the equation given above
        v_pi = np.matmul(
            np.linalg.inv(np.eye(self.P[0].shape[0]) - gamma * P_pi),
            r_pi)

        return v_pi

    def calc_q_pi(self, pi, gamma):
        # P_pi_control: SxA -> SxA
        P_pi_control = np.concatenate([pi[:, a] * np.concatenate(self.P)
                                       for a in range(self.action_space)], 1)
        sa_visitation = np.linalg.inv(np.eye(P_pi_control[0].shape[0]) \
                        - gamma * P_pi_control)
        r_sa = self.r.reshape(-1, 1, order='F')
        q_pi = np.matmul(sa_visitation, r_sa).reshape(
            -1, self.action_space, order='F')

        return q_pi

    def calc_d_gamma(self, pi, gamma):
        # calculate P_pi from the transition matrix P and the policy pi
        P_pi = np.zeros(self.P[0].shape)
        for a in range(pi.shape[1]):
            P_pi += self.P[a] * pi[:, a].reshape(-1, 1)

        # calculate d_gamma
        d_gamma = np.linalg.inv(np.eye(self.P[0].shape[0]) - gamma * P_pi)

        return d_gamma

class Agent():
    def __init__(self, num_actions, policy_features, value_features,
                 policy_stepsize, critic_stepsize, nstep, gamma,
                 FLAG_BASELINE, FLAG_PG_TYPE='regular',
                 policy_weight_init_left=0, policy_weight_init_right=0,
                 env=None, start_state=None, FLAG_POPULAR_PG=False,
                 value_weight_init=0, grad_bias=None, grad_noise=None):
        self.policy_features = policy_features
        self.value_features = value_features
        self.num_actions = num_actions

        self.policy_weight = np.zeros((policy_features.shape[1],
                                       num_actions))
        if num_actions == 2:
            self.policy_weight[:, 0] = policy_weight_init_left
            self.policy_weight[:, 1] = policy_weight_init_right
        if num_actions > 2:
            self.policy_weight[:, :] = policy_weight_init_left
            self.policy_weight[:, -1] = policy_weight_init_right

        self.value_weight_init = value_weight_init
        if value_features is None:
            self.value_weight = None
        else:
            self.value_weight \
                = self.value_weight_init * np.ones((value_features.shape[1], 1))

        self.policy_stepsize = policy_stepsize
        self.critic_stepsize = critic_stepsize

        self.FLAG_BASELINE = FLAG_BASELINE
        self.FLAG_POPULAR_PG = FLAG_POPULAR_PG
        self.FLAG_PG_TYPE = FLAG_PG_TYPE
        self.gamma = gamma
        self.nstep = nstep

        self.pi = None
        self.FLAG_POLICY_UPDATED = True

        self.env = env
        self.start_state = start_state

        self.grad_bias = grad_bias
        self.grad_noise = grad_noise

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
        out = e_x / e_x.sum(1).reshape(-1, 1)
        return out
        
    def take_action(self, state):
        if self.FLAG_POLICY_UPDATED:
            theta = np.matmul(self.policy_features, self.policy_weight)
            self.pi = self.softmax(theta)
            self.FLAG_POLICY_UPDATED = False
        action = np.random.choice(self.num_actions, p=self.pi[state])
        return action, self.pi[state, action]

    def calc_nstep_return(self, t, traj, v_pi):
        reward_list = traj['reward_list']
        next_state_list = traj['next_state_list']
        traj_length = len(reward_list)

        nstep = self.nstep
        assert nstep  == 'inf' or nstep > 0
        if nstep == 'inf' or nstep > traj_length:
            nstep = traj_length

        nstep_return = 0
        discount = 1
        for i in range(t, min(t+nstep, traj_length)):
            nstep_return += discount * reward_list[i]
            discount *= self.gamma
        i = min(t+nstep, traj_length) - 1
        nstep_return += discount * v_pi[next_state_list[i]]
        
        return nstep_return

    def pred_v_pi(self):
        return np.matmul(self.value_features, self.value_weight)

    def update_value_fn(self, traj):
        state_list = traj['state_list']
        traj_length = len(state_list)

        for t in range(traj_length):
            state = state_list[t]
            v_pi_pred = self.pred_v_pi()
            G = self.calc_nstep_return(t, traj, v_pi_pred)
                    
            pred = v_pi_pred[state]
            self.value_weight = self.value_weight \
                + self.critic_stepsize * (G - pred) \
                * self.value_features[state].reshape(self.value_weight.shape)

    def calc_grad_log_pi(self, state, action):
        x = self.policy_features[state].reshape(-1, 1)
        theta = np.matmul(x.T, self.policy_weight)
        pi = self.softmax(theta).T

        I_action = np.zeros((self.num_actions, 1))
        I_action[action] = 1

        one_vec = np.ones((1, self.num_actions))

        if self.FLAG_PG_TYPE == 'regular':
            grad = np.matmul(x, (I_action - pi).T) # equivalent
        elif self.FLAG_PG_TYPE == 'alternate':
            grad = np.matmul(x, I_action.T) # equivalent

        return grad

    def calc_expected_pg(self, method='expected'):
        theta = np.matmul(self.policy_features, self.policy_weight)
        pi = self.softmax(theta)
        d_gamma = self.env.calc_d_gamma(pi, self.gamma)[self.start_state]
        q_pi = self.env.calc_q_pi(pi, self.gamma)
        v_pi = self.env.calc_v_pi(pi, self.gamma)
            
        grad = np.zeros((self.policy_features.shape[1], self.num_actions))

        for s in range(self.env.P[0].shape[0]):
            x = self.policy_features[s].reshape(-1, 1)
            
            if method == 'expected':
                grad_s = np.matmul(x,
                                   (pi[s] * (q_pi[s] - v_pi[s])).reshape(1, -1))
            else:
                grad_s = np.zeros((policy_features.shape[1],
                                   self.num_actions))
                for a in range(num_actions):
                    I_action = np.zeros((self.num_actions, 1))
                    I_action[a] = 1

                    ## add qpi and vpi
                    if method == 'regular':
                        grad_s += pi[s][a] * np.matmul(
                            x, (I_action - pi[s].reshape(-1, 1)).T)
                    elif method == 'alternate':
                        grad_s += pi[s][a] * np.matmul(x, I_action.T)

            grad += d_gamma[s] * grad_s#(1 - self.gamma) * d_gamma[s] * grad_s

        return grad

    def calc_reinforce_pg(self, traj, v_pi):
        state_list = traj['state_list']
        action_list = traj['action_list']
        traj_length = len(state_list)
        
        policy_grad = np.zeros(self.policy_weight.shape)
        discounting = 1
        for t in range(traj_length):
            state = state_list[t]
            action = action_list[t]
            G = self.calc_nstep_return(t, traj, v_pi)
            grad_log_pi = self.calc_grad_log_pi(state, action)
            baseline = v_pi[state] if self.FLAG_BASELINE else 0

            policy_grad += discounting * (G - baseline) * grad_log_pi

            if self.FLAG_POPULAR_PG == False:
                discounting *= self.gamma

        return policy_grad
    
    def update_policy(self, traj=None, v_pi=None):
        if self.FLAG_PG_TYPE in ['regular', 'alternate']:
            policy_grad = self.calc_reinforce_pg(traj, v_pi)
        elif self.FLAG_PG_TYPE == 'expected':
            policy_grad = self.calc_expected_pg()

        if self.grad_bias is not None or self.grad_noise is not None:
            policy_grad += np.random.normal(loc=self.grad_bias,
                                            scale=self.grad_noise,
                                            size=policy_grad.shape)
        
        self.policy_weight += self.policy_stepsize * policy_grad

        self.FLAG_POLICY_UPDATED = True

        theta = np.matmul(self.policy_features, self.policy_weight)
        pi = self.softmax(theta)
        expected_ret = self.env.calc_v_pi(
            pi, self.gamma)[self.start_state].item()

        return expected_ret


def run_experiment(num_runs, num_episodes,
                   P, r, start_state, terminal_states,
                   num_actions, policy_features, value_features,
                   policy_stepsize, critic_stepsize, nstep, gamma,
                   FLAG_BASELINE, FLAG_LEARN_VPI, FLAG_PG_TYPE,
                   reward_noise=0, vpi_bias=0,
                   policy_weight_init_left=0, policy_weight_init_right=0,
                   value_weight_init=0, episode_cutoff_length=1000,
                   grad_bias=None, grad_noise=None, FLAG_FIXED_VPI_OPTIM=False):
    return_across_runs = []
    ep_len_across_runs = []
    vpi_across_runs = []

    for run in range(num_runs):
        np.random.seed(run)

        # define the agent and the environment
        env = LinearChain(P=P, r=r, start_state=start_state,
                          terminal_states=terminal_states,
                          reward_noise=reward_noise,
                          episode_cutoff_length=episode_cutoff_length)

        agent = Agent(num_actions=num_actions,
                      policy_features=policy_features,
                      value_features=value_features,
                      policy_stepsize=policy_stepsize,
                      critic_stepsize=critic_stepsize,
                      nstep=nstep, gamma=gamma,
                      FLAG_BASELINE=FLAG_BASELINE,
                      FLAG_PG_TYPE=FLAG_PG_TYPE,
                      policy_weight_init_left=policy_weight_init_left,
                      policy_weight_init_right=policy_weight_init_right,
                      env=env, start_state=start_state,
                      FLAG_POPULAR_PG=False,
                      value_weight_init=value_weight_init,
                      grad_bias=grad_bias, grad_noise=grad_noise)

        return_across_episodes = []
        ep_len_across_episodes = []
        vpi_across_episodes = []

        episode_i = 0
        while episode_i < num_episodes:
            episode_i += 1

            if FLAG_PG_TYPE in ['regular', 'alternate']:
                traj = {'state_list': [],
                        'action_list': [],
                        'action_prob_list': [],
                        'reward_list': [],
                        'next_state_list': []}

                # sample a trajectory by following the current policy
                done = False
                state = env.reset()
                while not done:
                    action, action_prob = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)

                    traj['state_list'].append(state)
                    traj['action_list'].append(action)
                    traj['action_prob_list'].append(action_prob)
                    traj['reward_list'].append(reward)
                    traj['next_state_list'].append(next_state)

                    state = next_state

                # evaluate the current policy pi
                if FLAG_FIXED_VPI_OPTIM:
                    v_pi = agent.pred_v_pi() + vpi_bias
                else:
                    if FLAG_LEARN_VPI:
                        v_pi = agent.pred_v_pi() + vpi_bias
                    else:
                        v_pi = env.calc_v_pi(agent.pi, gamma)

                # update the policy
                expected_ret = agent.update_policy(traj, v_pi)

                if np.isnan(agent.policy_weight).any() == True:
                    # if policy is nan, kill the training and give bad return
                    tmp = num_episodes - episode_i + 1
                    return_across_episodes += [-1/(1 - gamma)] * tmp
                    vpi_across_episodes += [-1/(1 - gamma)] * tmp
                    ep_len_across_episodes += [1] * tmp
                    break
                else:
                    # update the value function
                    if FLAG_LEARN_VPI:
                        agent.update_value_fn(traj)

                    # save the returns for this episode
                    factor = 1
                    discounted_return = 0
                    for reward in traj['reward_list']:
                        discounted_return += factor * reward
                        factor *= gamma

                    return_across_episodes.append(discounted_return)
                    vpi_across_episodes.append(expected_ret)
                    ep_len_across_episodes.append(len(traj['reward_list']))

            elif FLAG_PG_TYPE == 'expected':
                # update the policy
                expected_ret = agent.update_policy(traj=None, v_pi=None)

                # save the returns for this episode
                vpi_across_episodes.append(expected_ret)
        
        return_across_runs.append(return_across_episodes)
        vpi_across_runs.append(vpi_across_episodes)
        ep_len_across_runs.append(ep_len_across_episodes)

    dat = {'returns': return_across_runs,
           'ep_len': ep_len_across_runs,
           'vpi_s0': vpi_across_runs}

    return dat
