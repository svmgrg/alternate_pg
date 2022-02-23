import numpy as np
from scipy.stats import entropy
import random
import math
import pdb

from PyRlEnvs.domains.MountainCar import GymMountainCar, MountainCar
from PyRlEnvs.domains.Acrobot import Acrobot
from PyRlEnvs.domains.Cartpole import Cartpole

class Agent():
    def __init__(self, num_actions, policy_stepsize, critic_stepsize, gamma,
                 FLAG_BASELINE, FLAG_PG_TYPE='regular',
                 FLAG_POLICY_MAPPING='softmax', FLAG_POPULAR_PG=False,
                 theta_init_0=0, theta_init_1=0, theta_init_2=0, ent_tau=0,
                 escort_p=None,
                 value_weight_init=0, grad_bias=None, grad_noise=None,
                 tilecoder=None):
        self.num_actions = num_actions
        
        self.policy_stepsize = policy_stepsize
        self.critic_stepsize = critic_stepsize
        self.gamma = gamma

        self.FLAG_BASELINE = FLAG_BASELINE
        self.FLAG_PG_TYPE = FLAG_PG_TYPE
        self.FLAG_POLICY_MAPPING = FLAG_POLICY_MAPPING
        self.FLAG_POPULAR_PG = FLAG_POPULAR_PG

        self.tilecoder = tilecoder
        # add a one to the length for a constant "always on" bias unit
        self.state_feature_length = tilecoder.total_tiles + 1
        # Andy's tilecoder scales the tilecoding features so that the
        # the stepsizes don't have to be manually adjusted later
        # (for reference, see Sec 9.5.4 Tile Coding, Sutton and Barto, 2018)
        # the reduced_state_feat is the same for all states!
        num_active_state_features = tilecoder.num_tiling + 1
        individual_feature_magnitude = 1 / num_active_state_features
        self.reduced_state_feat = individual_feature_magnitude \
            * np.ones((num_active_state_features, 1))

        # Attention! Only works when the environment has three (or two) actions
        self.policy_weight = np.zeros((self.state_feature_length, num_actions))
        self.policy_weight[:, 0] = theta_init_0
        self.policy_weight[:, 1] = theta_init_1
        if num_actions > 2:
            self.policy_weight[:, 2] = theta_init_2

        self.value_weight_init = value_weight_init
        self.value_weight = value_weight_init \
            * np.ones((self.state_feature_length, 1))

        self.ent_tau = ent_tau
        self.escort_p = escort_p
        
        self.grad_bias = grad_bias
        self.grad_noise = grad_noise

        self.pi_current_state = None # policy vector for the current state
        self.FLAG_POLICY_WEIGHTS_UPDATED = True

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum()
        return out

    def escort(self, x):
        x_p = np.abs(x)**(self.escort_p)
        out = x_p / x_p.sum()
        return out

    def get_state_features(self, state):
        state_feat = np.array(self.tilecoder.encode(state)).reshape(-1, 1)
        # add an "always on" bias feature to the feature vector
        state_feat = np.append(state_feat, [[1]], axis=0) 
        return state_feat

    def get_state_features_indices(self, state):
        idx = self.tilecoder.get_indices(state)
        # add the "always on" bias index
        idx = np.append(idx, self.state_feature_length - 1) 
        return idx

    def take_action(self, state):
        if self.FLAG_POLICY_WEIGHTS_UPDATED:
            idx = self.get_state_features_indices(state)
            theta = np.matmul(self.policy_weight[idx].T,
                              self.reduced_state_feat).flatten()
            
            # try: # testing code
            #     theta_tmp = np.matmul(self.policy_weight.T,
            #                        self.get_state_features(state)).flatten()
            #     assert np.allclose(theta_tmp, theta)
            # except:
            #     pdb.set_trace()

            if self.FLAG_POLICY_MAPPING == 'softmax':
                self.pi_current_state = self.softmax(theta)
            elif self.FLAG_POLICY_MAPPING == 'escort':
                self.pi_current_state = self.escort(theta)
            else:
                raise NotImplementedError()
            self.FLAG_POLICY_WEIGHTS_UPDATED = False

        action = np.random.choice(self.num_actions, p=self.pi_current_state)
        return action, self.pi_current_state[action]

    def calc_policy_entropy(self):
        return entropy(self.pi_current_state)

    def pred_v_pi(self, state):
        idx = self.get_state_features_indices(state)
        v_pred = np.matmul(self.value_weight[idx].T,
                           self.reduced_state_feat).flatten()
        
        # try: # testing code
        #     v_pred_tmp = np.matmul(self.value_weight.T,
        #                     self.get_state_features(state)).flatten()
        #     assert np.allclose(v_pred_tmp, v_pred)
        # except:
        # print(v_pred)
        return v_pred

    def update_value_fn(self, state, delta):
        idx = self.get_state_features_indices(state)
        grad_idx = delta * self.reduced_state_feat
        
        # try: # testing code
        #     grad_tmp = delta * self.get_state_features(state)
        #     assert np.allclose(grad_tmp[idx], grad_idx)
        #     assert math.isclose(np.abs(grad_tmp[idx]).sum(),
        #                         np.abs(grad_tmp).sum())
        # except: 
        #     pdb.set_trace()
            
        self.value_weight[idx] += self.critic_stepsize * grad_idx
    
    def update_policy(self, state, action, delta, discounting_term):
        idx = self.get_state_features_indices(state)
        theta = np.matmul(self.policy_weight[idx].T,
                          self.reduced_state_feat).flatten()
        # theta = self.policy_weight[idx].sum(0)
        
        I_action = np.zeros((self.num_actions, 1))
        I_action[action] = 1

        if self.FLAG_POLICY_MAPPING == 'softmax':
            pi_current_state = self.softmax(theta).reshape(-1, 1)
            if self.FLAG_PG_TYPE == 'regular':
                grad_log_pi_idx = np.matmul(self.reduced_state_feat,
                                            (I_action - pi_current_state).T)
            elif self.FLAG_PG_TYPE == 'alternate':
                grad_log_pi_idx = np.matmul(self.reduced_state_feat, I_action.T)

            # add gradient of the entopy term
            ent_pi_state = entropy(pi_current_state).item()
            
            # numerically stable way of computing vec1 = pi * log(pi)
            vec1 = np.zeros(pi_current_state.shape)
            rdx = np.nonzero(pi_current_state)
            vec1[rdx] = pi_current_state[rdx] * np.log(pi_current_state[rdx])
            vec2 = pi_current_state * ent_pi_state 
            grad_entropy_idx = -1 * np.matmul(self.reduced_state_feat,
                                              (vec1 + vec2).T)
        elif self.FLAG_POLICY_MAPPING == 'escort':
            escort_denom = (np.abs(theta)**(self.escort_p)).sum()
            tmp_vec = (np.sign(theta) \
                       * np.abs(theta)**(self.escort_p - 1)).reshape(-1, 1)
            grad_log_pi_idx = self.escort_p * np.matmul(
                self.reduced_state_feat,
                (I_action / theta[action] - tmp_vec / escort_denom).T)
            grad_entropy_idx = 0 # Not Implemented
        else:
            raise NotImplementedError()


        # try: # testing code
        #     state_feat = self.get_state_features(state)
        #     theta_tmp = np.matmul(self.policy_weight.T, state_feat).flatten()
        #     assert np.allclose(theta_tmp, theta)
        #     if self.FLAG_PG_TYPE == 'regular':
        #         grad_log_pi_tmp = np.matmul(state_feat,
        #                                     (I_action - pi_current_state).T)
        #     elif self.FLAG_PG_TYPE == 'alternate':
        #         grad_log_pi_tmp = np.matmul(state_feat, I_action.T)
        #     assert np.allclose(grad_log_pi_tmp[idx], grad_log_pi_idx)
        #     assert math.isclose(np.abs(grad_log_pi_tmp[idx]).sum(),
        #                         np.abs(grad_log_pi_tmp).sum())
        # except:
        #     pdb.set_trace()
        
        policy_grad_idx = discounting_term * (
            delta * grad_log_pi_idx + self.ent_tau * grad_entropy_idx)
                
        # self.policy_weight += self.policy_stepsize * policy_grad
        self.policy_weight[idx] += self.policy_stepsize * policy_grad_idx

        # possibly add some random noise
        if self.grad_bias is not None or self.grad_noise is not None:
            self.policy_weight += self.policy_stepsize \
                * np.random.normal(loc=self.grad_bias, scale=self.grad_noise,
                                   size=self.policy_weight.shape)
        self.FLAG_POLICY_WEIGHTS_UPDATED = True

def run_experiment(env_name, seed_number, num_total_timesteps, tilecoder,
                   policy_stepsize, critic_stepsize, gamma,
                   FLAG_BASELINE, FLAG_LEARN_VPI, FLAG_PG_TYPE,
                   FLAG_POLICY_MAPPING, FLAG_POPULAR_PG,
                   theta_init_0=0, theta_init_1=0, theta_init_2=0,
                   value_weight_init=0, episode_cutoff_length=1000,
                   escort_p=None, ent_tau=0,
                   grad_bias=None, grad_noise=None, FLAG_SWITCH_ACTIONS=False,
                   FLAG_CAPTURE_ENTROPY=False):

    # define the agent and the environment
    if env_name == 'MountainCar':
        env = MountainCar(seed=seed_number)
    elif env_name == 'GymMountainCar':
        env = GymMountainCar(seed=seed_number)
    elif env_name == 'Acrobot':
        env = Acrobot(seed=seed_number)
    elif env_name == 'Cartpole':
        env = Cartpole(seed=seed_number)
    else:
        raise NotImplementedError()
    
    np.random.seed(seed_number)
    random.seed(seed_number)

    num_actions = len(env.actions(None))

    agent = Agent(num_actions=num_actions,
                  policy_stepsize=policy_stepsize,
                  critic_stepsize=critic_stepsize, gamma=gamma,
                  FLAG_BASELINE=FLAG_BASELINE,
                  FLAG_PG_TYPE=FLAG_PG_TYPE,
                  FLAG_POLICY_MAPPING=FLAG_POLICY_MAPPING,
                  FLAG_POPULAR_PG=FLAG_POPULAR_PG,
                  theta_init_0=theta_init_0,
                  theta_init_1=theta_init_1,
                  theta_init_2=theta_init_2,
                  value_weight_init=value_weight_init,
                  escort_p=escort_p, ent_tau=ent_tau,
                  grad_bias=grad_bias, grad_noise=grad_noise,
                  tilecoder=tilecoder)

    return_across_episodes = []
    ep_len_across_episodes = []
    vpi_s0_across_episodes = []
    entropy_across_episodes = [] if FLAG_CAPTURE_ENTROPY else None
    
    episode_i = 0
    total_timesteps = 0

    FLAG_NAN_ENCOUNTERED_ABORT = False
    while total_timesteps < num_total_timesteps:
        episode_i += 1

        if FLAG_PG_TYPE not in ['regular', 'alternate']:
            raise NotImplementedError()
        if FLAG_POLICY_MAPPING not in ['softmax', 'escort']:
            raise NotImplementedError()
        # traj = {'state_list': [],
        #         'action_list': [],
        #         'action_prob_list': [],
        #         'reward_list': [],
        #         'next_state_list': []}

        # sample a trajectory by following the current policy
        discounting_term = 1
        done = False
        state = env.start()
        start_state = state.copy() # keep the start state for plotting vpi_pred
        time_per_episode = 0
        factor = 1
        discounted_return = 0
        entropy_for_one_episode = [] if FLAG_CAPTURE_ENTROPY else None
        while not done:
            action, action_prob = agent.take_action(state)

            if env_name == 'GymMountainCar':
                switching_timestep = 100000
            elif env_name == 'Acrobot':
                switching_timestep = 200000
            else:
                switching_timestep = None
            
            env_action_mapping = action
            if FLAG_SWITCH_ACTIONS:
                if total_timesteps >= switching_timestep:
                    if action == 0:
                        env_action_mapping = 2
                    elif action == 2:
                        env_action_mapping = 0
                    
            reward, next_state, done = env.step(action=env_action_mapping)

            # traj['state_list'].append(state)
            # traj['action_list'].append(action)
            # traj['action_prob_list'].append(action_prob)
            # traj['reward_list'].append(reward)
            # traj['next_state_list'].append(next_state)

            v_pi_pred_state = agent.pred_v_pi(state)
            if done:
                # terminal state has a value of zero by definition
                # (see the end of Section 2.1 of my MSc thesis)
                # Also note that, if we don't set this, TD style updates
                # will drive the value estimates to infinite
                # (even when the algorithm is able to visit the terminal
                # state) with gamma = 1. Interestingly, the value estimates
                # will still be driven (if we do set v_sx = 0) to infinite
                # with gamma = 1, if the algorithm is never able to visit
                # the terminal state; and it makes sense that timeouts will
                # not help with this.
                v_pi_pred_next_state = 0
            else:
                v_pi_pred_next_state = agent.pred_v_pi(next_state)

            # entropy regularized reward
            reward_ent = reward + ent_tau * agent.calc_policy_entropy()
                
            delta = reward_ent + gamma * v_pi_pred_next_state \
                - v_pi_pred_state

            # update the policy
            agent.update_policy(state, action, delta, discounting_term)

            # update the value function
            if FLAG_LEARN_VPI:
                agent.update_value_fn(state, delta)

            if not FLAG_POPULAR_PG:
                discounting_term *= gamma
                
            state = next_state
            time_per_episode += 1
            discounted_return += factor * reward
            factor *= gamma

            if FLAG_CAPTURE_ENTROPY:
                entropy_for_one_episode.append(agent.calc_policy_entropy())

            # if policy is infinite, kill the training and give bad return
            # assume that the agent will be unable to solve the task
            # in rest of the episodes, and therefore calculate the remaining
            # episodes it will go through based on episode_cutoff_length
            pol_weight_sum = agent.policy_weight.sum()
            val_weight_sum = agent.value_weight.sum()
            if (np.isinf(pol_weight_sum) or np.isnan(pol_weight_sum)
                or np.isinf(val_weight_sum) or np.isnan(val_weight_sum)):
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
                if FLAG_CAPTURE_ENTROPY:
                    entropy_across_episodes += \
                        [0] * remaining_ep_count
                FLAG_NAN_ENCOUNTERED_ABORT = True
                break

            if time_per_episode >= episode_cutoff_length: # timeout
                done = True
                
        if FLAG_NAN_ENCOUNTERED_ABORT:
            FLAG_NAN_ENCOUNTERED_ABORT = False
            break
        else:
            # save the returns, ep_len, vpi_s0 for this episode
            return_across_episodes.append(discounted_return)
            ep_len_across_episodes.append(time_per_episode)
            vpi_s0_across_episodes.append(
                agent.pred_v_pi(start_state).item())
            if FLAG_CAPTURE_ENTROPY:
                entropy_across_episodes.append(np.mean(entropy_for_one_episode))

            # print(episode_i, time_per_episode)

        total_timesteps += time_per_episode

    dat = {'returns': return_across_episodes,
           'ep_len': ep_len_across_episodes,
           'vpi_s0': vpi_s0_across_episodes,
           'entropy': entropy_across_episodes}

    return dat
