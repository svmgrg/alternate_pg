import numpy as np
import random
import pdb

import tools
    
class BatchGradientBanditAgent(object):
    def __init__(self, num_actions, alpha, beta, action_pref_init,
                 TYPE=None, eta_flag='zero', baseline_flag='zero',
                 batch_size=1, true_reward=None, baseline_init=0,
                 grad_bias=None, grad_noise=None):
        '''
        rpi_flag can be in {'fixed', 'learned', 'rpi'}
        eta_flag can be in {'zero', 'pi'}
        '''
        assert len(action_pref_init) == num_actions
        self.theta = np.array([action_pref_init] * batch_size).T
        self.theta = self.theta.astype(np.float32)

        self.baseline_init = baseline_init
        if baseline_init is not None:
            self.baseline = baseline_init * np.ones(batch_size)
        else:
            self.baseline = None
            
        self.t = 1
        
        self.pi = None
        self.FLAG_PERFORM_CALLED = False
        
        self.num_actions = num_actions
        self.alpha = alpha # actor stepsize
        self.beta = beta # critic stepsize
        self.TYPE = TYPE # only needed if using expected grad 
        self.eta_flag = eta_flag
        self.baseline_flag = baseline_flag
        self.batch_size = batch_size
        self.true_reward = true_reward
        self.action_pref_init = action_pref_init
        self.grad_bias = grad_bias
        self.grad_noise = grad_noise
        
        self.bandit_detail_dict = {'num_actions': num_actions,
                                   'alpha': alpha,
                                   'beta': beta,
                                   'TYPE': TYPE,
                                   'eta_flag': eta_flag,
                                   'baseline_flag': baseline_flag,
                                   'batch_size': batch_size}
                
    def reset(self):
        self.t = 1

        if self.baseline_init is not None:
            self.baseline = self.baseline_init * np.ones(self.batch_size)
        else:
            self.baseline = None
        
        assert len(self.action_pref_init) == self.num_actions
        self.theta = np.array(
            [self.action_pref_init] * self.batch_size).T
        self.theta = self.theta.astype(np.float32)
                
    def act(self):
        self.FLAG_PERFORM_CALLED = True
        self.pi = tools.softmax(self.theta)
        action = tools.vectorized_categorical_sampling(self.pi)
        return action

    def calc_rpi(self):
        return (self.true_reward * self.pi).sum(0)

    def calc_true_grad(self):
        return self.pi * (self.true_reward - self.calc_rpi())

    def train(self, action, reward):
        if self.FLAG_PERFORM_CALLED == False:
            raise PerformNotCalledError()
        self.t += 1
        
        #----------------------------------------------------------
        # Set the baseline
        #----------------------------------------------------------
        if self.baseline_flag == 'fixed':
            self.baseline = self.baseline_init * np.ones(self.batch_size)
        elif self.baseline_flag == 'rpi':
            self.baseline = self.calc_rpi()
        elif self.baseline_flag == 'learned':
            pass # update the baseline after updating theta
                        
        #----------------------------------------------------------
        # Set eta
        #----------------------------------------------------------
        if self.eta_flag == 'zero': # alternate estimator
            eta = np.zeros((self.num_actions, self.batch_size))
        elif self.eta_flag == 'pi': # regular estimator
            eta = self.pi

        #----------------------------------------------------------
        # Update theta
        #----------------------------------------------------------
        if self.TYPE == 'expected_pg':
            grad = self.calc_true_grad()
            self.theta += self.alpha * grad
        elif self.TYPE == 'sample_based':
            theta_old = self.theta[action, range(self.batch_size)].copy()
            self.theta = self.theta \
                - self.alpha * (reward - self.baseline) * eta
            self.theta[action, range(self.batch_size)] \
                = theta_old + self.alpha * (reward - self.baseline) \
                * (1 - eta[action, range(self.batch_size)])

        if self.grad_bias is not None or self.grad_noise is not None:
            self.theta = self.theta + self.alpha * np.random.normal(
                loc=self.grad_bias, scale=self.grad_noise,
                size=self.theta.shape)

        #----------------------------------------------------------
        # Update the baseline
        #----------------------------------------------------------
        if self.baseline_flag == 'learned':
            self.baseline = self.beta * reward + (1 - self.beta) * self.baseline

        self.FLAG_PERFORM_CALLED = False
        

class BatchBandit(object):
    def __init__(self, num_actions, mean, std, reward_noise, random_walk_std,
                 batch_size=1, true_reward_list=None):
        self.num_actions = num_actions
        self.mean = mean
        self.std = std
        self.reward_noise = reward_noise
        self.random_walk_std = random_walk_std
        self.batch_size = batch_size
        self.true_reward_list = true_reward_list

        if self.true_reward_list is None:
            self.true_reward = np.random.normal(
                loc=self.mean, scale=self.std,
                size=(self.num_actions, self.batch_size))
        else:
            assert len(true_reward_list) == num_actions
            self.true_reward = np.array(
                [self.true_reward_list] * self.batch_size).T

    def reset(self):
        if self.true_reward_list is None:
            self.true_reward = np.random.normal(
                loc=self.mean, scale=self.std,
                size=(self.num_actions, self.batch_size))
        else:
            assert len(self.true_reward_list) == self.num_actions
            self.true_reward = np.array(
                [self.true_reward_list] * self.batch_size).T

    def random_walk(self):
        self.true_reward += np.random.normal(
            loc=0, scale=self.random_walk_std,
            size=(self.num_actions, self.batch_size))

    def step(self, action):
        return np.random.normal(
            loc=self.true_reward[action, range(self.batch_size)],
            scale=self.reward_noise)

    def optimal_arm(self):
        return np.argmax(self.true_reward, axis=0)

