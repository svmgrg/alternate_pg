import numpy as np
import json
import pickle
import pdb
import matplotlib.pyplot as plt

def moving_average(a, n=100):
    # USAGE: plt.plot(moving_average(list_reward_const))
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class NumpyEncoder(json.JSONEncoder):
    '''https://stackoverflow.com/questions/26646362/
    numpy-array-is-not-json-serializable
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
    
def softmax(x):
    e_x = np.exp(x - np.max(x, 0))
    out = e_x / e_x.sum(0)
    return out

def run_exp(bandit_env, agent, num_iter, num_capture=10):
    mean_optimal_hit_ratio = []
    mean_reward = []
    list_rpi = []
    list_reward = []
    list_theta = []
    list_baseline = []

    bandit_env.reset()
    agent.reset()

    action = agent.act()
    reward = bandit_env.step(action)
    agent.train(action, reward)
    
    for i in range(num_iter):
        hit_optimal_ratio = np.array(action == bandit_env.optimal_arm(),
                                     dtype=np.int32)
        mean_optimal_hit_ratio.append(hit_optimal_ratio.mean())
        
        action = agent.act()
        reward = bandit_env.step(action)
        agent.train(action, reward)
        
        mean_reward.append(reward.mean())
        list_theta.append(agent.theta[:, :num_capture].copy())
        list_baseline.append(agent.baseline[:num_capture].copy())
        list_rpi.append(agent.calc_rpi()[:num_capture].copy())
        list_reward.append(reward[:num_capture].copy())

    dat = {'bandit_details': agent.bandit_detail_dict,
           'mean_optimal_hit_ratio': np.array(mean_optimal_hit_ratio),
           'mean_reward': np.array(mean_reward),
           'list_reward': np.stack(list_reward),
           'list_rpi': np.stack(list_rpi),
           'list_theta': np.stack(list_theta),
           'list_baseline': np.stack(list_baseline)}

    return dat

def vectorized_categorical_sampling(p):
    # https://stackoverflow.com/questions/47722005/vectorizing-numpy-ra
    #         ndom-choice-for-given-2d-array-of-probabilities-along-an-a
    # https://stackoverflow.com/questions/34187130/fast-random-weighted-
    #         selection-across-all-rows-of-a-stochastic-matrix
    # SUPPORT NOT AVAILABLE ? : https://github.com/numpy/numpy/pull/7810
    #                           https://github.com/numpy/numpy/issues/2724
    # Vector replacement for:
    # action = np.random.choice(array, prob)
    cumsum = np.cumsum(p, axis=0)
    cumsum[-1, :] = 1 # set the last value as 1 (to avoid numerical errors)
    random_numbers = np.random.rand(p.shape[1])
    sampled_numbers = np.sum(cumsum < random_numbers, axis=0)
    return sampled_numbers

#----------------------------------------------------------------------
# Misc Plotting Functions
#----------------------------------------------------------------------
def color_gradient(mix, c1='blue', c2='green', c3='red'):
    # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1) to
    # and then from c2 (mix=1) to c3 (mix=2))
    # Modified from: "https://stackoverflow.com/questions/25668828/
    # /how-to-create-colour-gradient-in-python"
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    c3 = np.array(mpl.colors.to_rgb(c3))
    return mpl.colors.to_hex(((1 - mix) * c1 + mix * c2) * (mix <= 1) \
                             + ((2 - mix) * c2 + (mix - 1) * c3) * (mix > 1))

def plot_traj(ax, list_prob):
    ax.axis('equal')
    ax.axis('off')
    border_mat = np.array(
        [[1 / np.sqrt(2), -1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
         [-1 / np.sqrt(6), -1 / np.sqrt(6),
          np.sqrt(2) / np.sqrt(3), -1 / np.sqrt(6)]])
    ax.plot(border_mat[0], border_mat[1], color='k', linewidth=1)
    ax.text(border_mat[0, 0] - 0.1, border_mat[1, 0] - 0.1,
            r'$r(a_0): {}$'.format(q_values[0]), color='c')
    ax.text(border_mat[0, 1] - 0.1, border_mat[1, 1] - 0.1,
            r'$r(a_1): {}$'.format(q_values[1]), color='k')
    ax.text(border_mat[0, 2] - 0.1, border_mat[1, 2] + 0.025,
            r'$r(a_2): {}$'.format(q_values[2]), color='m')
    proj_2d_mat = np.array(
        [[1 / np.sqrt(2), -1 / np.sqrt(2), 0],
         [-1 / np.sqrt(6), -1 / np.sqrt(6), np.sqrt(2) / np.sqrt(3)],
         [0, 0, 0]])
    points = np.matmul(proj_2d_mat, list_prob)
    
    ax.plot(points[0:25, 0, :50], points[0:25, 1, :50],
            color='blue', linewidth=1, alpha=0.25)
    ax.plot(points[25:100, 0, :50], points[25:100, 1, :50],
            color='green', linewidth=1, alpha=0.25)
    ax.plot(points[100:1000, 0, :50], points[100:1000, 1, :50],
            color='red', linewidth=1, alpha=0.25)
    
    # for j in range(1000):
    #     ax.plot(points[j:j+2, 0, :5], points[j:j+2, 1, :5],
    #              color=color_gradient(mix=j/500), linewidth=1, alpha=0.25)

def plot_alpha_sensitivity(ax, folder_name, alpha_list, pg_type, eta_flag,
                           baseline_flag, beta, plt_color, linewidth,
                           start_plotting_iter, end_plotting_iter):
    perf_mean_list = []
    perf_stderr_list = []
    for alpha in alpha_list:
        if pg_type == 'expected_pg':
            filename = '{}/expectedPG__alpha_{}'.format(folder_name, alpha)
        elif pg_type == 'stochastic_pg':
            filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'.format(
                folder_name, eta_flag, baseline_flag, alpha, beta)
        
        with open(filename, 'rb') as handle:
            dat = pickle.load(handle)
        final_perf = dat['list_rpi'][
            start_plotting_iter:end_plotting_iter, :].mean(0)
        
        perf_mean_list.append(final_perf.mean())
        perf_stderr_list.append(final_perf.std() / np.sqrt(final_perf.shape[0]))

    ax.errorbar(np.log(alpha_list) / np.log(2),
                perf_mean_list, yerr=perf_stderr_list,
                color=plt_color,
                label='{}_{}_{}'.format(eta_flag, baseline_flag, beta),
                linewidth=linewidth)
    # ax.scatter(0, 3, s=0) # add an invisible point to make scaling from 0 -> 1
    ax.scatter(0, 1, s=0) # add an invisible point to make scaling from 0 -> 1

def plot_learning_curves(ax1, ax2, folder_name, pg_type, eta_flag,
                         baseline_flag, alpha, beta):
    if pg_type == 'expected_pg':
        filename = '{}/expectedPG__alpha_{}'.format(folder_name, alpha)
    elif pg_type == 'stochastic_pg':
        filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'.format(
            folder_name, eta_flag, baseline_flag, alpha, beta)
    with open(filename, 'rb') as handle:
        dat = pickle.load(handle)

    # plot individual runs
    if baseline_flag == 'learned':
        ax1.plot(dat['list_baseline'], color='tab:cyan', linewidth=0.1,
                 alpha=0.1)
    ax1.plot(dat['list_rpi'], color='tab:pink', linewidth=0.1, alpha=0.1)

    # plot the average performance
    if baseline_flag == 'learned':
        ax1.plot(dat['list_baseline'].mean(1), color='blue', linewidth=0.5)
    ax1.plot(dat['list_rpi'].mean(1), color='red', linewidth=0.5)

    # plot action preferences (individual runs)
    ax2.plot(dat['list_theta'][:, 0, :], color='tab:red', linewidth=0.1,
             alpha=0.1)
    ax2.plot(dat['list_theta'][:, 1, :], color='tab:green', linewidth=0.1,
             alpha=0.1)
    ax2.plot(dat['list_theta'][:, 2, :], color='tab:blue', linewidth=0.1,
             alpha=0.1)

    # plot action preferences (average performance)
    ax2.plot(dat['list_theta'][:, 0, :].mean(1), color='red', linewidth=0.5)
    ax2.plot(dat['list_theta'][:, 1, :].mean(1), color='green', linewidth=0.5)
    ax2.plot(dat['list_theta'][:, 2, :].mean(1), color='blue', linewidth=0.5)

def plot_learning_curves_simple(ax1, folder_name, pg_type, eta_flag,
                                baseline_flag, alpha, beta):
    if pg_type == 'expected_pg':
        filename = '{}/expectedPG__alpha_{}'.format(folder_name, alpha)
    elif pg_type == 'stochastic_pg':
        filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'.format(
            folder_name, eta_flag, baseline_flag, alpha, beta)
    with open(filename, 'rb') as handle:
        dat = pickle.load(handle)

    # plot individual runs
    if pg_type == 'expected_pg':
        plt_color = 'black'
    elif pg_type == 'stochastic_pg':
        if eta_flag == 'pi':
            plt_color = 'red' if baseline_flag == 'learned' else 'tab:red'
        elif eta_flag == 'zero':
            plt_color = 'blue' if baseline_flag == 'learned' else 'tab:blue'
    
    dat_rpi_mean = dat['list_rpi'].mean(1)
    dat_rpi_err = dat['list_rpi'].std(1) / np.sqrt(dat['list_rpi'].shape[1])

    ax1.plot(np.arange(dat_rpi_mean.shape[0]), dat_rpi_mean,
             linewidth=1.5, color=plt_color,
             label='{}_{}_{}'.format(pg_type, alpha, beta))
    ax1.fill_between(np.arange(dat_rpi_mean.shape[0]),
                     dat_rpi_mean - dat_rpi_err, dat_rpi_mean + dat_rpi_err,
                     alpha=0.2, facecolor=plt_color)

def plot_alpha_sensitivity_2(ax, folder_name, alpha_list, pg_type, eta_flag,
                              baseline_flag, beta, plt_color, linewidth,
                              start_plotting_iter, end_plotting_iter,
                              action_pref_init):
    perf_mean_list = []
    perf_stderr_list = []
    for alpha in alpha_list:
        if pg_type == 'expected_pg':
            filename = '{}/expectedPG__alpha_{}'.format(folder_name, alpha)
        elif pg_type == 'stochastic_pg':
            filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'.format(
                folder_name, eta_flag, baseline_flag, alpha, beta)
        
        with open(filename, 'rb') as handle:
            dat = pickle.load(handle)
        final_perf = dat['list_rpi'][
            start_plotting_iter:end_plotting_iter, :].mean(0)
        
        perf_mean_list.append(final_perf.mean())
        perf_stderr_list.append(final_perf.std() / np.sqrt(final_perf.shape[0]))

    ax.errorbar(np.log(alpha_list) / np.log(2),
                perf_mean_list, yerr=perf_stderr_list,
                color=plt_color,
                label='{}'.format(str(action_pref_init)),
                linewidth=linewidth)
    # ax.scatter(0, 1, s=0) # add an invisible point to make scaling from 0 -> 1


def plot_alpha_sensitivity_3(ax, folder_name, alpha_list, pg_type, eta_flag,
                             baseline_flag, beta, plt_color, linewidth,
                             start_plotting_iter, end_plotting_iter,
                             grad_bias, grad_noise):
    perf_mean_list = []
    perf_stderr_list = []
    for alpha in alpha_list:
        if pg_type == 'expected_pg':
            filename = '{}/expectedPG__alpha_{}__gradBias_{}'\
                '__gradNoise_{}'.format(folder_name, alpha,
                                        grad_bias, grad_noise)
        elif pg_type == 'stochastic_pg':
            filename = '{}/eta_{}__baseline_{}__alpha_{}__beta_{}'\
                '__gradBias_{}__gradNoise_{}'.format(folder_name, eta_flag,
                                                     baseline_flag, alpha, beta,
                                                     grad_bias, grad_noise)
        with open(filename, 'rb') as handle:
            dat = pickle.load(handle)
        final_perf = dat['list_rpi'][
            start_plotting_iter:end_plotting_iter, :].mean(0)
        
        perf_mean_list.append(final_perf.mean())
        perf_stderr_list.append(final_perf.std() / np.sqrt(final_perf.shape[0]))

    ax.errorbar(np.log(alpha_list) / np.log(2),
                perf_mean_list, yerr=perf_stderr_list,
                color=plt_color,
                label='{}_{}_{}_{}'.format(
                    eta_flag, baseline_flag, beta, grad_noise),
                linewidth=linewidth)
    # ax.scatter(0, 1, s=0) # add an invisible point to make scaling from 0 -> 1
