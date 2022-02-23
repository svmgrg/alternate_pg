import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pdb

#----------------------------------------------------------------------
# some useful variables
#----------------------------------------------------------------------
border_mat = np.array(
    [[1 / np.sqrt(2), -1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
     [-1 / np.sqrt(6), -1 / np.sqrt(6),
      np.sqrt(2) / np.sqrt(3), -1 / np.sqrt(6)]])

proj_2d_mat = np.array(
    [[1 / np.sqrt(2), -1 / np.sqrt(2), 0],
     [-1 / np.sqrt(6), -1 / np.sqrt(6), np.sqrt(2) / np.sqrt(3)],
     [0, 0, 0]])

POS = +1
NEG = -1

color_red = np.array(mpl.colors.to_rgb('red'))
color_blue = np.array(mpl.colors.to_rgb('blue'))
color_steelblue = np.array(mpl.colors.to_rgb('steelblue'))
color_green = np.array(mpl.colors.to_rgb('green'))
color_yellow = np.array(mpl.colors.to_rgb('yellow'))
color_silver = np.array(mpl.colors.to_rgb('silver'))
color_black = np.array(mpl.colors.to_rgb('black'))

#----------------------------------------------------------------------
# utility functions
#----------------------------------------------------------------------
def generate_policy_points(resolution, FLAG_INCLUDE_BOUNDARY=True):
    if FLAG_INCLUDE_BOUNDARY:
        boundary_factor = 1
    else:
        boundary_factor = 0
    num_points = int((resolution + boundary_factor) \
                     * (resolution + boundary_factor + 1) / 2)
    points = np.zeros((3, num_points))
    idx = 0
    for i in range(0, resolution + boundary_factor):
        for j in range(0, resolution + boundary_factor - i, 1):
            k = resolution - i - j - 1 + boundary_factor
            points[0][idx] = i/resolution
            points[1][idx] = j/resolution
            points[2][idx] = k/resolution
            idx = idx + 1
    return points, num_points

def calc_grad(pi, pg_type, reward, action, eta, baseline,
              reward_noise, noise_sign):
    I_A = np.zeros((3, 1))
    I_A[action] = 1
    rpi = np.dot(pi.T, reward).flatten()
    eps = 0 if reward_noise is None else noise_sign * reward_noise
    # generate gradients
    if pg_type == 'expected': # = E[(I_A - eta) (R - b)]
        grad = pi * (reward - baseline) - eta * (rpi - baseline)
    elif pg_type == 'sample_based':
        grad = (I_A - eta) * (reward[action] + eps - baseline)

    return grad

def update_pi(pi_old, grad, stepsize):
    theta_old = np.log(pi_old)
    theta_new = theta_old + stepsize * grad
    pi_new = np.exp(theta_new) / np.exp(theta_new).sum(0)

    return pi_new

def calc_variance(pi, reward, V_R_A, eta, baseline):
    rpi = np.matmul(pi.T, reward).flatten()
    E_R = rpi
    E_R2 = (np.matmul(pi.T, reward**2) + np.matmul(pi.T, V_R_A)).flatten()
    E_IA = pi
    E_IA_R = pi * reward
    E_IA_R2 = pi * reward**2 + pi * V_R_A

    V_IA_R = E_IA_R2 - E_IA_R**2
    V_IA = E_IA - E_IA**2
    V_R = E_R2 - E_R**2
    Cov_IA_R__IA = E_IA_R - E_IA_R * E_IA
    Cov_IA_R__R = E_IA_R2 - E_IA_R * E_R
    Cov_IA__R = E_IA_R - E_IA * E_R
    
    variance = V_IA_R + V_IA * baseline**2 + V_R * eta**2 \
        - 2 * Cov_IA_R__IA * baseline \
        - 2 * Cov_IA_R__R * eta \
        + 2 * Cov_IA__R * eta * baseline

    return variance

def assgn_color(val, pt0, pt1, pt2, pt3):
    if val <= pt1:
        col = (val - pt0) / (pt1 - pt0)
        final_color = (1 - col) * color_black + col * color_red
    elif val <= pt2:
        col = (val - pt1) / (pt2 - pt1)
        final_color = (1 - col) * color_red + col * color_yellow
    else:
        col = (val - pt2) / (pt3 - pt2)
        final_color = (1 - col) * color_yellow + col * color_steelblue
    return final_color

#----------------------------------------------------------------------
# plotting functions
#----------------------------------------------------------------------
def plot_policy_space(reward, res=30):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # generate points
    points, num_points = generate_policy_points(res, FLAG_INCLUDE_BOUNDARY=False)
    points_2d = np.matmul(proj_2d_mat, points)

    # assign colors to the points
    final_colors = (points[0].reshape(1, -1) * color_red.reshape(-1, 1)
                    + points[1].reshape(1, -1) * color_green.reshape(-1, 1)
                    + points[2].reshape(1, -1) * color_blue.reshape(-1, 1)).T

    # plot
    for i in range(num_points):
        ax.scatter(points_2d[0, i], points_2d[1, i],
                   color=final_colors[i], marker='h', s=24)
    ax.scatter(0, 0, color='black', marker='o', s=24)
    # draw a triangle and add labels
    ax.plot(border_mat[0], border_mat[1], color='k', linewidth=1) 
    ax.text(border_mat[0, 0] - 0.35, border_mat[1, 0] - 0.4,
            r'$(1, 0, 0)$' + '\n' + r'$r(a_0) = {}$'.format(reward[0][0]),
            color='red') # right label
    ax.text(border_mat[0, 1] - 0.3, border_mat[1, 1] - 0.4,
            r'$(0, 1, 0)$' + '\n' + r'$r(a_1) = {}$'.format(reward[1][0]),
            color='green') # left label
    ax.text(border_mat[0, 2] - 0.35, border_mat[1, 2] + 0.05,
            r'$r(a_2) = {}$'.format(reward[2][0]) + '\n' + r'$(0, 0, 1)$',
            color='blue') # top label

    ax.axis('equal')
    ax.axis('off')

    plt.savefig('policy_space.pdf')
    plt.close()
    
plot_policy_space(reward=np.array([[1, 2, 3]]).T)



def plot_gradient_field_single_ax(ax, pg_type, reward, action, eta, baseline,
                                  reward_noise=None, stepsize=1, res=10,
                                  FLAG_QUIVER=True):
    if pg_type == 'expected':
        assert action is None

    # generate points
    points, num_points = generate_policy_points(res, FLAG_INCLUDE_BOUNDARY=True)
    points_2d = np.matmul(proj_2d_mat, points)
  
    # calculate baseline
    if baseline == 'rpi':
        baseline = np.dot(points.T, reward).flatten()
    if eta == 'pi':
        eta = points

    # generate gradients
    grad_pos = calc_grad(pi=points, pg_type=pg_type, reward=reward,
                         action=action, eta=eta, baseline=baseline,
                         reward_noise=reward_noise, noise_sign=POS)
    pi_pos = update_pi(pi_old=points, grad=grad_pos, stepsize=stepsize)
    pi_pos_2d = np.matmul(proj_2d_mat, pi_pos)
    pos_diff_2d = pi_pos_2d - points_2d
    color_pos = 'brown' if pg_type == 'expected' else 'steelblue'

    if reward_noise != 0: # calculate the above difference for NEG
        grad_neg = calc_grad(pi=points, pg_type=pg_type, reward=reward,
                             action=action, eta=eta, baseline=baseline,
                             reward_noise=reward_noise, noise_sign=NEG)
        pi_neg = update_pi(pi_old=points, grad=grad_neg, stepsize=stepsize)
        pi_neg_2d = np.matmul(proj_2d_mat, pi_neg)
        neg_diff_2d = pi_neg_2d - points_2d

        color_pos = 'steelblue'
        color_neg = 'silver'

    # plot
    ax.plot(border_mat[0], border_mat[1], color='k', linewidth=0.25)
    if action is not None:
        color_dict = {0: 'tab:red', 1: 'tab:green', 2:'tab:blue'}
        corner_point = np.zeros((3, 1))
        corner_point[action] = 1
        corner_point_2d = np.matmul(proj_2d_mat, corner_point)
        ax.scatter(corner_point_2d[0], corner_point_2d[1], s=100, alpha=0.3,
                   color=color_dict[action])
    if FLAG_QUIVER:
        ax.scatter(points_2d[0], points_2d[1], s=1, color='purple')
        ax.quiver(points_2d[0], points_2d[1], pos_diff_2d[0], pos_diff_2d[1],
                  color=color_pos, width=0.007, headwidth=3, headlength=4,
                  scale=1, scale_units='xy', linewidth=0.1, edgecolor='black')
        if reward_noise != 0: # plot NEG as well
            ax.quiver(points_2d[0], points_2d[1], neg_diff_2d[0], neg_diff_2d[1],
                      color=color_neg, width=0.007, headwidth=3, headlength=4,
                      scale=1, scale_units='xy', linewidth=0.1,
                      edgecolor='black')
    else: # for debugging quiver; without quiver function; uglier plots
        ax.scatter(points_2d[0], points_2d[1], s=5, color='purple', marker='x')
        ax.plot([points_2d[0], pi_pos_2d[0]],
                [points_2d[1], pi_pos_2d[1]],
                color=color_pos, linewidth=0.5)
        for i in range(num_points):
            if reward_noise is not None: # plot NEG as well
                ax.plot([points_2d[0], pi_neg_2d[0]],
                        [points_2d[1], pi_neg_2d[1]],
                        color=color_neg, linewidth=0.5)
        ax.scatter(pi_pos_2d[0], pi_pos_2d[1], s=1, color=color_pos)
        if reward_noise is not None: # plot NEG as well
            ax.scatter(pi_neg_2d[0], pi_neg_2d[1], s=1, color=color_neg)

def plot_gradient_field(reward, reward_noise,
                            method_name_list, eta_list, baseline_list,
                            plot_expected_curves_cols, save_bname,
                            stepsize=1, res=10, FLAG_QUIVER=True):
    fig, axs = plt.subplots(4, 4)
    plt.subplots_adjust(wspace=0, hspace=0)
    for row in range(4):
        for col in range(4):
            axs[row, col].axis('equal')
            axs[row, col].axis('off')
    fig.suptitle('PG Update - Reward: [{}, {}, {}] | Reward Noise: {}'.format(
        reward[0, 0], reward[1, 0], reward[2, 0], reward_noise,
        save_bname))
    axs[0, 0].text(-1.5, 0, r'$A = a_0$', color='tab:red', size=14)
    axs[1, 0].text(-1.5, 0, r'$A = a_1$', color='tab:green', size=14)
    axs[2, 0].text(-1.5, 0, r'$A = a_2$', color='tab:blue', size=14)
    axs[3, 0].text(-1.5, 0, r'$\mathbb{E}(\nabla \mathcal{J}_\pi)$',
                   color='tab:gray', size=14)
    for col in range(4):
        axs[0, col].set_title(method_name_list[col])

    for col, eta, baseline in zip(range(4), eta_list, baseline_list):
        for row in range(3):
            # sample based methods
            plot_gradient_field_single_ax(ax=axs[row, col],
                                          pg_type='sample_based',
                                          reward=reward, action=row,
                                          eta=eta, baseline=baseline,
                                          reward_noise=reward_noise,
                                          stepsize=stepsize, res=res,
                                          FLAG_QUIVER=FLAG_QUIVER)
        # expected methods
        if col in plot_expected_curves_cols:
            plot_gradient_field_single_ax(ax=axs[3, col], pg_type='expected',
                                          reward=reward, action=None,
                                          eta=eta, baseline=baseline,
                                          reward_noise=0,
                                          stepsize=stepsize+0.3, res=res,
                                          FLAG_QUIVER=FLAG_QUIVER)

    plt.savefig('update__reward{}_{}_{}__rewardnoise{}__baseline{}'
                '__stepsize{}.pdf'.format(
                    reward[0, 0], reward[1, 0], reward[2, 0], reward_noise, 
                    save_bname, stepsize))
    plt.close()
    
def plot_variance(reward, V_R_A, method_name_list, eta_list, baseline_list,
                  save_bname, res=20):
    fig, axs = plt.subplots(4, 4)
    plt.subplots_adjust(wspace=0, hspace=0)
    for row in range(4):
        for col in range(4):
            axs[row, col].axis('equal')
            axs[row, col].axis('off')
    fig.suptitle('Variance - Reward: [{}, {}, {}] '
                 '| V[R|A]: {}_{}_{} | Baseline:{}'.format(
                     reward[0, 0], reward[1, 0], reward[2, 0],
                     V_R_A[0, 0], V_R_A[1, 0], V_R_A[2, 0], save_bname))
    axs[0, 0].text(-1.5, 0, r'$\mathbb{V} \; [g_0]$', color='tab:red', size=14)
    axs[1, 0].text(-1.5, 0, r'$\mathbb{V} \; [g_1]$', color='tab:green', size=14)
    axs[2, 0].text(-1.5, 0, r'$\mathbb{V} \; [g_2]$', color='tab:blue', size=14)
    axs[3, 0].text(-1.5, 0, 'Total', color='tab:gray', size=14)
    for col in range(4):
        axs[0, col].set_title(method_name_list[col])
    
    # generate points
    points, num_points = generate_policy_points(res, FLAG_INCLUDE_BOUNDARY=True)
    points_2d = np.matmul(proj_2d_mat, points)

    # calculate variance
    variance_list = np.zeros((4, 3, num_points))
    for method_idx, eta_tmp, baseline in zip(range(4), eta_list, baseline_list):
        if eta_tmp == 'pi':
            eta = points
        elif eta_tmp == 0:
            eta = 0
        b = np.dot(points.T, reward).flatten() if baseline == 'rpi' else baseline
        variance_list[method_idx, :, :] = calc_variance(
            pi=points, reward=reward, V_R_A=V_R_A, eta=eta, baseline=b)
        
    pt0 = variance_list.min()
    pt1 = min(variance_list[1].max(), variance_list[2].max())
    pt2 = max(variance_list[1].max(), variance_list[2].max())
    pt3 = variance_list.sum(1).max()

    height = np.sqrt(2) / np.sqrt(3) + 1 / np.sqrt(6)

    for method in range(4):
        for action in range(3):
                for i in range(num_points):
                    val = variance_list[method, action, i]
                    color_pi = assgn_color(val, pt0, pt1, pt2, pt3)
                    axs[action, method].scatter(points_2d[0, i], points_2d[1, i],
                                                color=color_pi, marker='h', s=15)
                    axs[0, -1].scatter(0.9, val / pt3 * height - 1 / np.sqrt(6),
                                       color=color_pi, marker='_', s=5) # legend
        for i in range(num_points):
            total_val = variance_list[method, :, i].sum()
            color_pi = assgn_color(total_val, pt0, pt1, pt2, pt3)
            axs[3, method].scatter(points_2d[0, i], points_2d[1, i],
                                   color=color_pi, marker='h', s=15)
            axs[0, -1].scatter(0.9, total_val / pt3 * height - 1 / np.sqrt(6),
                               color=color_pi, marker='_', s=5) # legend

        # legend
        axs[0, -1].text(0.96, pt0 / pt3 * height - 1 / np.sqrt(6) - 0.04,
                        "{:.2f}".format(pt0), color='black', size=6)
        axs[0, -1].text(0.96, pt1 / pt3 * height - 1 / np.sqrt(6) - 0.04,
                        "{:.2f}".format(pt1), color='black', size=6)
        axs[0, -1].text(0.96, pt2 / pt3 * height - 1 / np.sqrt(6) - 0.04,
                        "{:.2f}".format(pt2), color='black', size=6)
        axs[0, -1].text(0.96, pt3 / pt3 * height - 1 / np.sqrt(6) - 0.04,
                        "{:.2f}".format(pt3), color='black', size=6)

    plt.savefig('variance__reward{}_{}_{}__VRA{}_{}_{}__baseline{}.pdf'.format(
        reward[0, 0], reward[1, 0], reward[2, 0],
        V_R_A[0, 0], V_R_A[1, 0], V_R_A[2, 0], save_bname))
    plt.close()

def plot_entropy_vs_variance(reward, V_R_A, method_name_list, eta_list,
                             baseline_list, save_bname, res=20):
    fig, axs = plt.subplots(2, 4, sharey=True)
    fig.suptitle('Entropy - Reward: [{}, {}, {}] '
                 '| V[R|A]: {}_{}_{} | Baseline:{}'.format(
                     reward[0, 0], reward[1, 0], reward[2, 0],
                     V_R_A[0, 0], V_R_A[1, 0], V_R_A[2, 0], save_bname))
    for i in range(2):
        for j in range(4):
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['top'].set_visible(False)
    axs[0, 0].set_ylabel('Variance')
    axs[0, 1].set_xlabel('Entropy')
    axs[1, 0].set_ylabel('Variance')
    axs[1, 1].set_xlabel('Distance from corner')

    for col in range(4):
        axs[0, col].set_title(method_name_list[col])

    # generate points
    points, num_points = generate_policy_points(res, FLAG_INCLUDE_BOUNDARY=True)
    points_2d = np.matmul(proj_2d_mat, points)

    # calculate variance
    variance_list = np.zeros((4, 3, num_points))
    for method_idx, eta_tmp, baseline in zip(range(4), eta_list, baseline_list):
        if eta_tmp == 'pi':
            eta = points
        elif eta_tmp == 0:
            eta = 0
        b = np.dot(points.T, reward).flatten() if baseline == 'rpi' else baseline
        variance_list[method_idx, :, :] = calc_variance(
            pi=points, reward=reward, V_R_A=V_R_A, eta=eta, baseline=b)
    total_variance = variance_list.sum(1)
    
    # calculate entropy
    log_pi = np.log(points)
    log_pi[log_pi == -1 * np.inf] = 0
    entropy_list = -1 * (points * log_pi).sum(0)

    # plot
    color_dict = {0: 'tab:red', 1: 'tab:green', 2: 'tab:blue'}
    for method_idx in range(4):
        for action in range(3): # individual components of the variance
            axs[0, method_idx].scatter(entropy_list,
                                       variance_list[method_idx, action],
                                       color=color_dict[action], s=1)
        for i in range(num_points): # total variance
            axs[0, method_idx].scatter(entropy_list[i],
                                       total_variance[method_idx, i],
                                       color='tab:gray', marker='x', s=1)
    # plot component wise variance as a distance from the corner
    for method_idx in range(4):
        for action in range(3):
            dist_corner = -1 * np.log(points[action])
            axs[1, method_idx].scatter(dist_corner,
                                       variance_list[method_idx, action],
                                       color=color_dict[action], s=1)

    # for generating legend
    ylim = axs[1, 0].get_ylim()[1]
    axs[1, 0].text(0, ylim, r'$\mathbb{V}\;[g_0]$', color='tab:red')
    axs[1, 0].text(0.9, ylim, r'$\mathbb{V}\;[g_1]$', color='tab:blue')
    axs[1, 0].text(1.8, ylim, r'$\mathbb{V}\;[g_2]$', color='tab:green')
    axs[1, 0].text(2.7, ylim, 'Total', color='tab:gray')

    plt.savefig('entropy__reward{}_{}_{}__VRA{}_{}_{}__baseline{}.pdf'.format(
        reward[0, 0], reward[1, 0], reward[2, 0],
        V_R_A[0, 0], V_R_A[1, 0], V_R_A[2, 0], save_bname))
    plt.close()

def plot_descent_factors(reward, res=20):
    # generate points
    points, num_points = generate_policy_points(res, FLAG_INCLUDE_BOUNDARY=True)
    points_2d = np.matmul(proj_2d_mat, points)

    rpi = np.dot(points.T, reward).flatten()
    denom =  (points**2 * (reward - rpi)).sum(0)
    numer =  (points**2 * reward * (reward - rpi)).sum(0)
    factor = numer / denom

    height = np.sqrt(2) / np.sqrt(3) + 1 / np.sqrt(6)

    neg_list = []
    pos_list = []
    for i in range(num_points):
        if denom[i] > 0:
            pos_list.append(factor[i])
        elif denom[i] < 0:
            neg_list.append(factor[i])
        else:
            print(denom[i])
    print(max(pos_list), min(neg_list))

    plt.subplot(311)
    plt.plot(denom)
    plt.subplot(312)
    plt.plot(numer)
    plt.subplot(313)
    plt.plot(factor)
    plt.show()
    
    pdb.set_trace()
        
    fig, axs = plt.subplots(1, 3)
    for col, val in zip(range(3), [numer, denom, factor]):
        pt0 = val.min()
        pt1 = 0
        pt2 = val.max()

        for i in range(num_points):
            if val[i] < pt1:
                tmp = (val[i] - pt0) / (pt1 - pt0)
                color_pos = (1 - tmp) * color_red + tmp * color_yellow
            else:
                tmp = (val[i] - pt1) / (pt2 - pt1)
                color_pos = (1 - tmp) * color_yellow + tmp * color_blue

            axs[col].scatter(points_2d[0, i], points_2d[1, i], marker='h', s=19,
                        color=color_pos)
            axs[col].scatter(0.8, (val[i] - pt0) / (pt2 - pt0) \
                             * height - 1 / np.sqrt(6),
                             color=color_pos, marker='_', s=5)

        axs[col].text(0.86, (pt0 - pt0) / (pt2 - pt0) * height - 1 / np.sqrt(6),
                 "{:.2f}".format(pt0), color='black', size=6)
        axs[col].text(0.86, (pt1 - pt0) / (pt2 - pt0) * height - 1 / np.sqrt(6),
                 "{:.2f}".format(pt1), color='black', size=6)
        axs[col].text(0.86, (pt2 - pt0) / (pt2 - pt0) * height - 1 / np.sqrt(6),
                 "{:.2f}".format(pt2), color='black', size=6)
        axs[col].axis('equal')
        axs[col].axis('off')

    plt.savefig('tmp.pdf')

#----------------------------------------------------------------------
# final plots
#----------------------------------------------------------------------
# plot_descent_factors(reward)

method_name_list = ['Regular', 'Reg w/o baseline',
                    'Alternate', 'Alt w/o baseline']
eta_list = ['pi', 'pi', 0, 0]
baseline_list = ['rpi', 0, 'rpi', 0]
save_bname = 'rpi'

# reward = np.array([0, 0, 1]).reshape(-1, 1); 
# stepsize = 0.8
# reward_noise = 0
# V_R_A = np.array([0, 0, 0]).reshape(-1, 1)
# plot_gradient_field(reward=reward,
#                     reward_noise=reward_noise,
#                     method_name_list=method_name_list,
#                     eta_list=eta_list, baseline_list=baseline_list,
#                     plot_expected_curves_cols=[0, 3],
#                     save_bname=save_bname, stepsize=stepsize)
# plot_variance(reward=reward, V_R_A=V_R_A,
#               method_name_list=method_name_list,
#               eta_list=eta_list, baseline_list=baseline_list,
#               save_bname=save_bname, res=20)
# plot_entropy_vs_variance(reward=reward, V_R_A=V_R_A,
#                          method_name_list=method_name_list,
#                          eta_list=eta_list,
#                          baseline_list=baseline_list,
#                          save_bname=save_bname, res=20)

# reward = np.array([0, 0, 1]).reshape(-1, 1); 
# stepsize = 0.4
# reward_noise = 1
# V_R_A = np.array([1, 1, 1]).reshape(-1, 1)
# plot_gradient_field(reward=reward,
#                     reward_noise=reward_noise,
#                     method_name_list=method_name_list,
#                     eta_list=eta_list, baseline_list=baseline_list,
#                     plot_expected_curves_cols=[0, 3],
#                     save_bname=save_bname, stepsize=stepsize)
# plot_variance(reward=reward, V_R_A=V_R_A,
#               method_name_list=method_name_list,
#               eta_list=eta_list, baseline_list=baseline_list,
#               save_bname=save_bname, res=20)
# plot_entropy_vs_variance(reward=reward, V_R_A=V_R_A,
#                          method_name_list=method_name_list,
#                          eta_list=eta_list,
#                          baseline_list=baseline_list,
#                          save_bname=save_bname, res=20)

# reward = np.array([1, 2, 3]).reshape(-1, 1);
# stepsize = 0.3
# reward_noise = 1
# V_R_A = np.array([1, 1, 1]).reshape(-1, 1)
# plot_gradient_field(reward=reward,
#                     reward_noise=reward_noise,
#                     method_name_list=method_name_list,
#                     eta_list=eta_list, baseline_list=baseline_list,
#                     plot_expected_curves_cols=[0, 3],
#                     save_bname=save_bname, stepsize=stepsize)
# plot_variance(reward=reward, V_R_A=V_R_A,
#               method_name_list=method_name_list,
#               eta_list=eta_list, baseline_list=baseline_list,
#               save_bname=save_bname, res=20)
# plot_entropy_vs_variance(reward=reward, V_R_A=V_R_A,
#                          method_name_list=method_name_list,
#                          eta_list=eta_list,
#                          baseline_list=baseline_list,
#                          save_bname=save_bname, res=20)

# reward = np.array([1, 2, 3]).reshape(-1, 1);
# stepsize = 0.5
# reward_noise = 0
# V_R_A = np.array([0, 0, 0]).reshape(-1, 1)
# plot_gradient_field(reward=reward,
#                     reward_noise=reward_noise,
#                     method_name_list=method_name_list,
#                     eta_list=eta_list, baseline_list=baseline_list,
#                     plot_expected_curves_cols=[0, 3],
#                     save_bname=save_bname, stepsize=stepsize)
# plot_variance(reward=reward, V_R_A=V_R_A,
#               method_name_list=method_name_list,
#               eta_list=eta_list, baseline_list=baseline_list,
#               save_bname=save_bname, res=20)
# plot_entropy_vs_variance(reward=reward, V_R_A=V_R_A,
#                          method_name_list=method_name_list,
#                          eta_list=eta_list,
#                          baseline_list=baseline_list,
#                          save_bname=save_bname, res=20)
            
# reward = np.array([1, 2, 3]).reshape(-1, 1); 
# reward_noise = 0
# V_R_A = np.array([0, 0, 0]).reshape(-1, 1)
# method_name_list = ['Reg (+4)', 'Alt (+4)', 'Reg (-4)', 'Alt (-4)']
# eta_list = ['pi', 0, 'pi', 0]
# baseline_list = [+4, +4, -4, -4]
# save_bname = 'opt'
# stepsize = 0.15
# plot_gradient_field(reward=reward,
#                     reward_noise=reward_noise,
#                     method_name_list=method_name_list,
#                     eta_list=eta_list, baseline_list=baseline_list,
#                     plot_expected_curves_cols=[0, 1, 3],
#                     save_bname=save_bname, stepsize=stepsize)
# plot_variance(reward=reward, V_R_A=V_R_A, method_name_list=method_name_list,
#               eta_list=eta_list, baseline_list=baseline_list,
#               save_bname=save_bname, res=20)
# plot_entropy_vs_variance(reward=reward, V_R_A=V_R_A,
#                          method_name_list=method_name_list, eta_list=eta_list,
#                          baseline_list=baseline_list, save_bname=save_bname,
#   #                        res=20)

# reward = np.array([-3, -2, -1]).reshape(-1, 1);
# stepsize = 0.3
# reward_noise = 1
# V_R_A = np.array([1, 1, 1]).reshape(-1, 1)
# plot_gradient_field(reward=reward,
#                     reward_noise=reward_noise,
#                     method_name_list=method_name_list,
#                     eta_list=eta_list, baseline_list=baseline_list,
#                     plot_expected_curves_cols=[0, 3],
#                     save_bname=save_bname, stepsize=stepsize)
# plot_variance(reward=reward, V_R_A=V_R_A,
#               method_name_list=method_name_list,
#               eta_list=eta_list, baseline_list=baseline_list,
#               save_bname=save_bname, res=20)
# plot_entropy_vs_variance(reward=reward, V_R_A=V_R_A,
#                          method_name_list=method_name_list,
#                          eta_list=eta_list,
#                          baseline_list=baseline_list,
#                          save_bname=save_bname, res=20)

# reward = np.array([-3, -2, -1]).reshape(-1, 1); 
# reward_noise = 0
# V_R_A = np.array([0, 0, 0]).reshape(-1, 1)
# method_name_list = ['Reg (+4)', 'Alt (+4)', 'Reg (-4)', 'Alt (-4)']
# eta_list = ['pi', 0, 'pi', 0]
# baseline_list = [+4, +4, -4, -4]
# save_bname = 'opt'
# stepsize = 0.15
# plot_gradient_field(reward=reward,
#                     reward_noise=reward_noise,
#                     method_name_list=method_name_list,
#                     eta_list=eta_list, baseline_list=baseline_list,
#                     plot_expected_curves_cols=[0, 1, 3],
#                     save_bname=save_bname, stepsize=stepsize)
# plot_variance(reward=reward, V_R_A=V_R_A, method_name_list=method_name_list,
#               eta_list=eta_list, baseline_list=baseline_list,
#               save_bname=save_bname, res=20)
# plot_entropy_vs_variance(reward=reward, V_R_A=V_R_A,
#                          method_name_list=method_name_list, eta_list=eta_list,
#                          baseline_list=baseline_list, save_bname=save_bname,
#                          res=20)

# reward = np.array([1, 2, 3]).reshape(-1, 1); 
# reward_noise = 1
# V_R_A = np.array([1, 1, 1]).reshape(-1, 1)
# method_name_list = ['Reg (+2)', 'Alt (+2)', 'Reg (-2)', 'Alt (-2)']
# eta_list = ['pi', 0, 'pi', 0]
# baseline_list = [+2, +2, -2, -2]
# save_bname = 'opt'
# stepsize = 0.15
# plot_gradient_field(reward=reward,
#                     reward_noise=reward_noise,
#                     method_name_list=method_name_list,
#                     eta_list=eta_list, baseline_list=baseline_list,
#                     plot_expected_curves_cols=[0, 1, 3],
#                     save_bname=save_bname, stepsize=stepsize)
# plot_variance(reward=reward, V_R_A=V_R_A, method_name_list=method_name_list,
#               eta_list=eta_list, baseline_list=baseline_list,
#               save_bname=save_bname, res=20)
# plot_entropy_vs_variance(reward=reward, V_R_A=V_R_A,
#                          method_name_list=method_name_list, eta_list=eta_list,
#                          baseline_list=baseline_list, save_bname=save_bname,
#                          res=20)

# reward = np.array([1, 1, 2]).reshape(-1, 1); 
# reward_noise = 1
# V_R_A = np.array([1, 1, 1]).reshape(-1, 1)
# method_name_list = ['Reg (+2)', 'Alt (+2)', 'Reg (-2)', 'Alt (-2)']
# eta_list = ['pi', 0, 'pi', 0]
# baseline_list = [+2, +2, -2, -2]
# save_bname = 'opt'
# stepsize = 0.15
# plot_gradient_field(reward=reward,
#                     reward_noise=reward_noise,
#                     method_name_list=method_name_list,
#                     eta_list=eta_list, baseline_list=baseline_list,
#                     plot_expected_curves_cols=[0, 1, 3],
#                     save_bname=save_bname, stepsize=stepsize)
# plot_variance(reward=reward, V_R_A=V_R_A, method_name_list=method_name_list,
#               eta_list=eta_list, baseline_list=baseline_list,
#               save_bname=save_bname, res=20)
# plot_entropy_vs_variance(reward=reward, V_R_A=V_R_A,
#                          method_name_list=method_name_list, eta_list=eta_list,
#                          baseline_list=baseline_list, save_bname=save_bname,
#                          res=20)

# reward = np.array([1, 2, 2]).reshape(-1, 1); 
# reward_noise = 1
# V_R_A = np.array([1, 1, 1]).reshape(-1, 1)
# method_name_list = ['Reg (+2)', 'Alt (+2)', 'Reg (-2)', 'Alt (-2)']
# eta_list = ['pi', 0, 'pi', 0]
# baseline_list = [+2, +2, -2, -2]
# save_bname = 'opt'
# stepsize = 0.15
# plot_gradient_field(reward=reward,
#                     reward_noise=reward_noise,
#                     method_name_list=method_name_list,
#                     eta_list=eta_list, baseline_list=baseline_list,
#                     plot_expected_curves_cols=[0, 1, 3],
#                     save_bname=save_bname, stepsize=stepsize)
# plot_variance(reward=reward, V_R_A=V_R_A, method_name_list=method_name_list,
#               eta_list=eta_list, baseline_list=baseline_list,
#               save_bname=save_bname, res=20)
# plot_entropy_vs_variance(reward=reward, V_R_A=V_R_A,
#                          method_name_list=method_name_list, eta_list=eta_list,
#                          baseline_list=baseline_list, save_bname=save_bname,
#                          res=20)

exit()

