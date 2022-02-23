import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy as kl_div
import pdb

a = np.arange(0, 1, 0.001)
b = np.arange(0, 1, 0.001)

reward = np.array([1, 2, 3])
baseline = -4
alpha = 1

fp = (1 / (reward - baseline)) / (1 / (reward - baseline)).sum()
print(fp)

for i in range(len(a)):
    for j in range(len(a)):
        if a[i] + b[j] < 1:
            pi = np.array([a[i], b[j], 1 - a[i] - b[j]])
            t1 = np.exp(alpha * pi[0] * (reward[0] - baseline))
            t2 = (pi * np.exp(alpha * pi * (reward - baseline))).sum()
            if (pi - fp)[0] == (pi - fp).max() and t1 <= t2: 
                print(pi)
exit()

pi = np.array([0.56, 0.022, 0.418])
fp = np.array([0.39252336, 0.3271028, 0.28037383])

#======================================================================

pi0 = np.arange(0, 1, 0.01)
pi1 = 1 - pi0
reward = np.array([1, 2])
baseline = -4

ft = (1 / (reward - baseline)) / (1 / (reward - baseline)).sum()

t1 = pi0 - ft[0]
t2 = pi1 - ft[1]

t3 = pi0 / ft[0]
t4 = pi1 / ft[1]

plt.plot(pi0, t1, color='red', label='t1')
plt.plot(pi0, t2, color='green', label='t2')
plt.plot(pi0, t3, color='blue', label='t3')
plt.plot(pi0, t4, color='black', label='t4')
plt.legend()
plt.show()
exit()

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
color_white = np.array(mpl.colors.to_rgb('white'))

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

def cond(pi, reward, baseline, corner, alpha):
    term = np.exp(alpha * pi * (reward.flatten() - baseline))

    if (pi * term).sum() < term[corner]:
        return True
    else:
        return False


#----------------------------------------------------------------------
# plotting functions
#----------------------------------------------------------------------

def plot_policy_space(axs, col, reward, baseline, res=30):
    color_list = [color_red, color_green, color_blue]
    
    tmp = 1 / (reward - baseline)
    kappa = tmp.sum()
    fp = (tmp / kappa)
    fp_2d = np.matmul(proj_2d_mat, fp).copy()

    # tmp = 1 / (reward - baseline)
    # kappa = 10 # tmp.sum()
    # fp = (tmp / kappa)

    fp = fp.flatten()
    
    # generate points
    points, num_points = generate_policy_points(
        res, FLAG_INCLUDE_BOUNDARY=True)
    
    points_2d = np.matmul(proj_2d_mat, points)

    final_bg = np.ones(points.T.shape)
    final_fg = np.ones(points.T.shape)

    print('fp', fp)
    
    for i in range(num_points):
        final_bg[i] = color_white
        final_fg[i] = color_white

        pi = points[:, i]
        shifted_pi = points[:, i] - fp
        alpha = 0.01
        term = np.exp(alpha * pi * (reward.flatten() - baseline))
        expectation_term = (pi * term).sum()

        cond_bg = expectation_term < term[col]
        # cond_bg = term.max() == term[col]
        cond_fg = shifted_pi.max() == shifted_pi[col]
        
        if cond_bg:
            final_bg[i] = color_list[col]
            final_fg[i] = color_list[col]
        if cond_fg:
            final_fg[i] = color_silver

        if cond_fg and not cond_bg:
            final_fg[i] = color_steelblue
                    
    # plot
    ax = axs[col]
    for i in range(num_points):
        ax.scatter(points_2d[0, i], points_2d[1, i],
                   color=final_bg[i], marker='h', s=24)
        ax.scatter(points_2d[0, i], points_2d[1, i],
                   color=final_fg[i], marker='o', s=5)
    ax.scatter(fp_2d[0], fp_2d[1], color=color_black, marker='x', s=24)
    
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

fig, axs = plt.subplots(1, 3, figsize=(9, 9))

reward = np.array([[1, 2, 3]]).T
baseline = -4

for j in range(3):
    plot_policy_space(axs, j, reward=reward, baseline=baseline)

plt.savefig('pessi_plot.pdf')    
# plt.show()
