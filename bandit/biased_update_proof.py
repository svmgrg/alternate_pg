import numpy as np
import matplotlib.pyplot as plt
import pdb

reward = np.array([1, 2, 3])
pi = np.array([0.2, 0.1, 0.7])

def calc_beta(alpha, b):
    return alpha * (reward - b)

def calc_t1(alpha, b):
    return 1 / (1 / calc_beta(alpha, b)).sum()

def calc_t2(alpha, b):
    return np.log((pi * np.exp(pi * calc_beta(alpha, b))).sum())

def calc_range(alpha, b):
    Z = pi * alpha * (reward - b)
    return (Z.max() - Z.min())

#======================================================================
# plotting code
#----------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 3))

#----------------------------------------------------------------------
b = -4
stepsize_list = np.arange(0.1, 5.5, 0.001)

# calculate stuff
t1_list = [calc_t1(alpha, b) for alpha in stepsize_list]
t2_list = [calc_t2(alpha, b) for alpha in stepsize_list]
lbound_list = [(pi**2 * calc_beta(alpha, b)).sum() for alpha in stepsize_list]
ubound_list = [((pi**2 * calc_beta(alpha, b)).sum() \
                + calc_range(alpha, b)**2 / 8)
               for alpha in stepsize_list]

# plot t1
axs[0].plot(stepsize_list, t1_list, color='tab:orange', label='t1')

# plot t2 and its bounds
axs[0].plot(stepsize_list, lbound_list, '-.', color='tab:blue', label='lbound',
            linewidth=0.5)
axs[0].plot(stepsize_list, ubound_list, '-.', color='tab:blue', label='ubound',
            linewidth=0.5)
axs[0].fill_between(stepsize_list, lbound_list, ubound_list, alpha=0.1,
                    facecolor='tab:blue')
axs[0].plot(stepsize_list, t2_list, color='tab:blue', label='t2')

# plot x axis
# axs[0].plot(stepsize_list, [0] * len(stepsize_list),  color='k', linewidth=0.5)

axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)

#----------------------------------------------------------------------
alpha = 0.1
baseline_list =  np.arange(2-2, 2+2, 0.001)

# calculate stuff
t1_list = [calc_t1(alpha, b) for b in baseline_list]
t2_list = [calc_t2(alpha, b) for b in baseline_list]
lbound_list = [(pi**2 * calc_beta(alpha, b)).sum() for b in baseline_list]
ubound_list = [((pi**2 * calc_beta(alpha, b)).sum() \
                + calc_range(alpha, b)**2 / 8)
               for b in baseline_list]

# color regions to denote optimistic and pessimistic baselines
axs[1].axvspan(baseline_list.min(), reward.min(), alpha=0.1, color='tab:red')
axs[1].axvspan(reward.max(), baseline_list.max(), alpha=0.1, color='tab:green')

# plot t1
axs[1].plot(baseline_list, t1_list, color='tab:orange', label='t1')

# for hiding the infinite lines
a_coeff = 1
b_coeff = -2 * reward.mean()
n = len(reward)
c_coeff = np.mean([reward[i % n] * reward[(i + 1) % n] for i in range(n)])
D = b_coeff**2 - 4 * a_coeff * c_coeff
b_inf_m = (-b_coeff - np.sqrt(D)) / (2 * a_coeff)
b_inf_p = (-b_coeff + np.sqrt(D)) / (2 * a_coeff)
axs[1].axvline(x=b_inf_m, color='white', linewidth=4)
axs[1].axvline(x=b_inf_p, color='white', linewidth=4)

# plot t2 and its bounds
axs[1].plot(baseline_list, lbound_list, '-.', color='tab:blue', label='lbound',
            linewidth=0.5)
axs[1].plot(baseline_list, ubound_list, '-.', color='tab:blue', label='ubound',
            linewidth=0.5)
axs[1].fill_between(baseline_list, lbound_list, ubound_list, alpha=0.1,
                    facecolor='tab:blue')
axs[1].plot(baseline_list, t2_list, color='tab:blue', label='t2')

# plot x axis
axs[1].plot(baseline_list, [0] * len(baseline_list), color='k', linewidth=0.5)

# show where the rewards lie
for r in reward:
    axs[1].axvline(x=r, linestyle=':', color='black', linewidth=0.5)

axs[1].set_ylim(np.min(t2_list) - 0.03, np.max(t2_list) + 0.03)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)

#----------------------------------------------------------------------
b = +4
stepsize_list = np.arange(0.1, 5.5, 0.001)

# calculate stuff
t1_list = [calc_t1(alpha, b) for alpha in stepsize_list]
t2_list = [calc_t2(alpha, b) for alpha in stepsize_list]
lbound_list = [(pi**2 * calc_beta(alpha, b)).sum() for alpha in stepsize_list]
ubound_list = [((pi**2 * calc_beta(alpha, b)).sum() \
                + calc_range(alpha, b)**2 / 8)
               for alpha in stepsize_list]

# plot t1
axs[2].plot(stepsize_list, t1_list, color='tab:orange', label='t1')

# plot estimated alpha
tmp = pi * (reward - b)
alpha_max_est = (8 / (tmp.max() - tmp.min())**2) \
    * (1 / (1 / (reward - b)).sum() - (pi**2 * (reward - b)).sum())
axs[2].axvline(x=alpha_max_est, color='tab:green', linewidth=0.7)

# plot actual alpha for this problem
idx = np.abs(np.array(t2_list) - np.array(t1_list)).argmin()
alpha_max_actual = stepsize_list[idx]
axs[2].axvline(x=alpha_max_actual, color='tab:green', linewidth=0.7)

# plot t2 and its bounds
axs[2].plot(stepsize_list, lbound_list, '-.', color='tab:blue', label='lbound',
            linewidth=0.5)
axs[2].plot(stepsize_list, ubound_list, '-.', color='tab:blue', label='ubound',
            linewidth=0.5)
axs[2].fill_between(stepsize_list, lbound_list, ubound_list, alpha=0.1,
                    facecolor='tab:blue')
axs[2].plot(stepsize_list, t2_list, color='tab:blue', label='t2')

# plot x axis
# axs[2].plot(stepsize_list, [0] * len(stepsize_list), color='k', linewidth=0.5)

axs[2].spines['right'].set_visible(False)
axs[2].spines['top'].set_visible(False)

plt.legend()
# plt.show()
plt.savefig('biased_update_proof.pdf')
