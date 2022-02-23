import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb

fig = plt.figure()
ax = fig.gca(projection='3d')

min_val = -10
max_val = +10
num_pts = 100
r = [1, 2]

theta0 = theta1 = np.linspace(min_val, max_val, num_pts)
X, Y = np.meshgrid(theta0, theta1)

normalizing_constant = np.exp(X) + np.exp(Y)
P0 = np.exp(X) / normalizing_constant
P1 = np.exp(Y) / normalizing_constant
J = P0 * r[0] + P1 * r[1]

surf = ax.plot_surface(X, Y, J, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()

pdb.set_trace()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig = plt.figure()

# Make data.

X = np.arange(-3, 3, 0.25)
Y = np.arange(-3, 3, 0.25)
X, Y = np.meshgrid(X, Y)

ax = fig.add_subplot(1, 2, 1, projection='3d')
for alpha in [0.000001, 0.01, 0.05, 0.1]:
    c = -2*np.log(2*np.sqrt(2)*np.pi*alpha)

    Z = c - (X+Y)**2/4
    Z[Z<0] = 0
    Z = np.sqrt(Z)

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
		    linewidth=0, antialiased=False)
    ax.plot_surface(X, Y, -Z, cmap=cm.coolwarm,
		    linewidth=0, antialiased=False)

ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')

ax = fig.add_subplot(1, 2, 2, projection='3d')
for alpha in [0.000001, 0.0001, 0.02]:
    c = -2*np.log((2*np.pi)**(3/2)*alpha)

    Z = c - X**2 - Y**2
    Z[Z<0] = 0
    Z = np.sqrt(Z)

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
		    linewidth=0, antialiased=False)
    ax.plot_surface(X, Y, -Z, cmap=cm.coolwarm,
		    linewidth=0, antialiased=False)

ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')

plt.show()
