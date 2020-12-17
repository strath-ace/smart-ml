import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gp_set = np.load('GP_creationSet.npy')
nn_set = np.load('TestSetNN.npy')
train_set = np.load('training_points_reduced.npy')

fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(gp_set[:,0]/1000, gp_set[:,1], gp_set[:,2]/1000, alpha=1)

ax.set_xlabel('Gust start height [km]')
ax.set_ylabel('Gust intensity [m/s]')
ax.set_zlabel('Gust zone size [km]')

plt.savefig('gp_set.png', bbox_inches = 'tight', pad_inches = 0.2)

fig = plt.figure(1)
bx = fig.add_subplot(111, projection='3d')

bx.scatter(nn_set[:,0]/1000, nn_set[:,1], nn_set[:,2]/1000, alpha=0.7)

bx.set_xlabel('Gust start height [km]')
bx.set_ylabel('Gust intensity [m/s]')
bx.set_zlabel('Gust zone size [km]')

plt.savefig('nn_set.png', bbox_inches = 'tight', pad_inches = 0.2)

fig = plt.figure(2)
cx = fig.add_subplot(111, projection='3d')

cx.scatter(train_set[:,0]/1000, train_set[:,1], train_set[:,2]/1000, alpha=0.7)

cx.set_xlabel('Gust start height [km]')
cx.set_ylabel('Gust intensity [m/s]')
cx.set_zlabel('Gust zone size [km]')

plt.savefig('train_set.png', bbox_inches = 'tight', pad_inches = 0.2)

plt.show()