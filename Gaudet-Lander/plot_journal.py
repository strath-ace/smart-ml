import numpy as np
from time import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab
import matplotlib
import matplotlib.gridspec as gridspec
 
class Plot_journal(object):
    def __init__(self, history):
        self.history = history 

        plt.clf()
        plt.cla()
        self.fig1 = plt.figure(1,figsize=plt.figaspect(0.5))
        self.fig2 = plt.figure(2,figsize=plt.figaspect(0.5))
        self.fig3 = plt.figure(3,figsize=plt.figaspect(0.5))
        self.fig4 = plt.figure(4,figsize=plt.figaspect(0.5))
        self.fig5 = plt.figure(5,figsize=plt.figaspect(0.5))

        self.plot_rewards()
        self.plot_learning()
        self.plot_rf()
        self.plot_vf()

    def plot_rewards(self):
        self.fig2.clear()
        plt.figure(self.fig2.number)
        self.fig2.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        ax = plt.gca()
        ax2 = ax.twinx()

        lns1=ax.plot(ep,self.history['MeanReward'],'r',label='Mean Reward')
        lns2=ax.plot(ep,np.asarray(self.history['MeanReward'])-np.asarray(self.history['StdReward']),'b',label='StdDev Reward')
        lns3=ax.plot(ep,self.history['MinReward'],'g',label='Min Reward')
        lns4=ax2.plot(ep,self.history['MeanSteps'],'m',linestyle=':',label='Mean Steps')

        lns = lns1+lns2+lns3+lns4
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax2.set_ylabel("Steps")
        ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=5, mode="expand", borderaxespad=0.)
        ax.grid(True)
        ax = plt.gca()
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig2.canvas.draw()

    def plot_learning(self):
        self.fig3.clear()
        plt.figure(self.fig3.number)
        self.fig3.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        ax = plt.gca()
        ax2 = ax.twinx()
        lns1=ax.plot(ep,self.history['PolicyEntropy'],'r',label='Entropy')
        lns2=ax2.plot(ep,self.history['KL'],'b',label='KL Divergence')
        lns3=ax.plot(ep,self.history['ExplainedVarNew'],'g',label='Explained Variance')
        #lns4=ax.plot(ep,self.history['Beta'],'k',label='Beta')
        #foo = 10*np.asarray(self.history['Variance'])
        #lns5=ax.plot(ep,foo,'m',label='10X Variance')


        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("Update")
        ax.set_ylabel("Entropy, Explained Variance")
        ax2.set_ylabel("KL Divergence")
        ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig3.canvas.draw()

    def plot_rf(self):
        self.fig4.clear()
        plt.figure(self.fig4.number)
        self.fig4.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']

        plt.plot(ep,self.history['Norm_rf'],'r',label='Mean Miss')
        plt.plot(ep,self.history['SD_rf'], 'b',linestyle=':',label='StdDev Miss')
        plt.plot(ep,self.history['Max_rf'], 'g',linestyle=':',label='Max Miss')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Miss Distance (m)")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig4.canvas.draw()

    def plot_vf(self):
        self.fig5.clear()
        plt.figure(self.fig5.number)
        self.fig5.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']

        plt.plot(ep,self.history['Norm_vf'],'r',label='Mean Terminal Speed')
        plt.plot(ep,self.history['SD_vf'], 'b',linestyle=':',label='StdDev Terminal Speed')
        plt.plot(ep,self.history['Max_vf'], 'g',linestyle=':',label='Max Terminal Speed')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Terminal Speed (m/s)")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig5.canvas.draw()


   
def render_traj_3dof(traj, vf=None, scaler=None):
    fig1 = plt.figure(1,figsize=plt.figaspect(0.5))
    fig1.clear()
    plt.figure(fig1.number)
    fig1.set_size_inches(8, 4, forward=True)
    gridspec.GridSpec(2,2)
    t = np.asarray(traj['t'])
    pos = np.asarray(traj['position'])
    vel = np.asarray(traj['velocity'])
    norm_pos = np.linalg.norm(pos,axis=1)
    norm_vel = np.linalg.norm(vel,axis=1)

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    plt.subplot2grid( (2,2) , (0,0) )
    plt.plot(t,x,'r',label='x')
    plt.plot(t,y,'b',label='y')
    plt.plot(t,z,'g',label='z')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Position (m)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)


    x = vel[:,0]
    y = vel[:,1]
    z = vel[:,2]
    plt.subplot2grid( (2,2) , (0,1))
    plt.plot(t,x,'r',label='x')
    plt.plot(t,y,'b',label='y')
    plt.plot(t,z,'g',label='z')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Velocity (m/s)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    xy = np.sqrt(y**2 + x**2)
    #xy = xy[::-1]
    plt.subplot2grid( (2,2) , (1,0) )
    if vf is not None and scaler is not None:
        state = np.hstack((pos,vel))
        values = vf.predict(scaler.apply(state))
        plt.plot(t,values,'r',label='V')
    else:
        plt.plot(xy,z,'r',label='altitude')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_xlabel("norm crossrange and downrange (m)")
    plt.gca().set_ylabel('altitude (m)')
    plt.xlim(1500,0)
    plt.ylim(0,2000)
    plt.grid(True)


    thrust = np.asarray(traj['thrust'])
    x = thrust[:,0]
    y = thrust[:,1]
    z = thrust[:,2]
    plt.subplot2grid( (2,2) , (1,1) )
    plt.plot(t,x,'r',label='x')
    plt.plot(t,y,'b',label='y')
    plt.plot(t,z,'g',label='z')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Thrust (N)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    plt.tight_layout(h_pad=3.0)
    fig1.canvas.draw()

 
def render_traj(traj, vf=None, scaler=None):

    fig1 = plt.figure(1,figsize=plt.figaspect(0.5))
    fig1.clear()
    plt.figure(fig1.number)
    fig1.set_size_inches(8, 6, forward=True)
    gridspec.GridSpec(3,2)
    t = np.asarray(traj['t'])
    pos = np.asarray(traj['position'])
    vel = np.asarray(traj['velocity'])
    norm_pos = np.linalg.norm(pos,axis=1)
    norm_vel = np.linalg.norm(vel,axis=1)

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    plt.subplot2grid( (3,2) , (0,0) )
    plt.plot(t,x,'r',label='x')
    plt.plot(t,y,'b',label='y')
    plt.plot(t,z,'g',label='z')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Position (m)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)


    x = vel[:,0]
    y = vel[:,1]
    z = vel[:,2]
    plt.subplot2grid( (3,2) , (0,1))
    plt.plot(t,x,'r',label='x')
    plt.plot(t,y,'b',label='y')
    plt.plot(t,z,'g',label='z')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Velocity (m/s)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    xy = np.sqrt(y**2 + x**2)
    #xy = xy[::-1]
    plt.subplot2grid( (3,2) , (1,0) )
    if vf is not None and scaler is not None:
        state = np.hstack((pos,vel))
        values = vf.predict(scaler.apply(state))
        plt.plot(t,values,'r',label='V')
    else:
        plt.plot(xy,z,'r',label='altitude')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_xlabel("norm crossrange and downrange (m)")
    plt.gca().set_ylabel('altitude (m)')
    plt.xlim(1500,0)
    plt.ylim(0,2000)
    plt.grid(True)


    thrust = np.asarray(traj['thrust'])
    x = thrust[:,0]
    y = thrust[:,1]
    z = thrust[:,2]
    plt.subplot2grid( (3,2) , (1,1) )
    plt.plot(t,x,'r',label='x')
    plt.plot(t,y,'b',label='y')
    plt.plot(t,z,'g',label='z')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Thrust (N)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    attitude = np.asarray(traj['attitude_321'])
    plt.subplot2grid( (3,2) , (2,0) )
    colors = ['r','b','k','g']
    #print('debug: ',attitude.shape[1],len(colors))
    labels = ['yaw','pitch','roll']
    for i in range(attitude.shape[1]):
        plt.plot(t,attitude[:,i],colors[i],label=labels[i])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Attitude (rad)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    w = np.asarray(traj['w'])
    x = w[:,0]
    y = w[:,1]
    z = w[:,2]
    plt.subplot2grid( (3,2) , (2,1) )
    plt.plot(t,x,'r',label='roll')
    plt.plot(t,y,'b',label='pitch')
    plt.plot(t,z,'k',label='yaw')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_xlabel("Time (s)")
    plt.gca().set_ylabel('Rot. Velocity (rad/s)')
    plt.grid(True)


    plt.tight_layout(h_pad=3.0)
    fig1.canvas.draw()

