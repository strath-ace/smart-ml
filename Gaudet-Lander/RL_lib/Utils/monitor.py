import numpy as np
from time import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab
import matplotlib

class RL_stats(object):

    def __init__(self,logger,allow_plotting=True, x_steps=True):
        self.logger = logger
        self.x_steps = x_steps
        self.history =  { 'TotalSteps' : [],  'Episode' : [] , 'MeanReward' : [], 'StdReward' : [] , 'MinReward' : [],  'KL' : [], 'Beta' : [], 'Variance' : [], 'PolicyEntropy' : [], 'ExplainedVarNew' :  [] }

        self.allow_plotting = allow_plotting
        self.last_time  = time()
        self.episode = 0
        self.update_cnt = 0



        if allow_plotting:
            plt.clf()
            plt.cla()
            self.fig2 = plt.figure(2,figsize=plt.figaspect(0.5))
            self.fig3 = plt.figure(3,figsize=plt.figaspect(0.5))



    def update_episode(self,sum_rewards,steps):
        self.episode += 1

    # called by render at policy update
    def show(self):

        self.history['MeanReward'].append(self.logger.log_entry['_MeanReward'])
        self.history['StdReward'].append(self.logger.log_entry['_StdReward'])
        self.history['MinReward'].append(self.logger.log_entry['_MinReward'])
        self.history['KL'].append(self.logger.log_entry['KL'])
        self.history['Beta'].append(self.logger.log_entry['Beta'])
        self.history['Variance'].append(self.logger.log_entry['Variance'])
        self.history['PolicyEntropy'].append(self.logger.log_entry['PolicyEntropy'])
        self.history['ExplainedVarNew'].append(self.logger.log_entry['ExplainedVarNew'])
        self.history['Episode'].append(self.episode)
        self.history['TotalSteps'].append(self.logger.log_entry['TotalSteps'])
        if self.allow_plotting:
            self.plot_rewards()
            self.plot_learning()

    def plot_rewards(self):
        self.fig2.clear()
        plt.figure(self.fig2.number)
        #dpi = pylab.gcf().get_dpi()
        #dpi = 2 * dpi
        self.fig2.set_size_inches(8, 3, forward=True)
        if not self.x_steps:
            ep = self.history['Episode']
            xname = "Episode"
        else:
            ep = self.history['TotalSteps']
            xname = "Steps"
        plt.plot(ep,self.history['MeanReward'],'r',label='Mean R')
        plt.plot(ep,np.asarray(self.history['MeanReward'])-np.asarray(self.history['StdReward']),'b',linestyle=':', label='SD R')
        plt.plot(ep,self.history['MinReward'],'g',linestyle=':', label='Min R')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel(xname)
        ax.set_ylabel("Reward")
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        plt.grid(True)
        self.fig2.canvas.draw()

    def plot_learning(self):
        self.fig3.clear()
        plt.figure(self.fig3.number)
        self.fig3.set_size_inches(8, 3, forward=True)
        if not self.x_steps:
            ep = self.history['Episode']
            xname = "Episode"
        else:
            ep = self.history['TotalSteps']
            xname = "Steps"
        ax = plt.gca()
        ax2 = ax.twinx()
        lns1=ax.plot(ep,self.history['PolicyEntropy'],'r',label='Entropy')
        lns2=ax2.plot(ep,self.history['KL'],'b',label='KL Divergence')
        lns3=ax.plot(ep,self.history['ExplainedVarNew'],'g',label='Explained Variance')
        lns4=ax.plot(ep,self.history['Beta'],'k',label='Beta')
        foo = 10*np.asarray(self.history['Variance'])
        lns5=ax.plot(ep,foo,'m',label='10X SD')


        lns = lns1+lns2+lns3+lns4+lns5
        labs = [l.get_label() for l in lns]
        ax.set_xlabel(xname)
        ax2.set_ylabel("KL Divergence")
        ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig3.canvas.draw()


