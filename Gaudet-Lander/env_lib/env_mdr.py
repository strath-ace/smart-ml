import numpy as np
from time import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from . import env_utils as envu
from .ic_gen import Landing_icgen
import pylab
import matplotlib

class Env(object):
    def __init__(self, lander, dynamics, logger,  
                 glideslope_constraint=None,
                 shape_constraint=None,
                 attitude_constraint=None,
                 reward_object=None,
                 nav_period=0.20, 
                 rh_limit = 500, tf_limit=200.0, allow_plotting=True, print_every=1,  
                 scale_agent_action=True, pp_debug=False): 
        self.scale_agent_action = scale_agent_action
        self.rh_limit = rh_limit
        self.nav_period = nav_period 
        self.logger = logger
        self.lander = lander
        self.pp_debug = pp_debug
        self.rl_stats = RL_stats(lander,logger,print_every=print_every,allow_plotting=allow_plotting) 
        self.tf_limit = tf_limit
        self.display_errors = False
        self.dynamics = dynamics 
        self.allow_plotting = allow_plotting
        self.ic_gen = Landing_icgen()
        self.episode = 0
        self.glideslope_constraint = glideslope_constraint
        self.shape_constraint = shape_constraint
        self.attitude_constraint = attitude_constraint
        self.reward_object = reward_object

        if allow_plotting:
            plt.clf()
            plt.cla()
        print('lander env att 2 fixed')
        
    def reset(self): 

        self.ic_gen.set_ic(self.lander, self.dynamics)
        self.glideslope_constraint.reset(self.lander.state)
 
        self.steps = 0
        self.min_range = 1e12
        self.t = 0.0

        self.lander.clear_trajectory()
        if self.pp_debug:
            print('PP debug: ',self.lander.state['mass'], self.dynamics.g)
        return self.lander.get_state_agent(self.t)

    def check_for_done(self,lander):
        done = False
        vc = envu.get_vc(lander.state['position'], lander.state['velocity'])
        altitude = lander.state['position'][2]
        range = np.linalg.norm(lander.state['position'])
        if self.glideslope_constraint.get_margin() < 0.0 and self.glideslope_constraint.terminate_on_violation:  
            done = True
        if self.shape_constraint.get_margin(lander.state) < 0.0 and self.shape_constraint.terminate_on_violation:
            done = True
        if self.attitude_constraint.get_margin(lander.state) < 0.0 and self.attitude_constraint.terminate_on_violation:
            done = True
        if altitude < 0.0: 
            done = True
        if self.t > self.tf_limit:
            done = True
            error = True
        if (range - self.min_range) > self.rh_limit:
            done = True
            error = True

        return done

    def render(self):
        self.rl_stats.show(self.episode)
        self.rl_stats.clear()
        if self.allow_plotting:
            envu.render_traj(self.lander.trajectory)


    def step(self,action):
        action = action.copy()
        if len(action.shape) > 1:
            action = action[0]

        # update using current state and action so we capture IC
        self.lander.update_trajectory(False, self.t, action)

        steps_to_sim = int(np.round(self.nav_period / self.dynamics.h))
        self.lander.prev_state = self.lander.state.copy()
        for i in range(steps_to_sim): 
            self.dynamics.next(self.t, action, self.lander)
            done = self.check_for_done(self.lander)
            if done:
                break
        
        self.t +=  steps_to_sim*self.dynamics.h
        self.steps+=1
        self.glideslope_constraint.calculate(self.lander.state)
        self.min_range = np.minimum(self.min_range, np.linalg.norm(self.lander.state['position']))

        reward,reward_info = self.reward_object.get( self.lander, action, done, self.steps, self.shape_constraint, self.glideslope_constraint, self.attitude_constraint)
        state = self.lander.get_state_agent(self.t)
        if done:
            # also add last state to trajectory
            self.lander.update_trajectory(done, self.t, action)
            self.episode += 1
        return state,reward,done,reward_info

    def test_policy_batch(self, policy, input_normalizer, n, print_every=100, use_ts=False):
        self.lander.use_trajectory_list = True
        self.episode = 0
        self.lander.trajectory_list = []
        self.display_errors = True
        for i in range(n):
            self.test_policy_episode(policy, input_normalizer,use_ts=use_ts)
            if i % print_every == 0 and i > 0:
                print('i : ',i)
                self.lander.show_cum_stats()
                print('')
                self.lander.show_final_stats(type='final')
        print('')
        self.lander.show_cum_stats()
        print('')
        self.lander.show_final_stats(type='final')
        print('')
        self.lander.show_final_stats(type='ic')

    def test_policy_episode(self, policy, input_normalizer, use_ts=False): 
        trajectory = []
        obs = self.reset()
        done = False
        ts = 0.0
        while not done:
            obs = input_normalizer.apply(obs)
            obs = np.expand_dims(obs,axis=0)
            if use_ts:
                obs = np.append(obs, [[ts]], axis=1)
            _, env_action = policy.sample(obs)
            obs, _, done,_ =self.step(env_action)
            ts += 1e-3
        return self.lander.get_state_agent(0) 



class RL_stats(object):
    
    def __init__(self,lander,logger,allow_plotting=True,print_every=1, vf=None, scaler=None):
        self.logger = logger
        self.lander = lander
        self.scaler = scaler
        self.vf     = vf
        self.keys = ['r_f',  'v_f', 'r_i', 'v_i', 'norm_rf', 'norm_vf', 'gs_f', 'thrust', 'norm_thrust','fuel', 'rewards', 'fuel_rewards', 
                     'glideslope_rewards', 'glideslope_penalty', 'glideslope', 'norm_af', 'norm_wf', 
                     'sc_rewards','sc_penalty','sc_margin',
                     'att_rewards', 'att_penalty', 'attitude', 'w', 'a_f', 'w_f',
                     'landing_rewards','landing_margin', 'tracking_rewards', 'steps']
        self.formats = {'gs_f' : '{:8.1f}' , 'r_f' : '{:8.1f}', 'v_f' : '{:8.2f}' , 'r_i' : '{:8.1f}', 'v_i' : '{:8.2f}' , 'norm_rf' : '{:8.1f}', 'norm_vf' : '{:8.2f}', 'thrust' : '{:8.0f}',
                         'norm_af' : '{:8.2f}', 'norm_wf' : '{:8.2f}',
                        'fuel' : '{:8.0f}', 'steps'  : '{:8.0f}', 'r_bonus' : '{:8.2f}', 'v_bonus' : '{:8.2f}', 'norm_thrust' : '{:8.0f}', 'a_f' : '{:8.2f}',  'w_f' : '{:8.2f}',
                        'tracking_rewards' : '{:8.2f}', 'rewards' : '{:8.2f}', 'fuel_rewards' : '{:8.2f}', 'landing_margin' : '{:8.2f}',  
                        'glideslope_rewards' : '{:8.2f}',  'glideslope' : '{:8.2f}', 'glideslope_penalty' : '{:8.2f}',  
                        'sc_rewards' : '{:8.2f}', 'sc_penalty' : '{:8.2f}', 'sc_margin' : '{:8.2f}',
                        'att_rewards' :  '{:8.2f}', 'landing_rewards': '{:8.2f}', 'att_penalty' :  '{:8.2f}', 'attitude' : '{:8.2f}', 'w' : '{:8.2f}'}
        self.stats = {}
        self.history =  { 'Episode' : [] , 'MeanReward' : [], 'StdReward' : [] , 'MinReward' : [],  'KL' : [], 'Beta' : [], 'Variance' : [], 'PolicyEntropy' : [], 'ExplainedVarNew' :  [] , 
                          'Norm_rf' : [], 'Norm_vf' : [], 'SD_rf' : [], 'SD_vf' : [], 'Max_rf' : [], 'Max_vf' : [], 
                          'Norm_af' : [], 'Norm_wf' : [], 'SD_af' : [], 'SD_wf' : [], 'Max_af' : [], 'Max_wf' : [], 'MeanSteps' : [], 'MaxSteps' : []} 

        self.clear()
        
        self.allow_plotting = allow_plotting 
        self.last_time  = time() 

        self.update_cnt = 0
        self.episode = 0
        self.print_every = print_every 

        
        if allow_plotting:
            plt.clf()
            plt.cla()
            self.fig2 = plt.figure(2,figsize=plt.figaspect(0.5))
            self.fig3 = plt.figure(3,figsize=plt.figaspect(0.5))
            self.fig4 = plt.figure(4,figsize=plt.figaspect(0.5))
            self.fig5 = plt.figure(5,figsize=plt.figaspect(0.5))
            self.fig6 = plt.figure(6,figsize=plt.figaspect(0.5))
            self.fig7 = plt.figure(7,figsize=plt.figaspect(0.5))

    def clear(self):
        for k in self.keys:
            self.stats[k] = []
    
    def update_episode(self,sum_rewards,steps):    
        self.stats['rewards'].append(sum_rewards)
        self.stats['fuel_rewards'].append(np.sum(self.lander.trajectory['fuel_reward']))
        self.stats['tracking_rewards'].append(np.sum(self.lander.trajectory['tracking_reward']))
        self.stats['glideslope_rewards'].append(np.sum(self.lander.trajectory['glideslope_reward']))
        self.stats['glideslope_penalty'].append(np.sum(self.lander.trajectory['glideslope_penalty']))
        self.stats['glideslope'].append(np.min(self.lander.trajectory['glideslope']))
        self.stats['sc_rewards'].append(np.sum(self.lander.trajectory['sc_reward']))
        self.stats['sc_penalty'].append(np.sum(self.lander.trajectory['sc_penalty']))
        self.stats['sc_margin'].append(np.min(self.lander.trajectory['sc_margin']))
        self.stats['att_penalty'].append(np.sum(self.lander.trajectory['att_penalty']))
        self.stats['att_rewards'].append(np.sum(self.lander.trajectory['att_reward']))
        self.stats['landing_rewards'].append(np.sum(self.lander.trajectory['landing_reward'])) 
        self.stats['attitude'].append(self.lander.trajectory['attitude_321'])
        self.stats['w'].append(self.lander.trajectory['w'])

        self.stats['landing_margin'].append(np.sum(self.lander.trajectory['landing_margin']))
        self.stats['r_f'].append(self.lander.trajectory['position'][-1])
        self.stats['v_f'].append(self.lander.trajectory['velocity'][-1])
        self.stats['gs_f'].append(self.lander.trajectory['glideslope'][-1])
        self.stats['w_f'].append(self.lander.trajectory['w'][-1])
        self.stats['a_f'].append(self.lander.trajectory['attitude_321'][-1][1:3])
        self.stats['w_f'].append(self.lander.trajectory['w'][-1])
        self.stats['r_i'].append(self.lander.trajectory['position'][0])
        self.stats['v_i'].append(self.lander.trajectory['velocity'][0])
        self.stats['norm_rf'].append(np.linalg.norm(self.lander.trajectory['position'][-1]))
        self.stats['norm_vf'].append(np.linalg.norm(self.lander.trajectory['velocity'][-1]))
        self.stats['norm_af'].append(np.linalg.norm(self.lander.trajectory['attitude_321'][-1][1:3]))  # don't care about yaw
        self.stats['norm_wf'].append(np.linalg.norm(self.lander.trajectory['w'][-1]))

        self.stats['norm_thrust'].append(np.linalg.norm(self.lander.trajectory['thrust'],axis=1))
        self.stats['thrust'].append(self.lander.trajectory['thrust'])
        self.stats['fuel'].append(np.linalg.norm(self.lander.trajectory['fuel'][-1]))
        self.stats['steps'].append(steps)
        self.episode += 1
 
    # called by render at policy update 
    def show(self):
 
        self.history['MeanReward'].append(np.mean(self.stats['rewards']))
        self.history['StdReward'].append(np.std(self.stats['rewards']))
        self.history['MinReward'].append(np.min(self.stats['rewards']))

        self.history['KL'].append(self.logger.log_entry['KL'])
        self.history['Beta'].append(self.logger.log_entry['Beta'])
        self.history['Variance'].append(self.logger.log_entry['Variance'])
        self.history['PolicyEntropy'].append(self.logger.log_entry['PolicyEntropy'])
        self.history['ExplainedVarNew'].append(self.logger.log_entry['ExplainedVarNew'])
        self.history['Episode'].append(self.episode)

        self.history['Norm_rf'].append(np.mean(self.stats['norm_rf']))
        self.history['SD_rf'].append(np.mean(self.stats['norm_rf']+np.std(self.stats['norm_rf'])))
        self.history['Max_rf'].append(np.max(self.stats['norm_rf']))

        self.history['Norm_vf'].append(np.mean(self.stats['norm_vf']))
        self.history['SD_vf'].append(np.mean(self.stats['norm_vf']+np.std(self.stats['norm_vf'])))
        self.history['Max_vf'].append(np.max(self.stats['norm_vf']))

        self.history['Norm_af'].append(np.mean(self.stats['norm_af']))
        self.history['SD_af'].append(np.mean(self.stats['norm_af']+np.std(self.stats['norm_af'])))
        self.history['Max_af'].append(np.max(self.stats['norm_af']))

        self.history['Norm_wf'].append(np.mean(self.stats['norm_wf']))
        self.history['SD_wf'].append(np.mean(self.stats['norm_wf']+np.std(self.stats['norm_wf'])))
        self.history['Max_wf'].append(np.max(self.stats['norm_wf']))


        self.history['MeanSteps'].append(np.mean(self.stats['steps']))
        self.history['MaxSteps'].append(np.max(self.stats['steps']))

        if self.allow_plotting:
            envu.render_traj(self.lander.trajectory,vf=self.vf,scaler=self.scaler)

            self.plot_rewards()
            self.plot_learning()
            self.plot_rf()
            self.plot_vf()
            self.plot_af()
            self.plot_wf()
        if self.update_cnt % self.print_every == 0:
            self.show_stats()
            self.clear()
        self.update_cnt += 1

    def show_stats(self):
        et = time() - self.last_time
        self.last_time = time()

        r_f = np.linalg.norm(self.stats['r_f'],axis=1)
        v_f = np.linalg.norm(self.stats['v_f'],axis=1)       
 
        f = '{:6.2f}'
        print('Update Cnt = %d    ET = %8.1f   Stats:  Mean, Std, Min, Max' % (self.update_cnt,et))
        for k in self.keys:
            f = self.formats[k]    
            v = self.stats[k]
            if k == 'thrust' or k=='tracking_error' or k=='norm_thrust' or k=='attitude' or k=='w': 
                v = np.concatenate(v)
            v = np.asarray(v)
            if len(v.shape) == 1 :
                v = np.expand_dims(v,axis=1)
            s = '%-8s' % (k)
            #print('foo: ',k,v)
            s += envu.print_vector(' |',np.mean(v,axis=0),f)
            s += envu.print_vector(' |',np.std(v,axis=0),f)
            s += envu.print_vector(' |',np.min(v,axis=0),f)
            s += envu.print_vector(' |',np.max(v,axis=0),f)
            print(s)

        #print('R_F, Mean, SD, Min, Max: ',np.mean(r_f), np.std(r_f))
        #print('V_F, Mean, SD, Min, Max: ',np.mean(v_f), np.mean(v_f))
 
 
    def plot_rewards(self):
        self.fig2.clear()
        plt.figure(self.fig2.number)
        self.fig2.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        ax = plt.gca()
        ax2 = ax.twinx()
        
        lns1=ax.plot(ep,self.history['MeanReward'],'r',label='Mean R')
        lns2=ax.plot(ep,np.asarray(self.history['MeanReward'])-np.asarray(self.history['StdReward']),'b',label='SD R')
        lns3=ax.plot(ep,self.history['MinReward'],'g',label='Min R')
        lns4=ax2.plot(ep,self.history['MaxSteps'],'c',linestyle=':',label='Max Steps')
        lns5=ax2.plot(ep,self.history['MeanSteps'],'m',linestyle=':',label='Mean Steps')

        lns = lns1+lns2+lns3+lns4+lns5
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("Episode")

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
        lns4=ax.plot(ep,self.history['Beta'],'k',label='Beta')
        foo = 10*np.asarray(self.history['Variance'])
        lns5=ax.plot(ep,foo,'m',label='10X Variance')


        lns = lns1+lns2+lns3+lns4+lns5
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("Update")
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
        
        plt.plot(ep,self.history['Norm_rf'],'r',label='Norm_rf')
        plt.plot(ep,self.history['SD_rf'], 'b',linestyle=':',label='SD_rf')
        plt.plot(ep,self.history['Max_rf'], 'g',linestyle=':',label='Max_rf')
 
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig4.canvas.draw()

    def plot_vf(self):
        self.fig5.clear()
        plt.figure(self.fig5.number)
        self.fig5.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        
        plt.plot(ep,self.history['Norm_vf'],'r',label='Norm_vf')
        plt.plot(ep,self.history['SD_vf'], 'b',linestyle=':',label='SD_vf')
        plt.plot(ep,self.history['Max_vf'], 'g',linestyle=':',label='Max_vf')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig5.canvas.draw()

    def plot_af(self):
        self.fig6.clear()
        plt.figure(self.fig6.number)
        self.fig6.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']

        plt.plot(ep,self.history['Norm_af'],'r',label='Norm_af')
        plt.plot(ep,self.history['SD_af'], 'b',linestyle=':',label='SD_af')
        plt.plot(ep,self.history['Max_af'], 'g',linestyle=':',label='Max_af')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig6.canvas.draw()

    def plot_wf(self):
        self.fig7.clear()
        plt.figure(self.fig7.number)
        self.fig7.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']

        plt.plot(ep,self.history['Norm_wf'],'r',label='Norm_wf')
        plt.plot(ep,self.history['SD_wf'], 'b',linestyle=':',label='SD_wf')
        plt.plot(ep,self.history['Max_wf'], 'g',linestyle=':',label='Max_wf')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig7.canvas.draw()
 
