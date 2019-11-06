# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import pdb

from qnet_agent_v2 import QNetAgent, HARLAgent
from environment import Environment, LanderEnvironment
from lander_sim import heuristic


def data_smooth(data,n_avg):
	# A function to average data over n_avg timesteps
	ind_vec = np.arange(n_avg,len(data)+1,n_avg)
	data_avg = [0]
	for ind in ind_vec:
		data_avg.append(np.mean(data[ind-n_avg:ind]))
	return data_avg


def network_config():
	netcon = {}
	netcon['alpha'] = 0.1
	netcon['gamma_reg'] = 0.0621
	netcon['clip_norm'] = 1.0
	netcon['update_steps'] = 15
	netcon['N_hid'] = 20
	return netcon


def agent_config():
	agentcon = {}
	agentcon['gamma'] = 0.9
	agentcon['eps0'] = 0.0
	agentcon['epsf'] = 0.0
	agentcon['n_eps'] = 200
	agentcon['minib'] = 100
	agentcon['max_mem'] = 10000
	agentcon['max_exp'] = 500
	return agentcon

N_ep = 10
env = LanderEnvironment()
agent = HARLAgent(agent_config(),network_config(),env,heuristic)

# Train the network for N_ep episodes
R_ep = []
for ep_no in range(N_ep):
	print('Episode: ' + repr(ep_no))
	observation = env.reset()
	done = False
	r = 0
	n_step = 0
	while not done:
		action = agent.action_select(env,observation)
		observation, reward, done, info = env.step(action)
		agent.update_net(observation,reward,done)
		r += reward
		n_step +=1
	R_ep.append(r)
	print('R: ' + repr(r) + ' Length: ' + repr(n_step))

# # Visualise random policy
# for ep_no in range(2):
# 	observation = env.reset()
# 	done = False
# 	while done == False:
# 		action = env.rand_action()
# 		observation, _, done, _ = env.step(action)
# 		env.render()

# Visualise the learned policy
agent.mode_H=False
for ep_no in range(15):
	observation = env.reset()
	done = False
	while done == False:
		action = agent.action_select(env,observation)
		observation, _, done, _ = env.step(action)
		env.render()

agent.sess.close()

# Plot Reward
N_avg=50
R_plot=data_smooth(R_ep,N_avg)
plt.plot(np.arange(len(R_plot))*N_avg,R_plot,'r')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Average Total Discounted Reward', fontsize=12)
plt.show()