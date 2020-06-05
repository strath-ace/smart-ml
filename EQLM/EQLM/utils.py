# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2020 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------
"""Methods for training, testing, and visualising agents"""

import numpy as np
from tqdm.notebook import trange
from .q_agents import ReplayMemory, QAgent
import pickle


def train_agent(agent, env, N_ep, save_name=None, show_progress=False):
	"""
	Train an agent for a given number of episodes in a given environment
	...
	
	Parameters
	----------
	agent : EQLM.QAgent
		QLearning agent
	env : gym.Wrapper
		Envrionment on which the agent is trained
	N_ep : int
		Number of episodes
	save_name : str, optional
		Name of file to save results, by default does not save
	show_progress : bool, optional
		Displays a tqdm notebook progress bar
		
	Returns
	-------
	R_ep : list
		Cumulative reward for each episode
	steps : list
		Number of environment steps in each episode
	agent : EQLM.QAgent
		The trained agent
	"""
	R_ep = []
	steps=[]
	if show_progress:
		t = trange(N_ep, desc='bar_desc', leave=True)
	else:
		t = range(N_ep)
	for ep_no in t:
		s = env.reset()
		done = False
		Rt = 0
		n_step = 0
		while not done:
			a = agent.action_select(s)
			s, r, done, _ = env.step(a)
			agent.update(s,r,done)
			Rt += r
			n_step +=1
		R_ep.append(Rt)
		steps.append(n_step)
		if show_progress:
			if ep_no>10:
				t.set_description('R: {} Step: {}'.format(np.mean(R_ep[-10:]).round(1),n_step))
				t.refresh()
			else:
				t.set_description('R: {} Step: {}'.format(np.mean(R_ep).round(1),n_step))
				t.refresh()
		if save_name:
			data = {'params':agent.nn.get_params(),'R':R_ep,'step':steps}
			pickle.dump(data, open(save_name,'wb'))
	return R_ep, steps, agent

def agent_demo(agent, env, N_ep, render=False):
	"""Demonstrate a learned agent policy"""
	R_ep = Reward()
	steps=[]
	for ep_no in range(N_ep):
		s = env.reset()
		done = False
		Rt = 0
		n_step = 0
		while not done:
			a = agent.action_select(s)
			s, r, done, _ = env.step(a, render=render)
			Rt += r
			n_step +=1
		R_ep.append(Rt)
		steps.append(n_step)
	return R_ep, steps

def heuristic_demo(H, env, N_ep, render=False):
	"""Demonstrate a heuristic where a=H(s)"""
	R_ep = Reward()
	steps=[]
	for ep_no in trange(N_ep):
		s = env.reset()
		done = False
		Rt = 0
		n_step = 0
		while not done:
			a = H(s[0])
			s, r, done, _ = env.step(a, render=render)
			Rt += r
			n_step +=1
		R_ep.append(Rt)
		steps.append(n_step)
	return R_ep, steps

def save_results(fname, R, agent, hyper_params=None):
	"""Save results to a file"""
	try:
		results = pickle.load(open(fname,'rb'))
	except FileNotFoundError:
		results = {'R':[], 'agents':[], 'params':hyper_params}
	results['R'].append(R)
	results['agents'].append(agent.nn.get_params())
	pickle.dump(results, open(fname,'wb'))

def data_smooth(data,n_avg):
	"""For plotting learning curves"""
	x_vec = np.arange(0,len(data)+1,n_avg)
	data_avg = [0]
	for x in x_vec[1:]:
		data_avg.append(np.mean(data[x-n_avg:x]))
	return x_vec, data_avg
