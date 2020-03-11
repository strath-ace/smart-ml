# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
from tqdm import trange
import pickle
import pdb

class Reward(list):
	def __init__(self,**kwargs):
		list.__init__(self,[])
	def smooth(self, n_avg=100):
		x = np.arange(n_avg,len(self)+1,n_avg)
		R = [np.mean(self[ind-n_avg:ind]) for ind in x]
		return x, R

def do_run(agent, env, N_ep, save_name=None, show_progress=False):
	R_ep = Reward()
	steps=[]
	if show_progress:
		t = trange(N_ep, desc='bar_desc', leave=True)
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
	return R_ep, agent, env

def agent_demo(agent, env, N_ep):
	R_ep = Reward()
	for ep_no in range(N_ep):
		s = env.reset()
		done = False
		Rt = 0
		while not done:
			a = agent.action_select(s)
			s, r, done, _ = env.step(a)
			env.render()
			Rt += r
		R_ep.append(Rt)
	return R_ep

def heuristic_demo(H, env, N_ep, show=False):
	R_ep = Reward()
	for ep_no in range(N_ep):
		s = env.reset()
		done = False
		Rt = 0
		while not done:
			a = H(s[0])
			s, r, done, _ = env.step(a)
			if show:
				env.render()
			Rt += r
		R_ep.append(Rt)
	return R_ep