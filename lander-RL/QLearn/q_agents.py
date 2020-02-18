# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

from . import networks
from random import sample as _sample
import numpy as np
import numpy.random as rand
import pdb


class ReplayMemory(list):
	def __init__(self,memory_size=None,**kwargs):
		list.__init__(self,[])
		self.max_len = memory_size
	def add(self, list_add):
		self.append(list_add)
		if self.max_len:
			while len(self)>self.max_len:
				self.remove(self[0])
	def sample(self, n):
		return _sample(self,n)


class QAgent():
	def __init__(self,env,net_type='ELMNet',gamma=0.6,eps0=0.9,epsf=0.0,n_eps=400,update_steps=50,**kwargs):
		self.state_size = env.state_size
		self.action_size=env.action_size
		try:
			net_module = getattr(networks,net_type)
		except AttributeError:
			raise ValueError('Invalid network type: \'{}\''.format(net_type))

		self.nn = net_module(self.state_size, self.action_size,**kwargs)
		self.nn_target = net_module(self.state_size, self.action_size,is_target=True,**kwargs)

		self.memory = ReplayMemory(**kwargs)
		
		self.gamma = gamma
		self.eps = eps0
		self.eps0 = eps0
		self.epsf = epsf
		self.n_eps = n_eps
		
		self.prev_s = []
		self.prev_a = []
		self.ep_no = 0
		self.step_count = 0
		self.C = update_steps
	
	def action_select(self,state):
		if rand.random(1)<self.eps:
			action=rand.randint(self.action_size)
		else:
			q_s=self.nn.Q_predict(state)
			action=np.argmax(q_s)
		self.prev_s=state
		self.prev_a=action
		return action
	
	def update(self,state,reward,done):
		if done:
			self.ep_no+=1
			if self.ep_no<self.n_eps:
				self.eps=float(self.eps0)-self.ep_no*(float(self.eps0)-float(self.epsf))/float(self.n_eps)
			else:
				self.eps=self.epsf
		# Update memory
		if self.nn.prep_state is not None:
			s_prep = self.nn.sess.run(self.nn.prep_state,
										   feed_dict={self.nn.s_input:np.concatenate([self.prev_s,state])})
			self.memory.add([s_prep[0],self.prev_a,reward,s_prep[1],done])
		else:
			self.memory.add([self.prev_s.reshape(-1),self.prev_a,reward,state.reshape(-1),done])
			
		if len(self.memory)>=self.nn.k:
			D_update = self.memory.sample(self.nn.k)
		else:
			return
		
		s, a, r, Sd, St = (np.stack([d[0] for d in D_update]),
						   np.array([d[1] for d in D_update]),
						   np.array([d[2] for d in D_update]),
						   np.stack([d[3] for d in D_update]),
						   np.invert(np.array([d[4] for d in D_update])))
		indt = np.where(St)[0]
		if self.nn.prep_state is not None:
			Q = self.nn.Q_predict(s_prep=s)
			Qd = self.nn_target.Q_predict(s_prep=Sd[St])
		else:
			Q = self.nn.Q_predict(s=s)
			Qd = self.nn_target.Q_predict(s=Sd[St])
		Q[np.arange(Q.shape[0]),a] = r
		Q[indt,a[indt]] += self.gamma*np.max(Qd,1)
		self.nn.update(s,Q)
		
		self.step_count+=1
		if self.step_count >= self.C:
			self.nn_target.assign_params(self.nn.get_params())
			self.step_count=0


class HeuristicAgent():
	def __init__(self,env,f_heur,net_type='QNet',gamma=0.99,eps0=0.9,epsf=0.0,n_eps=400,N_heur=10,**kwargs):
		"""
		Same as the QAgent and uses a heuristic for initial episodes
		"""
		self.state_size = env.state_size
		self.action_size = env.action_size

		try:
			net_module = getattr(networks,net_type)
		except AttributeError:
			raise ValueError('Invalid network type: \'{}\''.format(net_type))

		self.nn = net_module(self.state_size, self.action_size,**kwargs)

		self.memory = ReplayMemory(**kwargs)

		self.gamma = gamma
		self.eps = eps0
		self.eps0 = eps0
		self.epsf = epsf
		self.n_eps = n_eps
		self.N_heur = N_heur
		self.f_heur = f_heur

		self.prev_s = []
		self.prev_a = []
		self.ep_no = 0

	def action_select(self,state):
		if self.ep_no<self.N_heur:
			action=self.f_heur(state[0])
		elif rand.random(1)<self.eps:
			action=rand.randint(self.action_size)
		else:
			q_s=self.nn.Q_predict(state)
			action=np.argmax(q_s)
		self.prev_s=state
		self.prev_a=action
		return action

	def update(self,state,reward,done):
		if done:
			self.ep_no+=1
			if self.ep_no<self.n_eps:
				self.eps=float(self.eps0)-self.ep_no*(float(self.eps0)-float(self.epsf))/float(self.n_eps)
			else:
				self.eps=self.epsf
		# Update memory
		if self.nn.prep_state is not None:
			s_prep = self.nn.sess.run(self.nn.prep_state,
										   feed_dict={self.nn.s_input:np.concatenate([self.prev_s,state])})
			self.memory.add([s_prep[0],self.prev_a,reward,s_prep[1],done])
		else:
			self.memory.add([self.prev_s.reshape(-1),self.prev_a,reward,state.reshape(-1),done])

		if len(self.memory)>=self.nn.k:
			D_update = self.memory.sample(self.nn.k)
		else:
			return

		s = np.stack([d[0] for d in D_update])
		a = np.array([d[1] for d in D_update])
		r = np.array([d[2] for d in D_update])
		Sd = np.stack([d[3] for d in D_update])
		St = np.invert(np.array([d[4] for d in D_update]))
		indt = np.where(St)[0]
		if self.nn.prep_state is not None:
			Q = self.nn.Q_predict(s_prep=s)
		else:
			Q = self.nn.Q_predict(s=s)
		Qd = self.nn.Q_target(Sd[St])
		Q[np.arange(Q.shape[0]),a] = r
		Q[indt,a[indt]] += self.gamma*np.max(Qd,1)
		self.nn.update(s,Q)