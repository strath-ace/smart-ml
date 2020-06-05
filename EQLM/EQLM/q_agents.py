# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2020 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------
"""An agent class which uses Q-Learning to solve problems"""

from . import networks
from random import sample as _sample
import numpy as np
import numpy.random as rand
import pdb


class ReplayMemory(list):
	"""A type of list with methods for limiting size and sampling"""
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


class QAgent(object):
	"""
	A RL agent which uses a Q-Network to select actions

	...

	Attributes
	----------
	state_size : int
		Size of the environment state space
	action_size : int
		Size of the environment action space - must be discrete for Q learning
	nn : network
		Neural network used to approximate Q-function
	nn_target : network
		Target network for stabilising Q-Network updates
	gamma : float
		Discount factor used for Q-learning 
	epsilon : float
		Exploration probability
	d_eps : float
		Decrement added to exploration probability after each episode
	eps_f : float
		Final persistent exploration probability
	f_heur : function
		Selects an action - used to guide initial episodes
	n_heur : int
		Number of episodes where f_heur is used
	target_steps : int
		Number of steps between target network updates
	memory : ReplayMemory
		Stores state transitions for experience replay
	prev_s, prev_a : list
		Placeholder for storing states/actions to add to memory
	ep_count, step_count : int
		Counts the number of episodes/update steps

	Methods
	-------
	action_select(state)
		Returns an action based on the state using an epsilon-greedy policy
	update(state,reward,done)
		Updates the agent's policy and network based on observed info
	"""
	def __init__(self,env,net_type='ELMNet',f_heur=None,n_heur=0,
				 gamma=0.6,eps_i=0.9,eps_f=0.0,n_eps=400,target_steps=50,**kwargs):
		"""
		Parameters
		----------
		env : Environment
			The environment with which the agent interacts
		net_type : {'ELMNet', 'QNet'}, optional
			Specifies which type of Q-Network to use
		f_heur : function, optional
			Heuristic for action selection; takes the state as input and returns an action
		n_heur : int, optional
			Number of episodes to use the heuristic action selection
		gamma : float, optional
			Discount factor for Q-learning
		eps_i : float, optional
			Initial exploration rate
		eps_f : float, optional
			Final exploration rate
		n_eps : int, optional
			Number of episodes for annealing epsilon
		target_steps : int, optional
			Number of steps between target network updates
		**kwargs
			Additional keyword arguments passed to the nn module and `ReplayMemory`
			
		Raises
		------
		ValueError
			If an invalid network type is passed
		"""
		self.state_size = env.state_size
		self.action_size = env.action_size
		try:
			net_module = getattr(networks,net_type)
		except AttributeError:
			raise ValueError('Invalid network type: \'{}\''.format(net_type))

		self.nn = net_module(self.state_size, self.action_size, **kwargs)
		self.nn_target = net_module(self.state_size, self.action_size, is_target=True, **kwargs)
		self.nn_target.assign_params(self.nn.get_params())
		
		self.gamma = gamma
		self.epsilon = eps_i
		self.d_eps = (eps_i-eps_f)/float(n_eps)
		self.eps_f = eps_f
		self.f_heur = f_heur
		self.n_heur = n_heur
		self.target_steps = target_steps
		
		self.memory = ReplayMemory(**kwargs)
		self.prev_s = []
		self.prev_a = []
		self.ep_count = 0
		self.step_count = 0
	
	def action_select(self,state):
		"""Returns an action based on the state using an epsilon-greedy policy
		
		Parameters
		----------
		state : array-like
			The environment state at the current timestep
		
		Returns
		-------
		action : int
			Action selected by the policy
		"""
		if self.ep_count<self.n_heur and self.f_heur is not None:
			action = self.f_heur(state)
		elif rand.random(1)<self.epsilon:
			action=rand.randint(self.action_size)
		else:
			q_s=self.nn.Q_predict(state)
			action=np.argmax(q_s)
		self.prev_s=state
		self.prev_a=action
		return action
	
	def update(self,state,reward,done):
		"""Updates the agent's policy and Q network based on observed info
		
		Parameters
		----------
		state : list
			Most recent observed state to be added to memory
		reward : float
			Observed reward
		done : bool
			Indicates a terminal state
		"""
		if done:
			self.ep_count += 1
			self.epsilon = np.max([self.epsilon-self.d_eps,self.eps_f])

		if self.nn.prep_state is not None:
			s_prep = self.nn.sess.run(self.nn.prep_state,
										   feed_dict={self.nn.s_input:np.concatenate([self.prev_s,state])})
			self.memory.add([s_prep[0],self.prev_a,reward,s_prep[1],done])
		else:
			self.memory.add([self.prev_s.reshape(-1),self.prev_a,reward,state.reshape(-1),done])
			
		if len(self.memory)<self.nn.k:
			return
		
		D_update = self.memory.sample(self.nn.k)
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
		Q[range(Q.shape[0]),a] = r
		Q[indt,a[indt]] += self.gamma*np.max(Qd,1)
		self.nn.update(s,Q)
		
		self.step_count += 1
		if self.step_count >= self.target_steps:
			self.nn_target.assign_params(self.nn.get_params())
			self.step_count = 0
