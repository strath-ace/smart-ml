# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
import sys
import gym

class Environment():
	def __init__(self,env_name='CartPole-v0'):
		try:
			self.gym_env = gym.make(env_name)
			try:
				self.state_size = self.gym_env.observation_space.n
			except AttributeError:
				self.state_size = int(np.prod(np.shape(self.gym_env.observation_space)))
			self.action_size = self.gym_env.action_space.n
		except gym.error.Error:
			raise ValueError(env_name + ' environment name does not exist')
		except AttributeError:
			raise ValueError(env_name + ' environment action space is incompatible')

	def render(self):
		self.gym_env.render()
		
	def rand_action(self):
		return self.gym_env.action_space.sample()
	
	def reset(self):
		s = self.gym_env.reset()
		if isinstance(s,int):
			state = np.zeros(self.state_size)
			state[s] = 1
			s = state
		return s.reshape(1,-1)
	
	def step(self,a,render=False):
		s,r,d,info = self.gym_env.step(a)
		if isinstance(s,int):
			state = np.zeros(self.state_size)
			state[s] = 1
			s = state
		if render:
			self.gym_env.render()
		return s.reshape(1,-1),r,d,info
