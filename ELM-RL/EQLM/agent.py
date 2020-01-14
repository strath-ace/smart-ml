# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

from networks import ELMNet, QNet
from random import sample as _sample
import numpy as np
import numpy.random as rand


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


class EQLMAgent():
	def __init__(self,env,gamma=0.6,eps0=0.9,epsf=0.0,n_eps=400,**kwargs):
		self.state_size = env.state_size
		self.action_size=env.action_size
		self.network = ELMNet(self.state_size, self.action_size,**kwargs)
		self.memory = ReplayMemory(**kwargs)
		
		self.eps = eps0
		self.prev_s=[]
		self.prev_a=[]
	
	def action_select(self,state):
		if rand.random(1)<self.eps:
			action=rand.randint(self.action_size)
		else:
			q_s=self.network.Q_predict(state)
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
		if self.network.prep_state is not None:
			s_prep = self.network.sess.run(self.network.prep_state,
										   feed_dict={self.network.s_input:np.concatenate([self.prev_s,state])})
			self.memory.add([s_prep[0],self.prev_a,reward,s_prep[1],done])
		else:
			self.memory.add([self.prev_s,self.prev_a,reward,state,done])
			
		if len(self.memory)>=self.network.k:
			D_update = self.memory.sample(self.network.k)
		else:
			return
		
		H = np.stack([d[0] for d in D_update])
		# vectorise updates...
		

# # 		with self.sess.as_default():
# 			H = np.stack([self.memory[ind][0] for ind in sample_ind])
# 			Sd = np.stack([self.memory[ind][3] for ind in sample_ind])
# 			Q = self.Qt.eval({self.act:H})
# 			Qd = self.Qt.eval({self.state_input:Sd})
# 			for i, ind in enumerate(sample_ind):
# 				r=self.memory[ind][2]
# 				a=self.memory[ind][1]
# 				if self.memory[ind][4]:
# 					Q[i][a] = r
# 				else:
# 					Q[i][a] = r + self.gamma*np.amax(Qd[i])