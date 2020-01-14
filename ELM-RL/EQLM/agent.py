# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

from networks import ELMNet, QNet
from random import sample as _sample


class ReplayMemory(list):
	def __init__(self,*args,memory_size=None):
		list.__init__(self,[args])
		self.max_len = memory_size
	def add(self, list_add):
		self.append(list_add)
		if self.max_len:
			while len(self)>self.max_len:
				self.remove(self[0])
	def sample(self, n):
		return _sample(self,n,False)


class EQLMAgent():
	def __init__(self,env,gamma=0.6,eps0=0.9,epsf=0.0,n_eps=400,**kwargs):
		self.state_size = env.state_size
		self.action_size=env.action_size
		self.network = ELMNet(self.state_size, self.action_size,**kwargs)
		self.memory = ReplayMemory(**kwargs)
		
		self.eps = self.eps0
		self.prev_s=[]
		self.prev_a=[]
		self.memory=[]
	
	def action_select(self,state):
		if np.random.random(1)<self.eps:
			action=env.rand_action()
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
# 			# Update memory (terminal state)
# 			with self.sess.as_default():
# 				self.memory.append([self.act.eval(feed_dict={self.state_input:self.prev_s.reshape(1,-1)}).tolist(),
# 					self.prev_a,reward,
# 					[]])
# 		else:
# 			# Update memory (non-terminal state)
# 			with self.sess.as_default():
# 				self.memory.append([self.act.eval(feed_dict={self.state_input:self.prev_s.reshape(1,-1)}).tolist(),
# 					self.prev_a,reward,
# 					self.act.eval(feed_dict={self.state_input:state.reshape(1,-1)}).tolist()])

# 		if len(self.memory)>self.max_mem:
# 			del self.memory[0]

# 		# Select data from memory
# 		if len(self.memory)>self.minib & self.minib>1:
# 			sample_ind=random.sample(range(1,len(self.memory)),self.minib)
# 		elif self.minib==1:
# 			sample_ind=[len(self.memory)-1]
# 		else:
# 			return

# 		# Create matrices of pre-processed states and targets
# 		H_update=[]
# 		q_target=[]
# 		with self.sess.as_default():
# 			for ind in sample_ind:
# 				h=self.memory[ind][0]
# 				r=self.memory[ind][2]
# 				a=self.memory[ind][1]
# 				q_s=self.Qt.eval(feed_dict={self.act:h})
# 				if len(self.memory[ind][3])>0:
# 					q_dash=self.Qt.eval({self.act:self.memory[ind][3]})
# 					q_s[0][a]=r+self.gamma*np.amax(q_dash)
# 				else:
# 					q_s[0][a]=r
# 				H_update+=h
# 				q_target+=q_s.reshape(1,-1).tolist()