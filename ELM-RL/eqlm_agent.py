# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import tensorflow as tf
import numpy as np
import random
import math
from environment import Environment
from operator import itemgetter
import pdb

class EQLMAgent():
	def __init__(self,agent_config,network_config,env):

		for key in agent_config.keys():
			setattr(self, key, agent_config[key])
		for key in network_config.keys():
			setattr(self, key, network_config[key])
		self.eps = self.eps0

		self.state_size = env.state_size
		self.action_size = env.action_size

		# Build network
		self.state_input = tf.placeholder(shape=[None,self.state_size],dtype=tf.float32)
		self.w_in = tf.Variable(tf.random.uniform([self.state_size,self.N_hid],0,self.init_mag))
		self.b_in = tf.Variable(tf.random.uniform([1,self.N_hid],0,self.init_mag/10.0))
		self.W = tf.Variable(tf.random.uniform([self.N_hid,self.action_size],0,self.init_mag))
		try:
			act_fn = getattr(tf,self.activation)
		except AttributeError:
			act_fn = tf.tanh
		self.act = act_fn(tf.add(tf.matmul(self.state_input,self.w_in),self.b_in), name=None)
		self.Q_est = tf.matmul(self.act,self.W)

		# Matrices for updating
		self.H = tf.placeholder(shape=[self.minib,self.N_hid],dtype=tf.float32)
		self.T = tf.placeholder(shape=[self.minib,self.action_size],dtype=tf.float32)
		self.H_trans = tf.transpose(self.H)
		self.A_inv = tf.Variable(tf.random_uniform([self.N_hid,self.N_hid],0,1))

		# Initialisation
		A_t1 = tf.add(tf.scalar_mul(1/self.gamma_reg,tf.eye(self.N_hid)),
			tf.matmul(self.H_trans,self.H))
		A_t1_inv = tf.linalg.inv(A_t1)
		W_t1 = tf.matmul(A_t1_inv,tf.matmul(self.H_trans,self.T))
		self.W_init = self.W.assign(W_t1)
		self.A_init = self.A_inv.assign(A_t1_inv)

		# Updating
		K1 = tf.add(tf.matmul(self.H,tf.matmul(self.A_inv,self.H_trans)),tf.eye(self.minib)) #1
		
# 		K_t = tf.subtract(tf.eye(self.N_hid),
# 				tf.matmul(self.A_inv,tf.matmul(self.H_trans,tf.matmul(tf.linalg.inv(K1),self.H)))) #2
# 		K_t = tf.subtract(tf.eye(self.N_hid),
# 				tf.matmul(tf.matmul(self.A_inv,self.H_trans),tf.matmul(tf.linalg.inv(K1),self.H))) #2
		
# 		W_new = tf.add(tf.matmul(K_t,self.W),
# 				tf.matmul(tf.matmul(K_t,self.A_inv),tf.matmul(self.H_trans,self.T))) #3
# 		W_new = tf.add(tf.matmul(K_t,self.W),
# 				tf.matmul(K_t,tf.matmul(self.A_inv,tf.matmul(self.H_trans,self.T)))) #3
		
		if self.minib >= self.N_hid:
			K_t = tf.subtract(tf.eye(self.N_hid),
				tf.matmul(self.A_inv,tf.matmul(self.H_trans,tf.matmul(tf.linalg.inv(K1),self.H)))) #2
		else:
			K_t = tf.subtract(tf.eye(self.N_hid),
				tf.matmul(tf.matmul(self.A_inv,self.H_trans),tf.matmul(tf.linalg.inv(K1),self.H))) #2
		
		if self.minib >= self.action_size:
			W_new = tf.add(tf.matmul(K_t,self.W),
				tf.matmul(K_t,tf.matmul(self.A_inv,tf.matmul(self.H_trans,self.T)))) #3
		else:
			W_new = tf.add(tf.matmul(K_t,self.W),
				tf.matmul(tf.matmul(K_t,self.A_inv),tf.matmul(self.H_trans,self.T))) #3
			
		print(self.minib)
		print(self.N_hid)
		print(self.action_size)
						   
		A_new = tf.matmul(K_t,self.A_inv)
		self.W_update = self.W.assign(W_new)
		self.A_update = self.A_inv.assign(A_new)

		# Target network update
		self.Wt = tf.Variable(tf.random_uniform([self.N_hid,self.action_size],0,self.init_mag))
		self.Qt = tf.matmul(self.act,self.Wt)
		self.Wt_assign = self.Wt.assign(self.W)

		# Start tensorflow session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		# Memory initialisation
		self.prev_s=[]
		self.prev_a=[]
		self.prev_h=[]
		self.memory=[]
		self.memory1=[]
		self.step_count=0
		self.ep_no=0
		self.first=True

	def action_select(self,env,state):

		q_s, h_s = self.sess.run((self.Q_est, self.act),feed_dict={self.state_input:state.reshape(1,-1)})
		if np.random.random(1)<self.eps:
			action=env.rand_action()
		else:
			action=np.argmax(q_s)
		self.prev_s=state
		self.prev_a=action
		self.prev_h=h_s[0]
		return action

	def update_net(self,state,reward,done):

		if done:
			self.ep_no+=1
			# Update exploration probability
			if self.ep_no<self.n_eps:
				self.eps=float(self.eps0)-self.ep_no*(float(self.eps0)-float(self.epsf))/float(self.n_eps)
			else:
				self.eps=self.epsf

		# Update memory
		self.memory.append([self.prev_h, self.prev_a, reward, state, done])
		if len(self.memory)>self.max_mem:
			del self.memory[0]

		# Select data from memory
		if len(self.memory)>self.minib & self.minib>1:
			sample_ind=random.sample(range(1,len(self.memory)),self.minib)
		elif self.minib==1:
			sample_ind=[len(self.memory)-1]
		else:
			return

		with self.sess.as_default():
			H = np.stack([self.memory[ind][0] for ind in sample_ind])
			Sd = np.stack([self.memory[ind][3] for ind in sample_ind])
			Q = self.Qt.eval({self.act:H})
			Qd = self.Qt.eval({self.state_input:Sd})
			for i, ind in enumerate(sample_ind):
				r=self.memory[ind][2]
				a=self.memory[ind][1]
				if self.memory[ind][4]:
					Q[i][a] = r
				else:
					Q[i][a] = r + self.gamma*np.amax(Qd[i])

		# Run updates
		if self.first:
			self.sess.run((self.W_init, self.A_init),feed_dict={self.H:H,self.T:Q})
			self.first = False
		else:
			self.sess.run((self.W_update, self.A_update),feed_dict={self.H:H,self.T:Q})

		# Target network update
		self.step_count += 1
		if self.step_count >= self.update_steps:
			self.sess.run(self.Wt_assign)
			self.step_count=0

################################################################################################

################################################################################################

class EQLMAgent2():
	def __init__(self,agent_config,network_config,env):

		for key in agent_config.keys():
			setattr(self, key, agent_config[key])
		for key in network_config.keys():
			setattr(self, key, network_config[key])
		self.eps = self.eps0

		self.state_size = env.state_size
		self.action_size = env.action_size

		# Build network
		self.state_input = tf.placeholder(shape=[None,self.state_size],dtype=tf.float32)
		self.w_in = tf.Variable(tf.random.uniform([self.state_size,self.N_hid],0,self.init_mag))
		self.b_in = tf.Variable(tf.random.uniform([1,self.N_hid],0,self.init_mag))
		self.W = tf.Variable(tf.random.uniform([self.N_hid,self.action_size],0,self.init_mag), use_resource=True)
		try:
			act_fn = getattr(tf,self.activation)
		except AttributeError:
			act_fn = tf.tanh
		self.act = act_fn(tf.matmul(self.state_input,tf.add(self.w_in,self.b_in)), name=None)
		self.Q_est = tf.matmul(self.act,self.W)

		# Matrices for updating
		self.H = tf.placeholder(shape=[self.minib,self.N_hid],dtype=tf.float32)
		self.T = tf.placeholder(shape=[self.minib,self.action_size],dtype=tf.float32)
		self.H_trans = tf.transpose(self.H)
		self.A_inv = tf.Variable(tf.random.uniform([self.N_hid,self.N_hid],0,1), use_resource=True)

		# Initialisation
		A_t1 = tf.add(tf.scalar_mul(1/self.gamma_reg,tf.eye(self.N_hid)),
			tf.matmul(self.H_trans,self.H))
		A_t1_inv = tf.linalg.inv(A_t1)
		W_t1 = tf.matmul(A_t1_inv,tf.matmul(self.H_trans,self.T))
		self.W_init = self.W.assign(W_t1)
		self.A_init = self.A_inv.assign(A_t1_inv)

		# Updating
		K1 = tf.add(tf.matmul(self.H,tf.matmul(self.A_inv,self.H_trans)),tf.eye(self.minib))
		# need to optimise
		K_t = tf.subtract(tf.eye(self.N_hid),
			tf.matmul(self.A_inv,tf.matmul(self.H_trans,tf.matmul(tf.linalg.inv(K1),self.H))))
		W_new = tf.add(tf.matmul(K_t,self.W),
			tf.matmul(tf.matmul(K_t,self.A_inv),tf.matmul(self.H_trans,self.T)))
		A_new = tf.matmul(K_t,self.A_inv)
		self.W_update = self.W.assign(W_new)
		self.A_update = self.A_inv.assign(A_new)

		# Target network update
		self.Wt = tf.Variable(tf.random_uniform([self.N_hid,self.action_size],0,self.init_mag))
		self.Qt = tf.matmul(self.act,self.Wt)
		self.Wt_assign = self.Wt.assign(self.W)

		# Start tensorflow session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		# Memory initialisation
		self.prev_s=[]
		self.prev_a=[]
		self.prev_h=[]
		self.memory=[]
		self.memory1=[]
		self.step_count=0
		self.ep_no=0
		self.first=True

	def action_select(self,env,state):

		q_s, h_s = self.sess.run((self.Q_est, self.act),feed_dict={self.state_input:state.reshape(1,-1)})
		if np.random.random(1)<self.eps:
			action=env.rand_action()
		else:
			action=np.argmax(q_s)
		self.prev_s=state
		self.prev_a=action
		self.prev_h=h_s[0]
		return action

	def update_net(self,state,reward,done):
		
		@tf.function
		def _net_init(H_i,T_i):
			A_t1 = tf.add(tf.scalar_mul(1/self.gamma_reg,tf.eye(self.N_hid)),tf.matmul(tf.transpose(H_i),H_i))
			A_t1_inv = tf.linalg.inv(A_t1)
			W_t1 = tf.matmul(A_t1_inv,tf.matmul(tf.transpose(H_i),T_i))
			self.W.assign(W_t1)
			self.A_inv.assign(A_t1_inv)
		
		@tf.function
		def _net_update(H_i,T_i):
			K1 = tf.add(tf.matmul(self.H,tf.matmul(self.A_inv,self.H_trans)),tf.eye(self.minib))
			# need to optimise
			K_t = tf.subtract(tf.eye(self.N_hid),
				tf.matmul(self.A_inv,tf.matmul(self.H_trans,tf.matmul(tf.linalg.inv(K1),self.H))))
			W_new = tf.add(tf.matmul(K_t,self.W),
				tf.matmul(tf.matmul(K_t,self.A_inv),tf.matmul(self.H_trans,self.T)))
			A_new = tf.matmul(K_t,self.A_inv)
			self.W.assign(W_new)
			self.A_inv.assign(A_new)
			return W_new, A_new

		if done:
			self.ep_no+=1
			# Update exploration probability
			if self.ep_no<self.n_eps:
				self.eps=float(self.eps0)-self.ep_no*(float(self.eps0)-float(self.epsf))/float(self.n_eps)
			else:
				self.eps=self.epsf

		# Update memory
		self.memory.append([self.prev_h, self.prev_a, reward, state, done])
		if len(self.memory)>self.max_mem:
			del self.memory[0]

		# Select data from memory
		if len(self.memory)>self.minib & self.minib>1:
			sample_ind=random.sample(range(1,len(self.memory)),self.minib)
		elif self.minib==1:
			sample_ind=[len(self.memory)-1]
		else:
			return

		with self.sess.as_default():
			H = np.stack([self.memory[ind][0] for ind in sample_ind])
			Sd = np.stack([self.memory[ind][3] for ind in sample_ind])
			Q = self.Qt.eval({self.act:H})
			Qd = self.Qt.eval({self.state_input:Sd})
			for i, ind in enumerate(sample_ind):
				r=self.memory[ind][2]
				a=self.memory[ind][1]
				if self.memory[ind][4]:
					Q[i][a] = r
				else:
					Q[i][a] = r + self.gamma*np.amax(Qd[i])

		# Run updates
		if self.first:
			with self.sess.as_default():
				print('W')
				print(self.sess.run(self.W))
				print('A')
				print(self.sess.run(self.A_inv))
			self.sess.run((self.W_init, self.A_init),feed_dict={self.H:H,self.T:Q})
			with self.sess.as_default():
				print('W')
				print(self.sess.run(self.W))
				print('A')
				print(self.sess.run(self.A_inv))
# 			_net_init(H,Q)
# 			with self.sess.as_default():
# 				print('W')
# 				print(self.sess.run(self.W))
# 				print('A')
# 				print(self.sess.run(self.A_inv))
			self.first = False
		else:
			self.sess.run((self.W_update, self.A_update),feed_dict={self.H:H,self.T:Q})
# 			w2, a2 = _net_update(H,Q)
			

		# Target network update
		self.step_count += 1
		if self.step_count >= self.update_steps:
			self.sess.run(self.Wt_assign)
			self.step_count=0
			with self.sess.as_default():
				print('W')
				print(self.sess.run(self.W))
				print('A')
				print(self.sess.run(self.A_inv))
