# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
import random
import math
from environment import Environment
from operator import itemgetter
import pdb

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import load_model

class DQNAgentKeras():
	def __init__(self,agent_config,network_config,env):

		for key in agent_config.keys():
			setattr(self, key, agent_config[key])
		for key in network_config.keys():
			setattr(self, key, network_config[key])
		self.eps = self.eps0

		self.state_size = env.state_size
		self.action_size = env.action_size
		
		self.model = self.initialize_model()

		# Memory initialisation
		self.prev_s=[]
		self.prev_a=[]
		self.memory=[]
		self.step_count=0
		self.ep_no=0

	def initialize_model(self):
		model = Sequential()
		model.add(Dense(512, input_dim=self.state_size, activation=relu))
		model.add(Dense(256, activation=relu))
		model.add(Dense(self.action_size, activation=linear))

		# Compile the model
		model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.alpha))
		print(model.summary())
		return model

	def action_select(self,env,state):

		if np.random.random(1)<self.eps:
			action=env.rand_action()
		else:
			q_s=self.model.predict(np.reshape(state,[1,self.state_size]))
			action=np.argmax(q_s)
		self.prev_s=state
		self.prev_a=action
		return action

	def update_net(self,state,reward,done):

		# Update epsilon
		if done:
			self.ep_no+=1
			if self.ep_no<self.n_eps:
				self.eps=float(self.eps0)-self.ep_no*(float(self.eps0)-float(self.epsf))/float(self.n_eps)
			else:
				self.eps=self.epsf

		# Update memory
		self.memory.append([self.prev_s,self.prev_a,reward,state,done])
		if len(self.memory)>self.max_mem:
			del self.memory[0]

		# Select data from memory
		if len(self.memory)>self.minib & self.minib>1:
			sample_ind=random.sample(range(1,len(self.memory)),self.minib)
		elif self.minib==1:
			sample_ind=[len(self.memory)-1]
		else:
			sample_ind=range(len(self.memory))

		# Update network
		S1 = np.stack([self.memory[ind][0] for ind in sample_ind])
		Sd1 = np.stack([self.memory[ind][3] for ind in sample_ind])
		q_s1 = self.model.predict(S1)
		q_d1 = self.model.predict(Sd1)
		for i, ind in enumerate(sample_ind):
			r=self.memory[ind][2]
			a=self.memory[ind][1]
			if self.memory[ind][4]:
				q_s1[i][a] = r
			else:
				q_s1[i][a] = r + self.gamma*np.amax(q_d1[i])
		self.model.fit(S1,q_s1, epochs=1,verbose=0)
            

class DQNAgentDemo():
	def __init__(self,agent_config,network_config,env,demo_transitions):

		for key in agent_config.keys():
			setattr(self, key, agent_config[key])
		for key in network_config.keys():
			setattr(self, key, network_config[key])
		self.eps = self.eps0

		self.state_size = env.state_size
		self.action_size = env.action_size
		
		################################################################
		
		################################################################

		# Build network
		self.state_input = tf.placeholder(shape=[None,self.state_size],dtype=tf.float32)
		
		self.weights = {'w1':tf.Variable(tf.random_normal([self.state_size,self.N_hid_1]),trainable=True),
					   'w2':tf.Variable(tf.random_normal([self.N_hid_1,self.N_hid_2]),trainable=True),
					   'out':tf.Variable(tf.random_normal([self.N_hid_2,self.action_size]),trainable=True)}
		self.biases = {'b1':tf.Variable(tf.random_normal([self.N_hid_1]),trainable=True),
					   'b2':tf.Variable(tf.random_normal([self.N_hid_2]),trainable=True),
					   'out':tf.Variable(tf.random_normal([self.action_size]),trainable=True)}
		param_names = ['w1', 'w2', 'wout', 'b1', 'b2', 'bout']
		
# 		try:
# 			act_fn = getattr(tf,self.activation)
# 		except AttributeError:
# 			act_fn = tf.tanh
		act_fn = tf.nn.relu
		
		
# 		self.w_in = tf.Variable(tf.random_uniform([self.state_size,self.N_hid],0,self.init_mag))
# 		self.b_in = tf.Variable(tf.random_uniform([1,self.N_hid],0,0))
# 		self.W = tf.Variable(tf.random_uniform([self.N_hid,self.action_size],0,self.init_mag))
# 		self.W1 = tf.Variable(tf.random_uniform([self.N_hid,self.action_size],0,self.init_mag))
# 		self.act = act_fn(tf.add(tf.matmul(self.state_input,self.w_in),self.b_in), name=None)
		
		layer_1 = act_fn(tf.add(tf.matmul(self.state_input,self.weights['w1']),self.biases['b1']))
		layer_2 = act_fn(tf.add(tf.matmul(layer_1,self.weights['w2']),self.biases['b2']))
		output = tf.add(tf.matmul(layer_2,self.weights['out']),self.biases['out'])
		
# 		self.Q_est = tf.matmul(self.act,self.W)
		self.Q_est = output

		# Updating
		self.nextQ = tf.placeholder(shape=[None,self.action_size],dtype=tf.float32)
		loss = tf.reduce_sum(tf.square(self.nextQ - self.Q_est))
		trainer = tf.train.GradientDescentOptimizer(self.alpha)
		
# 		if isinstance(self.clip_norm, float):
# 			grads = trainer.compute_gradients(loss,[self.W,self.w_in,self.b_in])
# 			cap_grads = [(tf.clip_by_norm(grad, self.clip_norm), var) for grad, var in grads]
# 			self.updateModel = trainer.apply_gradients(cap_grads)
# 		else:
# 			self.updateModel = trainer.minimize(loss,var_list=[self.W,self.w_in,self.b_in])

		if isinstance(self.clip_norm, float):
			grads = trainer.compute_gradients(loss)
			cap_grads = [(tf.clip_by_norm(grad, self.clip_norm), var) for grad, var in grads]
			self.updateModel = trainer.apply_gradients(cap_grads)
		else:
			self.updateModel = trainer.minimize(loss)
			
		# Target network
		self.target_weights = {'w1':tf.Variable(tf.random_normal([self.state_size,self.N_hid_1])),
					   'w2':tf.Variable(tf.random_normal([self.N_hid_1,self.N_hid_2])),
					   'out':tf.Variable(tf.random_normal([self.N_hid_2,self.action_size]))}
		self.target_biases = {'b1':tf.Variable(tf.random_normal([self.N_hid_1])),
					   'b2':tf.Variable(tf.random_normal([self.N_hid_2])),
					   'out':tf.Variable(tf.random_normal([self.action_size]))}
		
# 		self.wt = tf.Variable(tf.random_uniform([self.state_size,self.N_hid],0,0.1))
# 		self.bt = tf.Variable(tf.random_uniform([1,self.N_hid],0,0.01))
# 		self.Wt = tf.Variable(tf.random_uniform([self.N_hid,self.action_size],0,0.1))
# 		self.actt = act_fn(tf.add(tf.matmul(self.state_input,self.wt),self.bt), name=None)
		
		target_layer_1 = act_fn(tf.add(tf.matmul(self.state_input,self.target_weights['w1']),self.target_biases['b1']))
		target_layer_2 = act_fn(tf.add(tf.matmul(layer_1,self.target_weights['w2']),self.target_biases['b2']))
		target_output = tf.add(tf.matmul(layer_2,self.target_weights['out']),self.target_biases['out'])
		
# 		self.Qt = tf.matmul(self.actt,self.Wt)
		self.Qt = target_output
	
		def target_assign():
			with self.sess.as_default():
				for w in self.weights.keys():
					self.target_weights[w].assign(self.weights[w]).eval()
				for b in self.biases.keys():
					self.target_biases[b].assign(self.biases[b]).eval()
		
# 		self.wt_assign = self.wt.assign(self.w_in)
# 		self.bt_assign = self.bt.assign(self.b_in)
# 		self.Wt_assign = self.Wt.assign(self.W)
		
		self.update_target = target_assign
		
		################################################################
		
		################################################################

		# Start tensorflow session
		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())
		
		self.update_target()

		# Memory initialisation
		self.prev_s=[]
		self.prev_a=[]
		self.memory=[]
		self.memory_demo = demo_transitions
		self.step_count=0
		self.ep_no=0

	def action_select(self,env,state):

		if np.random.random(1)<self.eps:
			action=env.rand_action()
		else:
			q_s=self.sess.run(self.Q_est,feed_dict={self.state_input:state.reshape(1,-1)})
			action=np.argmax(q_s)
		self.prev_s=state
		self.prev_a=action
		return action

	def update_net(self,state,reward,done):

		# Update epsilon
		if done:
			self.ep_no+=1
			if self.ep_no<self.n_eps:
				self.eps=float(self.eps0)-self.ep_no*(float(self.eps0)-float(self.epsf))/float(self.n_eps)
			else:
				self.eps=self.epsf

		# Update memory
		self.memory.append([self.prev_s,self.prev_a,reward,state,done])
		if len(self.memory)>self.max_mem:
			del self.memory[0]

		# Select data from memory
		demo_ind = random.sample(range(1,len(self.memory_demo)),self.minib_demo)
		if len(self.memory)>self.minib & self.minib>1:
			sample_ind=random.sample(range(1,len(self.memory)),self.minib)
		elif self.minib==1:
			sample_ind=[len(self.memory)-1]
		else:
			sample_ind=range(len(self.memory))
		
		# Update network
		with self.sess.as_default():
			S1 = np.concatenate((np.stack([self.memory[ind][0] for ind in sample_ind]),
								np.stack([self.memory_demo[ind][0] for ind in demo_ind])))
			Sd1 = np.concatenate((np.stack([self.memory[ind][3] for ind in sample_ind]),
								np.stack([self.memory_demo[ind][3] for ind in demo_ind])))
			q_s1 = self.Qt.eval({self.state_input:S1})
			q_d1 = self.Qt.eval({self.state_input:Sd1})
			for i, ind in enumerate(sample_ind):
				r_mem = self.memory[ind][2]
				r= r_mem if np.abs(r_mem)==100 else 0
				a=self.memory[ind][1]
				if self.memory[ind][4]:
					q_s1[i][a] = r
				else:
					q_s1[i][a] = r + self.gamma*np.amax(q_d1[i])
			for i, ind in enumerate(demo_ind,len(sample_ind)):
				r_mem = self.memory_demo[ind][2]
				r= r_mem if np.abs(r_mem)==100 else 0
				a=self.memory_demo[ind][1]
				if self.memory_demo[ind][4]:
					q_s1[i][a] = r
				else:
					q_s1[i][a] = r + self.gamma*np.amax(q_d1[i])
			self.sess.run(self.updateModel,{self.state_input:S1,self.nextQ:q_s1})

		# Update target network
		self.step_count += 1
		if self.step_count>=self.update_steps:
# 			self.sess.run(self.wt_assign)
# 			self.sess.run(self.bt_assign)
# 			self.sess.run(self.Wt_assign)
			self.update_target()
			self.step_count=0