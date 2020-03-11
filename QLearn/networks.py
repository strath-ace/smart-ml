# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

try:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()
except ImportError:
	import tensorflow as tf
	
import pdb

			
class QNet():
	def __init__(self, state_size, action_size,
				N_hid=None, alpha=0.01, activation_function='tanh', update_steps=50, clip_norm=None,
				W_init_magnitude=1.0, w_init_magnitude=1.0, b_init_magnitude=0.0, minibatch_size=5, 
				 is_target=False, **kwargs):
		"""
		A network which uses gradient-based updates
		"""
		N_hid = action_size if N_hid is None else N_hid
		# build network
		self.s_input = tf.placeholder(shape=[None,state_size],dtype=tf.float32)
		self.w_in = tf.Variable(tf.random_uniform([state_size,N_hid],0,w_init_magnitude))
		self.b_in = tf.Variable(tf.random_uniform([1,N_hid],0,b_init_magnitude))
		self.W = tf.Variable(tf.random_uniform([N_hid,action_size],0,W_init_magnitude))
		try:
			act_fn = tf.keras.activations.get(activation_function)
		except ValueError:
			act_fn = tf.tanh
		self.act = act_fn(tf.add(tf.matmul(self.s_input,self.w_in),self.b_in))
		self.Q_est = tf.matmul(self.act,self.W)
		self.prep_state = None
		
		self.params = ['W', 'w_in', 'b_in']
		self.p_dict = {'W':self.W, 'w':self.w_in, 'b':self.b_in}
		self.new_params = {'W':tf.placeholder(shape=[None,None],dtype=tf.float32),
						  'w':tf.placeholder(shape=[None,None],dtype=tf.float32),
						  'b':tf.placeholder(shape=[None,None],dtype=tf.float32)}
		self.p_assign = [self.W.assign(self.new_params['W']),
						self.w_in.assign(self.new_params['w']),
						self.b_in.assign(self.new_params['b'])]
		
		self.target=is_target
		if self.target:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
			return
		
		# Update rules
		self.k = int(minibatch_size)
		self.nextQ = tf.placeholder(shape=[None,action_size],dtype=tf.float32)
		loss = tf.reduce_sum(tf.square(self.nextQ - self.Q_est))
		trainer = tf.train.RMSPropOptimizer(alpha)
		if clip_norm is not None:
			grads = trainer.compute_gradients(loss,[self.W,self.w_in,self.b_in])
			cap_grads = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grads]
			self.updateModel = trainer.apply_gradients(cap_grads)
		else:
			self.updateModel = trainer.minimize(loss,var_list=[self.W,self.w_in,self.b_in])
		
		# Start tensorflow session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		
	def assign_params(self,p_new):	
		p_assign_dict = {self.new_params['W']:p_new['W'],
						self.new_params['w']:p_new['w'],
						self.new_params['b']:p_new['b']}
		self.sess.run(self.p_assign, feed_dict=p_assign_dict)
		
	def get_params(self):
		return self.sess.run(self.p_dict)
		
	def Q_predict(self, s=None, s_prep=None):
		if s is not None:
			return self.sess.run(self.Q_est,feed_dict={self.s_input:s})
		elif s_prep is not None:
			return self.sess.run(self.Q_est,feed_dict={self.prep_state:s_prep})
		else:
			return []

	def update(self, S, Q):
		if not self.target:
			self.sess.run(self.updateModel,{self.s_input:S,self.nextQ:Q})
			
			
class MLPQNet():
	def __init__(self, state_size, action_size,
				hidden_layers=[20, 10], alpha=0.01, activation_function='tanh', update_steps=50, clip_norm=1.0,
				W_init_magnitude=0.1, w_init_magnitude=0.1, b_init_magnitude=0.0, minibatch_size=10,
				 is_target=False, **kwargs):
		"""
		A multi-layer feedforward network which uses gradient-based updates
		"""
		# build network
		self.n_layer = len(hidden_layers)
		self.s_input = tf.placeholder(shape=[None,state_size],dtype=tf.float32)
		self.w = []
		self.b = []
		N_hid = hidden_layers.copy()
		N_hid.insert(0,state_size)
		for i in range(self.n_layer):
			self.w.append(tf.Variable(tf.random_uniform([N_hid[i],N_hid[i+1]],0,w_init_magnitude)))
			self.b.append(tf.Variable(tf.random_uniform([1,N_hid[i+1]],0,b_init_magnitude)))
		self.W = tf.Variable(tf.random_uniform([N_hid[-1],action_size],0,W_init_magnitude))
		
		try:
			act_fn = tf.keras.activations.get(activation_function)
		except ValueError:
			act_fn = tf.tanh
		self.layers = [self.s_input]
		for w, b in zip(self.w,self.b):
			self.layers.append(act_fn(tf.add(tf.matmul(self.layers[-1],w),b)))
		self.Q_est = tf.matmul(self.layers[-1],self.W)
		self.var_list = self.w + self.b + [self.W]
		
		self.params = ['W', 'w', 'b']
		self.p_dict = {'W':self.W, 'w':self.w, 'b':self.b}
		self.new_params = {'W':tf.placeholder(shape=[None,None],dtype=tf.float32)}
		self.new_params['w'] = [tf.placeholder(shape=[None,None],dtype=tf.float32) for _ in self.w]
		self.new_params['b'] = [tf.placeholder(shape=[None,None],dtype=tf.float32) for _ in self.b]
		
		self.p_assign = [self.W.assign(self.new_params['W'])]
		self.p_assign += [w.assign(self.new_params['w'][i]) for i, w in enumerate(self.w)]
		self.p_assign += [b.assign(self.new_params['b'][i]) for i, b in enumerate(self.b)]
		
		self.target=is_target
		if self.target:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
			return
		
		# Update rules
		self.prep_state = None
		self.k = int(minibatch_size)
		self.nextQ = tf.placeholder(shape=[None,action_size],dtype=tf.float32)
		loss = tf.reduce_sum(tf.square(self.nextQ - self.Q_est))
		trainer = tf.train.RMSPropOptimizer(alpha)
		if clip_norm is not None:
			grads = trainer.compute_gradients(loss,self.var_list)
			cap_grads = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grads]
			self.updateModel = trainer.apply_gradients(cap_grads)
		else:
			self.updateModel = trainer.minimize(loss,var_list=self.var_list)

		# Start tensorflow session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def update(self, S, Q):
		self.sess.run(self.updateModel,{self.s_input:S,self.nextQ:Q})

	def assign_params(self,p_new):
		p_assign_dict = {self.new_params['W']:p_new['W']}
		for n in range(self.n_layer):
			p_assign_dict[self.new_params['w'][n]] = p_new['w'][n]
			p_assign_dict[self.new_params['b'][n]] = p_new['b'][n]
		self.sess.run(self.p_assign, feed_dict=p_assign_dict)
		
	def get_params(self):
		return self.sess.run(self.p_dict)
		
	def Q_predict(self, s=None, s_prep=None):
		if s is not None:
			return self.sess.run(self.Q_est,feed_dict={self.s_input:s})
		elif s_prep is not None:
			return self.sess.run(self.Q_est,feed_dict={self.prep_state:s_prep})
		else:
			return []

	def update(self, S, Q):
		if not self.target:
			self.sess.run(self.updateModel,{self.s_input:S,self.nextQ:Q})
			

class ELMNet():
	def __init__(self, state_size, action_size,
				N_hid=None, gamma_reg=0.001, activation_function='tanh', update_steps=50, 
				w_init_magnitude=1.0, b_init_magnitude=0.0, minibatch_size=5, 
				 prep_state=True, is_target=False, **kwargs):
		"""
		A network which uses updates based on sequential regularised ELM
		"""
		N_hid = action_size if N_hid is None else N_hid
		# build network
		self.s_input = tf.placeholder(shape=[None,state_size],dtype=tf.float32)
		self.w_in = tf.Variable(tf.random_uniform([state_size,N_hid],0,w_init_magnitude))
		self.b_in = tf.Variable(tf.random_uniform([1,N_hid],0,b_init_magnitude))
		self.W = tf.Variable(tf.random_uniform([N_hid,action_size],0,1))
		try:
			act_fn = tf.keras.activations.get(activation_function)
		except ValueError:
			act_fn = tf.tanh
		self.act = act_fn(tf.add(tf.matmul(self.s_input,self.w_in),self.b_in))
		self.Q_est = tf.matmul(self.act,self.W)
		self.prep_state = self.act if prep_state else None
		
		self.params = ['W', 'w_in', 'b_in']
		self.p_dict = {'W':self.W, 'w':self.w_in, 'b':self.b_in}
		self.new_params = {'W':tf.placeholder(shape=[None,None],dtype=tf.float32),
						  'w':tf.placeholder(shape=[None,None],dtype=tf.float32),
						  'b':tf.placeholder(shape=[None,None],dtype=tf.float32)}
		self.p_assign = [self.W.assign(self.new_params['W']),
						self.w_in.assign(self.new_params['w']),
						self.b_in.assign(self.new_params['b'])]
		
		self.target=is_target
		if self.target:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
			return
		
		# Matrices for updating
		self.k = minibatch_size
		self.H = tf.placeholder(shape=[self.k,N_hid],dtype=tf.float32)
		self.T = tf.placeholder(shape=[self.k,action_size],dtype=tf.float32)
		self.H_trans = tf.transpose(self.H)
		self.A_inv = tf.Variable(tf.random_uniform([N_hid,N_hid],0,1))

		# Initialisation
		A_t1 = tf.add(tf.scalar_mul(1.0/gamma_reg,tf.eye(N_hid)),tf.matmul(self.H_trans,self.H))
		A_t1_inv = tf.matrix_inverse(A_t1)
		W_t1 = tf.matmul(A_t1_inv,tf.matmul(self.H_trans,self.T))
		self.W_init = self.W.assign(W_t1)
		self.A_init = self.A_inv.assign(A_t1_inv)
		
		# Updating
		K1 = tf.add(tf.matmul(self.H,tf.matmul(self.A_inv,self.H_trans)),tf.eye(self.k))
		K_t = tf.subtract(tf.eye(N_hid),
			tf.matmul(self.A_inv,tf.matmul(self.H_trans,tf.matmul(tf.matrix_inverse(K1),self.H))))
		W_new = tf.add(tf.matmul(K_t,self.W),
			tf.matmul(tf.matmul(K_t,self.A_inv),tf.matmul(self.H_trans,self.T)))
		A_new = tf.matmul(K_t,self.A_inv)
		self.W_update = self.W.assign(W_new)
		self.A_update = self.A_inv.assign(A_new)
		
		# Start tensorflow session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		self.first = True
		
# 	def assign_params(self,p_new):
# 		p_assign = [p for p in p_new if p in self.params]
# 		self.sess.run([getattr(self,p).assign(p_new[p]) for p in p_assign])
		
# 	def get_params(self):
# 		p_list = self.sess.run([getattr(self,p) for p in self.params])
# 		return dict(zip(self.params,p_list))
	
	def assign_params(self,p_new):	
		p_assign_dict = {self.new_params['W']:p_new['W'],
						self.new_params['w']:p_new['w'],
						self.new_params['b']:p_new['b']}
		self.sess.run(self.p_assign, feed_dict=p_assign_dict)
		
	def get_params(self):
		return self.sess.run(self.p_dict)
		
	def Q_predict(self, s=None, s_prep=None):
		if s is not None:
			return self.sess.run(self.Q_est,feed_dict={self.s_input:s})
		elif s_prep is not None:
			return self.sess.run(self.Q_est,feed_dict={self.prep_state:s_prep})
		else:
			return []

	def update(self, H, T):
		if self.target:
			return
		if self.first:
			self.sess.run(self.W_init,feed_dict={self.H:H,self.T:T})
			self.sess.run(self.A_init,feed_dict={self.H:H})
			self.first = False
		else:
			self.sess.run(self.W_update,feed_dict={self.H:H,self.T:T})
			self.sess.run(self.A_update,feed_dict={self.H:H})