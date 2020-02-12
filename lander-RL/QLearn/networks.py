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
		self.target=is_target
		if self.target:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
			return
		
		# Update rules
		self.k = minibatch_size
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
		p_assign = [p for p in p_new if p in self.params]
		self.sess.run([getattr(self,p).assign(p_new[p]) for p in p_assign])
		
	def get_params(self):
		p_list = self.sess.run([getattr(self,p) for p in self.params])
		return dict(zip(self.params,p_list))
		
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
				N_hid=[20, 10], alpha=0.01, activation_function='tanh', update_steps=50, clip_norm=None,
				W_init_magnitude=0.1, w_init_magnitude=0.1, b_init_magnitude=0.0, minibatch_size=10, **kwargs):
		"""
		A multi-layer feedforward network which uses gradient-based updates
		"""
		# build network
		self.n_layer = len(N_hid)
		self.s_input = tf.placeholder(shape=[None,state_size],dtype=tf.float32)
		self.w = []
		self.b = []
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
		
		# Update rules
		self.prep_state = None
		self.k = minibatch_size
		self.nextQ = tf.placeholder(shape=[None,action_size],dtype=tf.float32)
		loss = tf.reduce_sum(tf.square(self.nextQ - self.Q_est))
		trainer = tf.train.RMSPropOptimizer(alpha)
		if clip_norm is not None:
			grads = trainer.compute_gradients(loss,self.var_list)
			cap_grads = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grads]
			self.updateModel = trainer.apply_gradients(cap_grads)
		else:
			self.updateModel = trainer.minimize(loss,var_list=self.var_list)

		# Target network
		self.wt = []
		self.bt = []
		for i in range(self.n_layer):
			self.wt.append(tf.Variable(tf.random_uniform([N_hid[i],N_hid[i+1]],0,1)))
			self.bt.append(tf.Variable(tf.random_uniform([1,N_hid[i+1]],0,1)))
		self.Wt = tf.Variable(tf.random_uniform([N_hid[-1],action_size],0,1))
		self.layers_t = [self.s_input]
		for w, b in zip(self.wt,self.bt):
			self.layers_t.append(act_fn(tf.add(tf.matmul(self.layers_t[-1],w),b)))
		self.Qt = tf.matmul(self.layers_t[-1],self.Wt)
		
		self.wt_assign = [self.wt[i].assign(self.w[i]) for i in range(self.n_layer)]
		self.bt_assign = [self.bt[i].assign(self.b[i]) for i in range(self.n_layer)]
		self.Wt_assign = self.Wt.assign(self.W)
		self.update_target = self.wt_assign + self.bt_assign + [self.Wt_assign]
		
		# Start tensorflow session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(self.update_target)
		
		self.step_count = 0
		self.C = update_steps
		
	def Q_predict(self, s):
		return self.sess.run(self.Q_est,feed_dict={self.s_input:s})

	def Q_target(self, s):
		return self.sess.run(self.Qt,feed_dict={self.s_input:s})

	def update(self, S, Q):
		self.sess.run(self.updateModel,{self.s_input:S,self.nextQ:Q})

		# Update target network
		self.step_count += 1
		if self.step_count>=self.C:
			self.sess.run(self.update_target)
			self.step_count=0