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


class ELMNet():
	def __init__(self, state_size, action_size,
				N_hid=None, gamma_reg=0.001, activation_function='tanh', update_steps=50, 
				w_init_magnitude=0.001, b_init_magnitude=0.001, minibatch_size=5, 
				 prep_state=True,**kwargs):
		"""
		A network which uses updates based on sequential regularised ELM
		"""
		N_hid = action_size if N_hid is None else N_hid
		# build network
		self.s_input = tf.placeholder(shape=[None,state_size],dtype=tf.float32)
		self.w_in = tf.Variable(tf.random_uniform([state_size,N_hid],-w_init_magnitude,w_init_magnitude))
		self.b_in = tf.Variable(tf.random_uniform([1,N_hid],0,b_init_magnitude))
		self.W = tf.Variable(tf.random_uniform([N_hid,action_size],0,1))
		try:
			act_fn = tf.keras.activations.get(activation_function)
		except ValueError:
			act_fn = tf.tanh
		self.act = act_fn(tf.add(tf.matmul(self.s_input,self.w_in),self.b_in))
		self.Q_est = tf.matmul(self.act,self.W)
		
		# Matrices for updating
		self.k = minibatch_size
		self.prep_state = self.act if prep_state else None
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

		# Target network update
		self.Wt = tf.Variable(tf.random_uniform([N_hid,action_size],0,1))
		self.Qt = tf.matmul(self.act,self.Wt)
		self.Wt_assign = self.Wt.assign(self.W)
		
		# Start tensorflow session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(self.Wt_assign)
		
		self.step_count = 0
		self.C = update_steps
		self.first = True
		
	def Q_predict(self, s):
		return self.sess.run(self.Q_est,feed_dict={self.s_input:s})
	
	def Q_predict_prep(self, H):
		return self.sess.run(self.Q_est,feed_dict={self.act:H})

	def Q_target(self, H):
		return self.sess.run(self.Qt,feed_dict={self.act:H})

	def update(self, H, T):
		if self.first:
			self.sess.run(self.W_init,feed_dict={self.H:H,self.T:T})
			self.sess.run(self.A_init,feed_dict={self.H:H})
			self.first = False
		else:
			self.sess.run(self.W_update,feed_dict={self.H:H,self.T:T})
			self.sess.run(self.A_update,feed_dict={self.H:H})

		self.step_count += 1
		if self.step_count >= self.C:
			self.sess.run(self.Wt_assign)
			self.step_count=0
	
class QNet():
	def __init__(self, state_size, action_size,
				N_hid=None, alpha=0.01, activation_function='tanh', update_steps=50, clip_norm=None,
				w_init_magnitude=0.001, b_init_magnitude=0.001, minibatch_size=5, **kwargs):
		"""
		A network which uses gradient-based updates
		"""
		N_hid = action_size if N_hid is None else N_hid
		# build network
		self.s_input = tf.placeholder(shape=[None,state_size],dtype=tf.float32)
		self.w_in = tf.Variable(tf.random_uniform([state_size,N_hid],-w_init_magnitude,w_init_magnitude))
		self.b_in = tf.Variable(tf.random_uniform([1,N_hid],0,b_init_magnitude))
		self.W = tf.Variable(tf.random_uniform([N_hid,action_size],0,1))
		try:
			act_fn = tf.keras.activations.get(activation_function)
		except ValueError:
			act_fn = tf.tanh
		self.act = act_fn(tf.add(tf.matmul(self.s_input,self.w_in),self.b_in))
		self.Q_est = tf.matmul(self.act,self.W)
		
		# Update rules
		self.prep_state = None
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

		# Target network
		self.wt = tf.Variable(tf.random_uniform([state_size,N_hid],0,0.01))
		self.bt = tf.Variable(tf.random_uniform([1,N_hid],0,0.01))
		self.Wt = tf.Variable(tf.random_uniform([N_hid,action_size],0,0.1))
		self.actt = act_fn(tf.add(tf.matmul(self.s_input,self.wt),self.bt))
		self.Qt = tf.matmul(self.actt,self.Wt)
		
		self.wt_assign = self.wt.assign(self.w_in)
		self.bt_assign = self.bt.assign(self.b_in)
		self.Wt_assign = self.Wt.assign(self.W)
		
		# Start tensorflow session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.sess.run([self.Wt_assign, self.wt_assign, self.bt_assign])
		
		self.step_count = 0
		self.C = update_steps
		
	def Q_predict(self, s):
		return self.sess.run(self.Q_est,feed_dict={self.s_input:s})

	def Q_target(self, s):
		return self.sess.run(self.Qt,feed_dict={self.s_input:s})

	def update(self, S, Q):
		self.sess.run(self.updateModel,{self.state_input:S,self.nextQ:Q})

		# Update target network
		self.step_count += 1
		if self.step_count>=self.C:
			self.sess.run([self.Wt_assign, self.wt_assign, self.bt_assign])
			self.step_count=0