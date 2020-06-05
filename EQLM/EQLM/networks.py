# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2020 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------
"""Classes containing tensorflow implementations of Q-Networks"""

try:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()
except ImportError:
	import tensorflow as tf

	
class SingleLayerNetwork(object):
	"""
	A Q-Network with a single hidden layer

	...

	Attributes
	----------
	s_input : tf.Tensor
		Placeholder for the state input to the Q-Network
	w_in : tf.Variable
		Input weights, shape SxN
	b_in : tf.Variable
		Neuron biases, shape 1xN
	W : tf.Variable
		Output weights, shape NxA
	act : tf.Tensor
		Neuron outputs, i.e. before output weights
	Q_est : tf.Tensor
		Estimated action values
	prep_state : tf.Tensor
		If states are 'preprocessed' in memory, this points to the preprocess layer
	params : dict
		Dictionary of network parameters
	new_params : dict
		Dictionary of placeholders for new parameters
	p_assign : list
		Assignment operations for new parameters
	updateModel : tf.Operation
		For running network updates
	sess : tf.Session
		Tensorflow session
	"""
	sess = tf.Session()
	def __init__(self, state_size, action_size,
				N_hid=None, activation_function='tanh',
				w_mag=0.1, b_mag=0.0, W_mag=0.1,
				is_target=False, **kwargs):
		"""
		Parameters
		----------
		state_size, action_size : int
			Size of the environment state space and action space
		N_hid : int, optional
			Number of hidden nodes, default N_hid = 2*action_size
		activation_function : str, optional
			Neuron activation function, default tanh
		w_mag, b_mag, W_mag : float, optional
			Initialisation magnitude for each set of parameters, default 0.1, 0.0, 0.1
		is_target : bool, optional
			Whether the network is a target network which does not update, default False
		"""
		self.N_hid = 2*action_size if N_hid is None else N_hid
		self.s_input = tf.placeholder(shape=[None,state_size],dtype=tf.float32)
		self.w_in = tf.Variable(tf.random_uniform([state_size,self.N_hid],0,w_mag))
		self.b_in = tf.Variable(tf.random_uniform([1,self.N_hid],0,b_mag))
		self.W = tf.Variable(tf.random_uniform([self.N_hid,action_size],0,W_mag))

		act_fn = tf.keras.activations.get(activation_function)
		self.act = act_fn(tf.add(tf.matmul(self.s_input,self.w_in),self.b_in))
		self.Q_est = tf.matmul(self.act,self.W)
		self.prep_state = None
		
		self.params = {'W':self.W, 'w':self.w_in, 'b':self.b_in}
		self.new_params = {'W':tf.placeholder(shape=[None,None],dtype=tf.float32),
						  'w':tf.placeholder(shape=[None,None],dtype=tf.float32),
						  'b':tf.placeholder(shape=[None,None],dtype=tf.float32)}
		self.p_assign = [self.W.assign(self.new_params['W']),
						self.w_in.assign(self.new_params['w']),
						self.b_in.assign(self.new_params['b'])]

		self.updateModel = tf.no_op()
		
	def var_init(self):
		"""Initial graph variables"""
		self.sess.run(tf.global_variables_initializer())
		
	def assign_params(self,p_new):
		"""Assign new values to updatable parameters"""
		p_assign_dict = {self.new_params['W']:p_new['W'],
						self.new_params['w']:p_new['w'],
						self.new_params['b']:p_new['b']}
		self.sess.run(self.p_assign, feed_dict=p_assign_dict)
		
	def get_params(self):
		"""Return current values of updatable parameters"""
		return self.sess.run(self.params)
		
	def Q_predict(self, s=None, s_prep=None):
		"""
		Return predicted action-values for the state
		
		...
		
		Parameters
		----------
		s : array-like, optional
			State in its `normal` form
		s_prep : array_like, optional
			State in a preprocessed form defined by `prep_state`
			
		Returns
		-------
		array-like
			Predicted action values for the state, empty if `s` and `s_prep` are `None`
		"""
		if s is not None:
			return self.sess.run(self.Q_est,feed_dict={self.s_input:s})
		elif s_prep is not None:
			return self.sess.run(self.Q_est,feed_dict={self.prep_state:s_prep})
		else:
			return []

	def update(self, *args):
		"""Method for updating network parameters"""
		self.sess.run(self.updateModel)
		
	def close(self):
		"""Close the tensorflow session"""
		self.sess.close()
		

class QNet(SingleLayerNetwork):
	"""
	A Q-Network which uses gradient based updates
	
	...
	
	Attributes
	----------
	k : int
		Size of minibatches for updating
	nextQ : tf.Tensor
		Placeholder for target action-values
	updateModel : tf.Operation
		Runs RMSPropOptimizer to minimize mean squared error
	"""
	def __init__(self, state_size, action_size, 
				 alpha=0.01, clip_norm=None, minibatch_size=5, **kwargs):
		"""
		Parameters
		----------
		state_size, action_size : int
			Size of the environment state space and action space
		alpha : float, optional
			Network learning rate
		clip_norm : float, optional
			Max gradient magnitude for clipping, default no clipping
		minibatch_size : int, optional
			Size of minibatches for updating
		**kwargs
			Additional keyword arguments passed to `SingleLayerNetwork`
		"""
		super().__init__(state_size, action_size, **kwargs)
		
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
			
		self.var_init()
		
	def update(self, S, Q):
		"""Updates based on states and target action values"""
		self.sess.run(self.updateModel,{self.s_input:S,self.nextQ:Q})
		
		
class ELMNet(SingleLayerNetwork):
	"""
	A Q-Network which uses ELM inspired updates
	
	...
	
	Attributes
	----------
	k : int
		Size of minibatches for updating
	prep_state : tf.Tensor
		Output of each neuron for use as a preprocessed state
	H : tf.Tensor
		Placeholder for minibatches of pre-processed states
	T : tf.Tensor
		Placeholder for target action-values
	initModel, updateModel : tuple of tf.Tensor
		Assignment operations for initialising and updating the weights
	first : bool
		Used to indicate the first update to initialise weights
	"""
	def __init__(self, state_size, action_size, 
				 gamma_reg=0.001, minibatch_size=5, **kwargs):
		"""
		Parameters
		----------
		state_size, action_size : int
			Size of the environment state space and action space
		gamma_reg : float, optional
			LS-IELM regularisation parameter
		minibatch_size : int, optional
			Size of minibatches for updating
		**kwargs
			Additional keyword arguments passed to `SingleLayerNetwork`
		"""
		super().__init__(state_size, action_size, **kwargs)

		self.k = int(minibatch_size)
		self.prep_state = self.act
		self.H = tf.placeholder(shape=[self.k,self.N_hid],dtype=tf.float32)
		self.T = tf.placeholder(shape=[self.k,action_size],dtype=tf.float32)
		H_t = tf.transpose(self.H)
		A_inv = tf.Variable(tf.random_uniform([self.N_hid,self.N_hid],0,1))

		A0 = tf.add(tf.scalar_mul(1.0/gamma_reg,tf.eye(self.N_hid)),tf.matmul(H_t,self.H))
		A0_inv = tf.matrix_inverse(A0)
		W0 = tf.matmul(A0_inv,tf.matmul(H_t,self.T))
		self.initModel = (self.W.assign(W0), A_inv.assign(A0_inv))

		K1 = tf.add(tf.matmul(self.H,tf.matmul(A_inv,H_t)),tf.eye(self.k))
		K_t = tf.subtract(tf.eye(self.N_hid),
			tf.matmul(A_inv,tf.matmul(H_t,tf.matmul(tf.matrix_inverse(K1),self.H))))
		W_new = tf.add(tf.matmul(K_t,self.W),
			tf.matmul(tf.matmul(K_t,A_inv),tf.matmul(H_t,self.T)))
		A_new = tf.matmul(K_t,A_inv)
		self.updateModel = (self.W.assign(W_new), A_inv.assign(A_new))
		
		self.first = True
		self.var_init()
		
	def update(self, H, T):
		"""Updates based on preprocessed states and target action values"""
		if self.first:
			self.sess.run(self.initModel,feed_dict={self.H:H,self.T:T})
			self.first = False
		else:
			self.sess.run(self.updateModel,feed_dict={self.H:H,self.T:T})
