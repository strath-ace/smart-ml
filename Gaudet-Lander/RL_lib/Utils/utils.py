"""

Scaler and Logger adapted from code by Patrick Coady (pat-coady.github.io)

"""
import numpy as np
import os
import itertools
import tensorflow as tf
import pickle

class Scaler(object):
	""" Generate scale and offset based on running mean and stddev along axis=0

		offset = running mean
		scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
	"""

	def __init__(self, obs_dim):
		"""
		Args:
			obs_dim: dimension of axis=1
		"""
		self.vars = np.zeros(obs_dim)
		self.means = np.zeros(obs_dim)
		self.m = 0
		self.n = 0
		self.first_pass = True

	def update(self, x):
		""" Update running mean and variance (this is an exact method)
		Args:
			x: NumPy array, shape = (N, obs_dim)

		see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
			   variance-of-two-groups-given-known-group-variances-mean
		"""
		if self.first_pass:
			self.means = np.mean(x, axis=0)
			self.vars = np.var(x, axis=0)
			self.m = x.shape[0]
			self.first_pass = False
		else:
			n = x.shape[0]
			new_data_var = np.var(x, axis=0)
			new_data_mean = np.mean(x, axis=0)
			new_data_mean_sq = np.square(new_data_mean)
			new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
			self.vars = (((self.m * (self.vars + np.square(self.means))) +
						  (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
						 np.square(new_means))
			self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
			self.means = new_means
			#print('Scaler: ',self.means,np.sqrt(self.vars))
			self.m += n

	def apply(self,obs):
		""" returns 2-tuple: (scale, offset) """
		scale = 1/(np.sqrt(self.vars) + 0.1)/3
		return (obs-self.means) * scale

	def reverse(self,obs):
		scale = 1/(np.sqrt(self.vars) + 0.1)/3
		return obs / scale + self.means

	def scale_by_sd(self,obs):
		scale = 1/(np.sqrt(self.vars) + 0.1)/3
		return obs * scale

class Logger(object):
	""" Simple training logger: saves to file and optionally prints to stdout """
	def __init__(self):
		"""
		Args:
			logname: name for log (e.g. 'Hopper-v1')
			now: unique sub-directory name (e.g. date/time string)
		"""

		self.write_header = True
		self.log_entry = {}
		self.writer = None  # DictWriter created with first call to write() method
		self.scores = []

	def write(self, display=True):
		""" Write 1 log entry to file, and optionally to stdout
		Log fields preceded by '_' will not be printed to stdout

		Args:
			display: boolean, print to stdout
		"""
		if display:
			self.disp(self.log_entry)

	@staticmethod
	def disp(log):
		"""Print metrics to stdout"""
		log_keys = [k for k in log.keys()]
		log_keys.sort()
		print('***** Episode {}, Mean R = {:.1f}  Std R = {:.1f}  Min R = {:.1f}'.format(log['_Episode'],
															   log['_MeanReward'], log['_StdReward'], log['_MinReward']))
		for key in log_keys:
			if key[0] != '_':  # don't display log items with leading '_'
				#print(key, log[key])
				print('{:s}: {:.3g}'.format(key, log[key]))
		print('\n')

	def log(self, items):
		""" Update fields in log (does not write to file, used to collect updates.

		Args:
			items: dictionary of items to update
		"""
		self.log_entry.update(items)

	def close(self):
		pass

class Mapminmax(object):
	def __init__(self, x, ymin, ymax):
		self.xmin = np.min(x,axis=0)
		self.xmax = np.max(x,axis=0)
		self.ymin = ymin
		self.ymax = ymax

	def update(self,x):
		self.xmin = np.min(x,axis=0)
		self.xmax = np.max(x,axis=0)

	def apply(self,x):
		y = (self.ymax-self.ymin)*(x-self.xmin) / (self.xmax-self.xmin) + self.ymin
		return y

	def reverse(self,x):
		y = (x-self.ymin) / (self.ymax-self.ymin) * (self.xmax-self.xmin) + self.xmin
		return y

def discretize(x,n,min_action,max_action):
	"""
		n is number of samples per dimension
		d is the dimension of x

	"""
	x = np.clip(x,min_action,max_action)
	bins = np.linspace(min_action,max_action,n+1)
	#print(bins)
	indices = np.digitize(x,bins) - 1
	#print(indices)
	idx = indices >= n
	indices[idx] = n-1
	return indices 

class Action_converter(object):
	def __init__(self,action_dim,actions_per_dim,min_action=0.,max_action=1.):
		self.action_dim = action_dim
		self.actions_per_dim = actions_per_dim
		self.min_action = min_action
		self.max_action = max_action
		self.actions = np.linspace(min_action, max_action, actions_per_dim)
		self.action_table = np.asarray(list(itertools.product(self.actions, repeat=action_dim)))
		print('Actions')
		print(self.action_table)
	def idx2action(self,idx):
		return self.action_table[idx].T

	def action2idx(self,action):
		idx = discretize(action,self.actions_per_dim,self.min_action,self.max_action) 
		return idx


def save_vars(model,filename):
	graph_key = tf.GraphKeys.TRAINABLE_VARIABLES
	with model.g.as_default():
		dict = {}
		foo_names =  [v.name for v in  tf.get_collection(graph_key)]
		foo_vars =  [v for v in  tf.get_collection(graph_key)]
		foo_vars = model.sess.run(foo_vars)
		for i in range(len(foo_names)):
			key = foo_names[i]
			var = [v for v in  tf.get_collection_ref(graph_key) if v.name == key]
			tmp = model.sess.run(var[0])
			dict[key]=tmp
	print ('Saved Vars: ',foo_names)
	filename = filename + ".pkl"
	f = open(filename,"wb")
	pickle.dump(dict,f)
	f.close()

def load_vars(model,filename):
	graph_key = tf.GraphKeys.TRAINABLE_VARIABLES
	filename = filename + ".pkl" 
	dict = pickle.load(open(filename,"rb"))
	with model.g.as_default():
		keys = dict.keys()
		print(keys)
		update_ops = []
		for key in keys:
			new_value = dict[key]
			var = [v for v in  tf.get_collection_ref(graph_key) if v.name == key]
			v = var[0]
			op = v.assign(new_value)
			update_ops.append(op)
	model.sess.run(update_ops)


def save_run(policy, scaler, history, fname):
	save_vars(policy,fname)
	with open(fname + "_scaler.pkl", "wb") as f:
		pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)
	np.save(fname + "_history",history)

def load_run(policy, fname):
	with open(fname + "_scaler.pkl", "rb") as f:
		scaler = pickle.load(f)
	load_vars(policy,fname)
	return scaler

def get_mini_ids(m,k):
	num_batches = max(m // k, 1)
	batch_size = m // num_batches
	last_batch_size = m % num_batches
	indices = []
	for j in range(num_batches):
		start = j * batch_size
		end = (j + 1) * batch_size
		indices.append([start,end])
	if last_batch_size > 0:
		start=end
		end = m
		indices.append([start,end])
	return indices

def normc_initializer(std=1.0, axis=0):
	def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
		return tf.constant(out)
	return _initializer

def ortho_init(scale=1.0):
	def _ortho_init(shape, dtype, partition_info=None):
		#lasagne ortho init for tf
		shape = tuple(shape)
		if len(shape) == 2:
			flat_shape = shape
		elif len(shape) == 4: # assumes NHWC
			flat_shape = (np.prod(shape[:-1]), shape[-1])
		else:
			raise NotImplementedError
		a = np.random.normal(0.0, 1.0, flat_shape)
		u, _, v = np.linalg.svd(a, full_matrices=False)
		q = u if u.shape == flat_shape else v # pick the one with the correct shape
		q = q.reshape(shape)
		return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
	return _ortho_init

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
	with tf.variable_scope(scope):
		nin = x.get_shape()[1].value
		w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
		b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
		return tf.matmul(x, w)+b


