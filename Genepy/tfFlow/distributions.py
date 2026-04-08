"""Prior and class dsitributions"""

import tensorflow as tf
import tensorflow_probability as tfp

import pdb

class FactorialPrior(tf.Module):
	"""Prior distribution with separable components"""
	def __init__(self, dim,
				 distname = 'Normal',
				 train_cov = True,
				 name = 'factorialprior'):
		super().__init__(name=name)
		loc = tf.zeros(dim,dtype=tf.float32)
		scale = tf.Variable(
			tf.ones(dim,dtype=tf.float32),
			name=name+'_cov',
			trainable=train_cov,
		)
		try:
			dfunc = getattr(tfp.distributions,distname)
		except AttributeError:
			raise ValueError(f'{distname} not a valid tensorflow distribution')
		self._dist = dfunc(loc=loc, scale=scale)
	def log_prob(self, x):
		return tf.reduce_sum(self._dist.log_prob(x),axis=-1)
	def sample(self, n):
		return self._dist.sample(n)

class ClassDistribution(tf.Module):
	"""Distribution of categorical conditional labels"""
	def __init__(self, N_class,
				 class_weights = None,
				 name = 'classdist'):
		super().__init__(name=name)
		self.N_class = N_class
		if class_weights is None:
			class_weights = tf.ones(N_class, dtype=tf.float32) / N_class
		if class_weights.shape[0] != N_class:
			raise ValueError(f'Shape of class_weights does not match N_class')
		self.probs = tf.Variable(class_weights,
								 dtype=tf.float32,
								 trainable=False,
								 name=name+'_probs')
		self._dist = tfp.distributions.Categorical(probs=self.probs)
		self.class_mat = tf.eye(N_class, dtype=tf.float32)
	def sample(self, n):
		i_samp = self._dist.sample(n)
		return tf.gather(self.class_mat,i_samp,axis=0)

class ContinuousDistribution(tf.Module):
	"""Distribution of conditional labels taking real values"""
	def __init__(self, min_val, max_val,
				 name = 'continuousclassdist'):
		super().__init__(name=name)
		self.low = tf.Variable(min_val,
							   dtype=tf.float32,
							   trainable=False,
							   name=name+'_low')
		self.high = tf.Variable(max_val,
								dtype=tf.float32,
								trainable=False,
								name=name+'high')
		self._dist = tfp.distributions.Uniform(low=self.low,high=self.high)
	def sample(self, n):
		return self._dist.sample(n)

class CombinedDistribution(tf.Module):
	"""Combination of real values and categorical classes"""
	def __init__(self, min_val, max_val, N_class,
				 class_weights = None,
				 name = 'combinedclassdist'):
		super().__init__(name=name)
		self.contdist = ContinuousDistribution(
			min_val, max_val,
			seed = seed,
			name = name+'_cont'
		)
		self.catdist = ClassDistribution(
			N_class-min_val.shape[0],
			class_weights = class_weights,
			seed = seed,
			name = name+'_cont'
		)
	def sample(self, n):
		x_cont = self.contdist.sample(n)
		x_cat = self.catdist.sample(n)
		return tf.concat([x_cont,x_cat],axis=-1)


