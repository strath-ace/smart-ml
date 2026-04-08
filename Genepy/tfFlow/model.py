"""Tensorflow modules for flow models"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import numpy as np

from .layers import *

import pdb


class NormalisingFlowModel(tf.Module):
	"""Class of generic flow model"""
	def __init__(self, prior, layers,
				 seed = None, 
				 name = None):
		"""
  prior : tf distribution

  layers : list of classes of tfFlow.layers

  seed : int

  name : str
  """
		super().__init__(name=name)
		rng = np.random.default_rng(seed=seed)
		tf.random.set_seed(rng.integers(1e7))
		self.prior = prior
		self.layers = layers
			
	def __call__(self, x, *args):
		return self.forward(x, *args)
	def forward(self, x, *args):
		"""Transformation to latent space and log-determinant"""
		z = x
		logdet_sum = 0
		for layer in self.layers:
			z, logdet = layer.forward(z, *args)
			logdet_sum += logdet
		return z, logdet_sum
	def inverse(self, z, *args):
		"""Transformation from latent space and log-determinant"""
		x = z
		logdet_sum = 0
		for layer in reversed(self.layers):
			x, logdet = layer.inverse(x, *args)
			logdet_sum += logdet
		return x, logdet_sum
	def log_likelihood(self, x, *args):
		"""Log-likelihood of point"""
		z, logdet = self.forward(x, *args)
		return tf.add(self.prior.log_prob(z), logdet)
	def sample_inv(self, n):
		"""Inverse transformation of samples from distribution"""
		z = self.prior.sample(n)
		return self.inverse(z)[0]
	def sample_ll_inv(self, n):
		"""Inverse transformation of samples from distribution with log likelihood"""
		z = self.prior.sample(n)
		x, logdet = self.inverse(z)
		return x, tf.add(self.prior.log_prob(z), logdet)
	def get_weights(self):
		w_all = [l.get_weights() for l in self.layers]
		return w_all
	def get_params(self):
		return {v.name: v.numpy() for v in self.variables}


class ConditionalFlowModel(NormalisingFlowModel):
	"""Class of conditional flow model"""
	def __init__(self, prior, layers, class_prior,
				 seed = None, 
				 name = None):
		"""
  prior : tf distribution

  layers : list of classes of tfFlow.layers

  class_prior : tf distribution

  seed : int

  name : str
  """
		super().__init__(prior, layers, seed=seed, name=name)
		self.class_prior = class_prior
		try:
			self.N_class = self.class_prior.N_class
		except AttributeError:
			self.N_class = None
			
	def log_likelihood(self, x, c):
		"""Log-likelihood of point given class c"""
		z, logdet = self.forward(x, c)
		return tf.add(self.prior.log_prob(z), logdet)
	def max_log_likelihood(self, x):
		"""Maximum log-likelihood of unlabelled point"""
		c_tile = tf.tile(self.class_prior.class_mat,[x.shape[0],1])
		ll_all = tf.TensorArray(tf.float32, size=self.N_class)
		for i in tf.range(self.N_class):
			c = tf.gather(c_tile,tf.range(i,c_tile.shape[0],self.N_class))
			ll_all = ll_all.write( i, self.log_likelihood(x,c) )
		return tf.reduce_max( tf.transpose(ll_all.stack()), axis=-1 )
	def sample_inv(self, n):
		"""Inverse transformation of samples from distribution"""
		z = self.prior.sample(n)
		c = self.class_prior.sample(n)
		return self.inverse(z, c)[0], c
	def sample_inv_class(self, n, c):
		"""Inverse transformation of samples from distribution given class c"""
		z = self.prior.sample(n)
		return self.inverse(z, c)[0]

