"""Layers for composition in a flow model"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import numpy as np

import pdb


class NiceCouplingLayer(tf.Module):
	"""
 Bijective additive coupling layer

 Switches order of outputs to allow composition
 """
	def __init__(self, m, d, name=None):
		"""
  m : trainable mapping

  d : split point for partitioning inputs, int

  name : str
  """
		super().__init__(name=name)
		self.m = m
		self.d = d
	def forward(self, x, *args):
		x_I1 = x[...,:self.d]
		x_I2 = x[...,self.d:]
		return tf.concat([x_I2+self.m(x_I1),x_I1],axis=-1), tf.zeros(x.shape[:-1], dtype=x.dtype)
	def inverse(self, y, *args):
		y_I1 = y[...,-self.d:]
		y_I2 = y[...,:-self.d]
		return tf.concat([y_I1,y_I2-self.m(y_I1)],axis=-1), tf.zeros(y.shape[:-1], dtype=y.dtype)
	def get_weights(self):
		return {v.name: v.numpy() for v in self.variables}

class NiceConditionalCouplingLayer(tf.Module):
	"""
 Bijective additive coupling layer with conditional input

 Switches order of outputs to allow composition
 """
	def __init__(self, m, d, name=None):
		"""
  m : trainable mapping

  d : split point for partitioning inputs, int

  name : str
  """
		super().__init__(name=name)
		self.m = m
		self.d = d
	def forward(self, x, c):
		x_I1 = x[...,:self.d]
		t = self.m(tf.concat([x_I1,c],axis=-1))
		x_I2 = x[...,self.d:]
		return tf.concat([x_I2+t,x_I1],axis=-1), tf.zeros(x.shape[:-1], dtype=x.dtype)
	def inverse(self, y, c):
		y_I1 = y[...,-self.d:]
		t = self.m(tf.concat([y[...,-self.d:],c],axis=-1))
		y_I2 = y[...,:-self.d]
		return tf.concat([y_I1,y_I2-t],axis=-1), tf.zeros(y.shape[:-1], dtype=y.dtype)
	def get_weights(self):
		return {v.name: v.numpy() for v in self.variables}


class ScalingLayer(tf.Module):
	"""Linear scaling layer"""
	def __init__(self, dim, name=None):
		"""
  dim : dimensionality of input, int

  name: str
  """
		super().__init__(name=name)
		self.S = tf.Variable(tf.zeros([dim],dtype=tf.float32),
							 trainable=True,
							 name='S_exp')
	def forward(self, x, *args):
		return x*tf.math.exp(self.S), tf.reduce_sum(self.S)
	def inverse(self, y, *args):
		return y*tf.math.exp(-self.S), tf.reduce_sum(-self.S)
	def get_weights(self):
		return {v.name: v.numpy() for v in self.variables}


class NormScaleLayer(tf.Module):
	"""Normlisation based on https://github.com/xqding/RealNVP"""
	def __init__(self, dim, xmax=1., name=None):
		"""
  dim : dimensionality of input, int

  xmax : maximum value of output, float

  name: str
  """
		super().__init__(name=name)
		self.xmax = tf.Variable(xmax,
								trainable=False,
								dtype=tf.float32,
								name='xmax')
	def forward(self, x, *args):
		logdet = tf.reduce_sum( tf.math.log(tf.abs( 1./(self.xmax*(1-(x/self.xmax)**2)) )) , axis=-1)
		return 0.5 * ( tf.math.log(1+x/self.xmax) - tf.math.log(1-x/self.xmax) ), logdet
	def inverse(self, y, *args):
		logdet = tf.reduce_sum( tf.math.log(tf.abs(self.xmax*(1-tf.tanh(y))**2)) , axis=-1)
		return self.xmax * tf.tanh(y), logdet
	def get_weights(self):
		return {v.name: v.numpy() for v in self.variables}
	
