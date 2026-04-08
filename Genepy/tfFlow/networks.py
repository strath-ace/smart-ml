"""Functions and classes for NNs"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

from .deltalayer import DiscontinuityDense

import pdb

def create_mlp_named(N_hid, dim_in, 
					 dim_out=None, 
					 activation='relu',
					 activation_out=None,
					 name='mlp'):
	"""
 Return a MLP that can be used as 'm' function

 N_hid : number of hidden nodes in each layer, list of int

 dim_in : input size, int

 dim_out : output size, int

 activation : str

 activation_out : str

 name : str
 """
	if dim_out is None:
		dim_out = dim_in

	model = tf.keras.Sequential(name=name)
	model.add( keras.Input(shape=(dim_in), name=name+f'/input') )
	for l, N in enumerate(N_hid):
		model.add( keras.layers.Dense(N, activation=activation, name=name+f'/dense_{l}') )
	model.add( keras.layers.Dense(dim_out, activation='linear', name=name+f'/output') )
	if activation_out is not None:
		model.add( keras.layers.Dense(dim_out, 
									  activation=activation_out,
									  name=name+f'/output_act') )
	
	return model

def create_disc_mlp_named(N_hid, dim_in,
						  layer_disc = [0],
						  dim_out=None, 
						  activation='relu',
						  name='d_mlp'):
	"""
 Return a discontinuous MLP with specific discontinuous layers and shape
 
 N_hid : number of hidden nodes in each layer, list of int

 dim_in : input size, int

 layer_disc : location(s) of discontinuous layers, list of int

 dim_out : output size, int

 activation : str

 name : str
 
 """
	if dim_out is None:
		dim_out = dim_in

	model = tf.keras.Sequential(name=name)
	model.add( keras.Input(shape=(dim_in), name=name+f'/input') )
	for l, N in enumerate(N_hid):
		if l in layer_disc:
			model.add( DiscontinuityDense(N, activation=activation, name=name+f'/discdense_{l}') )
		else:
			model.add( keras.layers.Dense(N, activation=activation, name=name+f'/dense_{l}') )
	model.add( keras.layers.Dense(dim_out, activation='linear', name=name+f'/output') )
	
	return model


class WGANDiscriminator(tf.Module):
	"""Discriminator network for GAN loss training"""
	def __init__(self, N_hid, dim_in):
		"""
  N_hid : number of hidden nodes in each layer, list of int
  
  dim_in : input size, int
  """
		self.nn = create_mlp_named(N_hid, dim_in, 
								   dim_out=1,
								   activation='relu',
								   activation_out='sigmoid',
								   name='d_nn')
	def __call__(self, x):
		return self.nn(x)
	def loss(self, x_true, x_fake):
		"""WGAN loss"""
		loss_true = -tf.reduce_mean(self.nn(x_true))
		loss_fake = tf.reduce_mean(self.nn(x_fake))
		return loss_true + loss_fake

