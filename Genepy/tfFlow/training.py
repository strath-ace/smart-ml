"""Training loops"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import numpy as np

import pdb


################################
## Loss functions
################################

def nice_loss(l):
	"""Loss related to log-likelihood"""
	return -tf.reduce_mean(l)

def dist_loss(z, A, inds):
	"""Loss of distances in latent space"""
	return 1.

class SampleConstraint(object):
	"""Losses penalising samples outwith bounds"""
	def __init__(self, inds, min_val, max_val):
		self.inds = tf.constant(inds, dtype=tf.int32)
		self.min_val = tf.constant(min_val,dtype=tf.float32)
		self.max_val = tf.constant(max_val,dtype=tf.float32)
	def sample_constraint(self, x_samp):
		X_inds = tf.gather(x_samp,self.inds,axis=1)
		l_lower = tf.reduce_mean(
			tf.reduce_sum(tf.nn.relu(self.min_val - X_inds)**2,axis=-1)
		)
		l_upper = tf.reduce_mean(
			tf.reduce_sum(tf.nn.relu(X_inds - self.max_val)**2,axis=-1)
		)
		return l_lower + l_upper

def no_samp_constraint(x_samp):
	"""No constraint on model samples"""
	return tf.zeros(1,dtype=tf.float32)

################################
## Training loops
################################

def ml_loss_train(model,
				  opt,
				  data_train, 
				  data_val=None,
				  n_epoch=1000,
				  batch_size=256,
				  f_const_samp=no_samp_constraint,
				  ):
	"""Training loop for only ML loss"""
	dataloader = data_train.batch(batch_size)
	if data_val is not None:
		val_tf = iter(data_val.batch(data_val.cardinality())).next()

	@tf.function
	def ml_train_step(inputs):
		with tf.GradientTape() as tape:
			x_samp = model.sample_inv(batch_size)
			l = model.log_likelihood(inputs)
			loss = nice_loss(l) + f_const_samp(x_samp)
		gradient = tape.gradient(loss,model.trainable_variables)
		opt.apply_gradients(zip(gradient,model.trainable_variables))

	@tf.function
	def val_step():
		return -tf.reduce_mean(model.log_likelihood(val_tf))

	for epoch in tf.range(n_epoch):
		tf.print('Epoch',epoch)
		for inputs in dataloader:
			ml_train_step(inputs)
		
		if data_val is not None:
			loss_val = val_step()
			tf.print('Validation loss',loss_val)

def ml_dist_loss_train(model,
					   opt,
					   data_train, 
					   u_func,
					   dim_dist=2,
					   data_val=None,
					   n_epoch=1000,
					   batch_size=256,
					   f_const_samp=no_samp_constraint,
					  ):
	"""Training loop for ML and EEL loss"""
	dataloader = data_train.batch(batch_size)
	if data_val is not None:
		val_tf = iter(data_val.batch(data_val.cardinality())).next()
	d = tf.constant(dim_dist)

	@tf.function
	def ml_train_step(inputs):
		ll_x = tf.cast(-u_func(tf.cast(inputs[:,:d],tf.float64)),tf.float32)
		with tf.GradientTape() as tape:
			x_samp = model.sample_inv(batch_size)
			ll_model = model.log_likelihood(inputs)
			eel = tf.reduce_mean( (ll_model-ll_x - tf.reduce_mean(ll_model-ll_x))**2. )
			l = model.log_likelihood(inputs)
			loss = nice_loss(l) + f_const_samp(x_samp) + eel
		gradient = tape.gradient(loss,model.trainable_variables)
		opt.apply_gradients(zip(gradient,model.trainable_variables))

	@tf.function
	def val_step():
		return -tf.reduce_mean(model.log_likelihood(val_tf))

	for epoch in tf.range(n_epoch):
		tf.print('Epoch',epoch)
		for inputs in dataloader:
			ml_train_step(inputs)
		
		if data_val is not None:
			loss_val = val_step()
			tf.print('Validation loss',loss_val)

def class_ml_loss_train(model,
						opt,
						data_train, 
						data_val=None,
						n_epoch=1000,
						batch_size=256,
						f_const_samp=no_samp_constraint,
				  ):
	"""Training loop for only 'ML' loss with classes"""
	dataloader = data_train.batch(batch_size)
	if data_val is not None:
		val_tf = iter(data_val.batch(data_val.cardinality())).next()

	@tf.function
	def ml_train_step(inputs, labels):
		with tf.GradientTape() as tape:
			x_samp, _ = model.sample_inv(batch_size)
			l = model.log_likelihood(inputs, labels)
			loss = nice_loss(l) + f_const_samp(x_samp)
		gradient = tape.gradient(loss,model.trainable_variables)
		opt.apply_gradients(zip(gradient,model.trainable_variables))

	@tf.function
	def val_step():
		return -tf.reduce_mean(model.log_likelihood(*val_tf))

	for epoch in tf.range(n_epoch):
		tf.print('Epoch',epoch)
		for inputs, labels in dataloader:
			ml_train_step(inputs, labels)
		
		if data_val is not None:
			loss_val = val_step()
			tf.print('Validation loss',loss_val)

def class_semi_ml_train(model,
						opt,
						data_train_label,
						data_train_unlabel,
						data_val=None,
						n_epoch=1000,
						batch_size=256,
						f_const_samp=no_samp_constraint):
	"""Training loop for ML loss with semi-supervised classification"""
	dataloader_l = data_train_label.batch(batch_size)
	dataloader_u = data_train_unlabel.batch(batch_size)
	if data_val is not None:
		val_tf = iter(data_val.batch(data_val.cardinality())).next()

	@tf.function
	def ml_train_step(inputs_l, labels, inputs_u):
		with tf.GradientTape() as tape:
			x_samp, _ = model.sample_inv(batch_size)
			ll_l = model.log_likelihood(inputs_l, labels)
			ll_u = model.max_log_likelihood(inputs_u)
			loss = nice_loss(ll_l) + nice_loss(ll_u) + f_const_samp(x_samp)
		gradient = tape.gradient(loss,model.trainable_variables)
		opt.apply_gradients(zip(gradient,model.trainable_variables))

	@tf.function
	def val_step():
		return -tf.reduce_mean(model.log_likelihood(*val_tf))
	
	for epoch in tf.range(n_epoch):
		tf.print('Epoch',epoch)
		for (inputs_l, labels), inputs_u in zip(dataloader_l, dataloader_u):
			ml_train_step(inputs_l, labels, inputs_u)

		if data_val is not None:
			loss_val = val_step()
			tf.print('Validation loss',loss_val)

def gan_train(model,
			  discriminator,
			  opt_g,
			  opt_d,
			  data_train,
			  data_val=None,
			  l_reg_gan=1.,
			  n_critic=5,
			  n_epoch=1000,
			  batch_size=256,
			  f_const_samp=no_samp_constraint,
			 ):
	"""Training loop for GAN loss"""
	if data_val is not None:
		val_tf = iter(data_val.batch(data_val.cardinality())).next()
	N_train = data_train.cardinality()
	n_repeat = tf.cast(tf.math.ceil(n_epoch*batch_size/N_train),tf.int64)
	discdataloader = data_train.shuffle(
		data_train.cardinality(),
		reshuffle_each_iteration=True
	).repeat(n_repeat).batch(batch_size)
	gendataloader = data_train.batch(batch_size)
	l_reg_gan = tf.constant(l_reg_gan,dtype=tf.float32)

	@tf.function
	def crit_train_step(x_true, x_fake):
		with tf.GradientTape() as tape:
			loss = discriminator.loss(x_true, x_fake)
		gradient = tape.gradient(loss,discriminator.trainable_variables)
		opt_d.apply_gradients(zip(gradient,discriminator.trainable_variables))

	@tf.function
	def gen_train_step(inputs):
		with tf.GradientTape() as tape:
			x_samp = model.sample_inv(batch_size)
			l = model.log_likelihood(inputs)
			gen_loss = nice_loss(l)
			disc_loss = - tf.reduce_mean(discriminator(x_samp))/l_reg_gan
			const_loss = f_const_samp(x_samp)
			loss = gen_loss + disc_loss + const_loss
		gradient = tape.gradient(loss,model.trainable_variables)
		opt_g.apply_gradients(zip(gradient,model.trainable_variables))

	@tf.function
	def val_step():
		return -tf.reduce_mean(model.log_likelihood(val_tf))
	
	for epoch, x_true in enumerate(discdataloader):
		if epoch >= n_epoch:
			break
		tf.print('Epoch',epoch)
		x_fake = model.sample_inv(batch_size)
		for _ in tf.range(n_critic):
			crit_train_step(x_true, x_fake)
		
		for inputs in gendataloader:
			gen_train_step(inputs)

		if data_val is not None:
			loss_val = val_step()
			tf.print('Validation loss',loss_val)

def class_gan_train(model,
					discriminator,
					opt_g,
					opt_d,
					data_train,
					l_reg_gan=1.,
					n_critic=5,
					n_epoch=1000,
					batch_size=256,
					f_const_samp=no_samp_constraint):
	"""Training loop for GAN loss with classification"""
	N_train = data_train.cardinality()
	n_repeat = tf.cast(tf.math.ceil(n_epoch*batch_size/N_train),tf.int64)
	discdataloader = data_train.shuffle(
		data_train.cardinality(),
		reshuffle_each_iteration=True
	).repeat(n_repeat).batch(batch_size)
	gendataloader = data_train.batch(batch_size)
	l_reg_gan = tf.constant(l_reg_gan,dtype=tf.float32)

	@tf.function
	def crit_train_step(x_label_true, x_label_fake):
		with tf.GradientTape() as tape:
			loss = discriminator.loss(x_label_true, x_label_fake)
		gradient = tape.gradient(loss,discriminator.trainable_variables)
		opt_d.apply_gradients(zip(gradient,discriminator.trainable_variables))

	@tf.function
	def gen_train_step(inputs, labels):
		with tf.GradientTape() as tape:
			x_samp, c_samp = model.sample_inv(batch_size)
			x_label_samp = tf.concat([x_samp,c_samp],axis=-1)
			l = model.log_likelihood(inputs, labels)
			gen_loss = nice_loss(l)
			disc_loss = - tf.reduce_mean(discriminator(x_label_samp))/l_reg_gan
			const_loss = f_const_samp(x_samp)
			loss = gen_loss + disc_loss + const_loss
		gradient = tape.gradient(loss,model.trainable_variables)
		opt_g.apply_gradients(zip(gradient,model.trainable_variables))
	
	for epoch, (x_true, c_true) in enumerate(discdataloader):
		if epoch >= n_epoch:
			break
		tf.print('Epoch',epoch)
		x_label_true = tf.concat([x_true,c_true],axis=-1)
		x_fake, c_fake = model.sample_inv(batch_size)
		x_label_fake = tf.concat([x_fake,c_fake],axis=-1)
		for _ in tf.range(n_critic):
			crit_train_step(x_label_true, x_label_fake)
		
		for inputs, labels in gendataloader:
			gen_train_step(inputs, labels)

def class_semi_gan_train(model,
						 discriminator,
						 opt_g,
						 opt_d,
						 data_train_label,
						 data_train_unlabel,
						 l_reg_gan=1.,
						 n_critic=5,
						 n_epoch=1000,
						 batch_size=256,
						 f_const_samp=no_samp_constraint):
	"""Training loop for GAN loss with semi-supervised classification"""
	N_train = data_train_label.cardinality()
	n_repeat = tf.cast(tf.math.ceil(n_epoch*batch_size/N_train),tf.int64)
	discdataloader = data_train_label.shuffle(
		data_train_label.cardinality(),
		reshuffle_each_iteration=True
	).repeat(n_repeat).batch(batch_size)
	gendataloader_l = data_train_label.batch(batch_size)
	gendataloader_u = data_train_unlabel.batch(batch_size)
	l_reg_gan = tf.constant(l_reg_gan,dtype=tf.float32)

	@tf.function
	def crit_train_step(x_label_true, x_label_fake):
		with tf.GradientTape() as tape:
			loss = discriminator.loss(x_label_true, x_label_fake)
		gradient = tape.gradient(loss,discriminator.trainable_variables)
		opt_d.apply_gradients(zip(gradient,discriminator.trainable_variables))

	@tf.function
	def gen_train_step(inputs_l, labels, inputs_u):
		with tf.GradientTape() as tape:
			x_samp, c_samp = model.sample_inv(batch_size)
			x_label_samp = tf.concat([x_samp,c_samp],axis=-1)
			ll_l = model.log_likelihood(inputs_l, labels)
			gen_loss_l = nice_loss(ll_l)
			ll_u = model.max_log_likelihood(inputs_u)
			gen_loss_u = nice_loss(ll_u)
			disc_loss = - tf.reduce_mean(discriminator(x_label_samp))/l_reg_gan
			const_loss = f_const_samp(x_samp)
			loss = gen_loss_l + gen_loss_u + disc_loss + const_loss
		gradient = tape.gradient(loss,model.trainable_variables)
		opt_g.apply_gradients(zip(gradient,model.trainable_variables))
	
	for epoch, (x_true, c_true) in enumerate(discdataloader):
		if epoch >= n_epoch:
			break
		tf.print('Epoch',epoch)
		x_label_true = tf.concat([x_true,c_true],axis=-1)
		x_fake, c_fake = model.sample_inv(batch_size)
		x_label_fake = tf.concat([x_fake,c_fake],axis=-1)
		for _ in tf.range(n_critic):
			crit_train_step(x_label_true, x_label_fake)
		
		for (inputs_l, labels), inputs_u in zip(gendataloader_l, gendataloader_u):
			gen_train_step(inputs_l, labels, inputs_u)

