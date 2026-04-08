import os

import numpy as np
import tensorflow as tf

from functools import partial
from datetime import datetime as dt
import json

from .model import NormalisingFlowModel, ConditionalFlowModel
from .networks import create_mlp_named, create_disc_mlp_named, WGANDiscriminator
from .distributions import FactorialPrior, ClassDistribution, ContinuousDistribution, CombinedDistribution
from .layers import NiceCouplingLayer, NiceConditionalCouplingLayer, ScalingLayer, NormScaleLayer

####
# Tensor data io

def save_tensor_data(data, 
					 str_suffix='', 
					 dirname=None,
					 fname=None):
	t_now = dt.now()
	if dirname is None:
		dirname = os.path.join('data_tf/',f'runs_{t_now.day}_{t_now.month}/')
	if not os.path.isdir(dirname):
		try:
			os.mkdir(dirname)
		except FileExistsError:
			pass
	if fname is None:
		fname = f'data_{t_now.hour}h{t_now.minute}_' + str_suffix
	t_out = tf.io.serialize_tensor(data)
	tf.io.write_file(os.path.join(dirname,fname), t_out)

def load_tensor_data(fname, dtype=tf.float32):
	t_load = tf.io.read_file(fname)
	return tf.io.parse_tensor(t_load,dtype)

####
# Training datasets

def load_train_data(fname, dim,
					N_train=2**12,
					N_val=2**8):
	data_load = load_tensor_data(fname)
	
	data_train = tf.data.Dataset.from_tensor_slices(data_load[:N_train,:dim])
	data_train = data_train.shuffle(buffer_size=data_train.cardinality(), 
									reshuffle_each_iteration=False)

	if N_val>0:
		data_val = tf.data.Dataset.from_tensor_slices(
			data_load[N_train:N_train+N_val,:dim])
	else:
		data_val = None
	return data_train, data_val

def load_class_data(fname, dim,
					N_train=2**12,
					N_val=2**8):
	data_load = load_tensor_data(fname)
	
	data_train = tf.data.Dataset.from_tensor_slices((
		data_load[:N_train,:dim],
		data_load[:N_train,dim:]
	))
	data_train = data_train.shuffle(buffer_size=data_train.cardinality(), 
									reshuffle_each_iteration=False)

	if N_val>0:
		data_val = tf.data.Dataset.from_tensor_slices((
			data_load[N_train:N_train+N_val,:dim],
			data_load[N_train:N_train+N_val,dim:]
		))
	else:
		data_val = None
	return data_train, data_val

def load_semi_data(fname_l, 
				   fname_u, 
				   dim,
				   N_train = 2**13,
				   N_val = 2**8):
	data_load_unlabel = load_tensor_data(fname_u)
	data_train_unlabel = tf.data.Dataset.from_tensor_slices(
		data_load_unlabel
	)
	data_train_unlabel = data_train_unlabel.shuffle(
		buffer_size=data_train_unlabel.cardinality(), 
		reshuffle_each_iteration=False,
	)
	
	data_load_label = load_tensor_data(fname_l)
	data_train_label = tf.data.Dataset.from_tensor_slices((
		data_load_label[:N_train,:dim],
		data_load_label[:N_train,dim:],
	))
	data_train_label = data_train_label.shuffle(
		buffer_size=data_train_unlabel.cardinality(), 
		reshuffle_each_iteration=False,
	)

	if N_val>0:
		data_val = tf.data.Dataset.from_tensor_slices((
			data_load_label[N_train:N_train+N_val,:dim],
			data_load_label[N_train:N_train+N_val,dim:]
		))
	else:
		data_val = None

	return data_train_label, data_train_unlabel, data_val

####
# Model building and saving

def make_flow_model(
	dirname_model_load = None,
	fname_model_load = None,
	dim = 2,
	d = 1,
	class_labels = False,
	class_dist = 'Categorical',
	class_weights = np.array([]),
	class_min = np.zeros(1),
	class_max = np.ones(1),
	pre_layers = [],
	couple_layers = ['Nice']*4,
	post_layers = ['Scale'],
	network_layers = ['cont']*4,
	N_hid_layers = [[256]*4]*4,
	layer_disc = [3],
	prior_type = 'Normal',
	train_cov = True,
	N_hid_gan = None,
	model_seed = None,
	name = None,
):
	# Number of features in conditional input
	if class_labels:
		if class_dist == 'Categorical':
			N_class = np.array(class_weights).shape[0]
		elif class_dist == 'Uniform':
			N_class = np.array(class_min).shape[0]
		elif class_dist == 'Combined':
			N_class = np.array(class_min).shape[0] + np.array(class_weights).shape[0]
		else:
			raise ValueError(f'Invalid class distribution: {class_dist}')
	else:
		N_class = 0
			
	if fname_model_load is None: # Build model with random weights
		
		# m for each coupling layer
		dim_in = d+N_class
		f_nn_disc = create_disc_mlp_named
		f_nn_cont = create_mlp_named
		m_func_all = []
		for l_type, N_hid in zip(network_layers,N_hid_layers):
			if l_type=='cont':
				m_func_all.append(
					partial(f_nn_cont, N_hid, dim_in,
							  dim_out=dim-d)
				)
			elif l_type=='disc':
				m_func_all.append(
					partial(f_nn_disc, N_hid, dim_in, 
							layer_disc=layer_disc,
							dim_out=dim-d)
				)
			else:
				raise ValueError('Layer type is cont or disc')

		# scaling/normalising layers
		layers = []
		for l_type in pre_layers:
			if l_type == 'Norm':
				layers.append(NormScaleLayer(dim))
			elif l_type == 'Scale':
				layers.append(ScalingLayer(dim))
			else:
				raise ValueError(f'Invalid layer type: {l_type}')

		# coupling layers
		for l, c_type in enumerate(couple_layers):
			if c_type == 'Nice':
				mname = name+f'm_layer{l}' if name is not None else f'm_layer{l}'
				m_layer = m_func_all[l](name=mname)
				layers.append(NiceCouplingLayer(m_layer, d))
				l += 1
			elif c_type == 'ConditionalNice':
				mname = name+f'm_layer{l}' if name is not None else f'm_layer{l}'
				m_layer = m_func_all[l](name=mname)
				layers.append(NiceConditionalCouplingLayer(m_layer, d))
				l += 1
			else:
				raise ValueError(f'Invalid couple layer: {c_type}')

		# scaling/normalising layers
		for l_type in post_layers:
			if l_type == 'Norm':
				layers.append(NormScaleLayer(dim))
			elif l_type == 'Scale':
				layers.append(ScalingLayer(dim))
			else:
				raise ValueError(f'Invalid layer type: {l_type}')

		# prior
		prior = FactorialPrior(dim, distname=prior_type, train_cov=train_cov)

		# distribution of conditional input
		if class_labels:
			if class_dist == 'Categorical':
				class_prior = ClassDistribution(
					N_class,
					class_weights = np.array(class_weights),
				)
			elif class_dist == 'Uniform':
				class_prior = ContinuousDistribution(
					class_min, class_max,
				)
			elif class_dist =='Combined':
				class_prior = CombinedDistribution(
					class_min, class_max, N_class,
					class_weights = np.array(class_weights),
				)
			else:
				raise ValueError(f'Invalid class distribution: {class_dist}')
			model = ConditionalFlowModel(prior, layers, class_prior,
										 seed = model_seed,
										 name = name)
		else:
			# no conditional inputs
			model = NormalisingFlowModel(prior, layers,
										 seed = model_seed,
										 name = name)
	else: # Build model then load saved weights
		print('Loading model ' + dirname_model_load + 'nice' + fname_model_load)
		model, config_load = load_model_json(dirname_model_load,
											 fname_model_load,
											 seed = model_seed)

	# Build discriminator network
	if N_hid_gan is not None:
		dim_disc = dim+N_class
		discriminator = WGANDiscriminator(N_hid_gan, dim_disc)
	else:
		discriminator = None
	
	return model, discriminator

def save_model_json(model, config, dirname, fname):
	"""Save all trainable variables in a named model"""
	if not os.path.isdir(dirname):
		try:
			os.mkdir(dirname)
		except FileExistsError:
			pass
	fname_model = 'nice' + fname + '.json'
	fname_config = 'config' + fname + '.json'
	params = model.get_params()

	params_save = {k: i.tolist() for k,i in params.items()}
	config_save = config.copy()
	for k in config_save['model_config']:
		if type(config_save['model_config'][k])==np.ndarray:
			config_save['model_config'][k] = config_save['model_config'][k].tolist()
	
	with open(os.path.join(dirname,fname_model),'w') as f:
		json.dump(params_save,f)
	with open(os.path.join(dirname,fname_config),'w') as f:
		json.dump(config_save,f)

def load_model_json(dirname, fname, seed=None):
	"""Load a trained model with saved config"""
	fname_model = 'nice' + fname + '.json'
	fname_config = 'config' + fname + '.json'
	with open(os.path.join(dirname,fname_config),'r') as f:
		config = json.load(f)

	model_config = config['model_config']
	model_config['fname_model_load'] = None
	model_config['model_seed'] = seed
	model, _ = make_flow_model(**model_config)
	
	with open(os.path.join(dirname,fname_model),'r') as f:
		data_load = json.load(f)
	for v in model.variables:
		v.assign(data_load[v.name])

	return model, config

