"""Training with uncertain parameters"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentdir)

from functools import partial
from inspect import signature
from multiprocessing import Pool
from datetime import datetime as dt
import time
import pdb

import numpy as np
from tensorflow import keras

from tfFlow.training import *
from tfFlow.utils import *

def save_model_config(model, 
					  model_config, 
					  train_config,
					  fname,
					  dirname = ''):
	sig_model = signature(make_flow_model)
	mconfig_save = {
		v.name: v.default for v in sig_model.parameters.values() if np.all(v.default != v.empty)
	}
	for p,v in model_config.items():
		mconfig_save[p] = v
	sig_train = signature(run_config)
	config_save = {
		v.name: v.default for v in sig_train.parameters.values() if np.all(v.default != v.empty)
	}
	for p,v in train_config.items():
		config_save[p] = v
	config_save['model_config'] = mconfig_save
	
	dirname_save = os.path.join(PATH_TO_MODELS,dirname)
	save_model_json(model, config_save, dirname_save, fname)

def run_config(
	model_config = {},
	fname_data_load = None,
	N_train = int(2**13),
	N_val = int(2**8),
	n_ml = 10,
	lr_ml = 1e-3,
	batch_size_ml = int(2**8),
	sample_constraint = False,
	ind_constraint = [],
	min_constraint = [],
	max_constraint = [],
):
	####
	# Build model
	keras.backend.clear_session()
	model, discriminator = make_flow_model(**model_config)

	####
	# Create Training Data 
	if fname_data_load is None:
		raise ValueError('fname must be specified for data')
	else:
		print('Loading data ' + fname_data_load)
		if not model_config['class_labels']:
			data_train, data_val = load_train_data(fname_data_load,
												   model_config['dim'],
												   N_train = N_train,
												   N_val = N_val)
		else:
			data_train, data_val = load_class_data(fname_data_load,
												   model_config['dim'],
												   N_train = N_train,
												   N_val = N_val)
	
	####
	# Train
	if sample_constraint:
		obj_const_samp = SampleConstraint(
			ind_constraint, min_constraint, max_constraint
		)
		f_const_samp = obj_const_samp.sample_constraint
	else:
		f_const_samp = no_samp_constraint

	print('Training ML')
	opt = keras.optimizers.Adam(learning_rate=lr_ml)
	t0 = time.time()
	if not model_config['class_labels']:
		ml_loss_train(model, opt, data_train, 
					  data_val = data_val, 
					  n_epoch = n_ml,
					  batch_size = batch_size_ml,
					  f_const_samp = f_const_samp)
	else:
		class_ml_loss_train(model, opt, data_train, 
							data_val = data_val, 
							n_epoch = n_ml,
							batch_size = batch_size_ml,
							f_const_samp = f_const_samp)
	t1 = time.time()
	print(f'Time: {t1-t0:.2f}')
		
	return model, data_train

def run_parallel(dirname, fname, model_config, train_config, run_no):
	fname_save = fname + f'_run{run_no}'
	if model_config['model_seed'] is not None:
		model_config['model_seed'] += run_no
	else:
		rng_seed = np.random.default_rng()
		model_config['model_seed'] = int(rng_seed.integers(1e9))
	
	model, _ = run_config(
		model_config=model_config,
		**train_config,
	)
	save_model_config(
		model,
		model_config,
		train_config,
		fname_save,
		dirname = dirname,
	)
	del model

	return True

if __name__=='__main__':
	n_run = 8
	n_process = np.min([8, n_run])

	PATH_TO_DATA = os.path.join(parentdir,'DataESGenepy')
	PATH_TO_MODELS = os.path.join(parentdir,'ModelsESGenepy')
	
	distname = 'Rastrigin'
	
	dirname_data = os.path.join(PATH_TO_DATA,'Parameters')
	if distname=='Rastrigin':
		param_str = 'omega_3p5_4p5_theta_0_10_n7'
		N_class = 2
		N_train = int(2**13)
	elif distname=='Schwefel':
		param_str = 'omega_1100_1800_theta_0_10_n7'
		N_class = 2
		N_train = int(2**13)
	elif distname=='Whitley':
		param_str = 'omega_0p2_0p3_theta_0_10_n7'
		N_class = 2
		N_train = int(2**14)
	elif distname=='CR5BP':
		param_str = 'mu_0p005_0p995_n9'
		N_class = 1
		N_train = int(2**13)
	fname_data_load = os.path.join(
		dirname_data,
		f'traindata_{distname}_{param_str}'
	)
	
	
	model_config = dict(
		dim = 2,
		d = 1,
		N_hid_layers = [[512]*3]*4,
		pre_layers = [],
		network_layers = ['cont']*4,
		couple_layers = ['ConditionalNice']*4,
		post_layers = ['Scale'],
		prior_type = 'Logistic',
		model_seed = None,
		fname_model_load = None,
		class_labels = True,
		class_dist = 'Uniform',
		class_min = -np.ones(N_class),
		class_max = np.ones(N_class)
	)
	train_config = dict(
		fname_data_load = fname_data_load,
		N_train = N_train,
		n_ml = 500,
		lr_ml = 1e-3,
		batch_size_ml = 256,
		sample_constraint = False,
	)
	
	t_now = dt.now()
	dirname_all = 'Parameters'
	fname_all = f"_{t_now.month}_{t_now.day}_{t_now.hour}h{t_now.minute}_{distname}_parameters"

	f_run = partial(run_parallel, 
					dirname_all,
					fname_all,
					model_config,
					train_config)

	with Pool(processes=n_process) as p:
		done = p.map(f_run, range(n_run))
