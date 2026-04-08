"""Training with class labels"""

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
	fname_data_label = None,
	fname_data_unlabel = None,
	N_train = int(2**13),
	N_val = int(2**10),
	n_ml = 0,
	lr_ml = 1e-3,
	batch_size_ml = 256,
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
	if fname_data_label is None:
		raise ValueError('fname must be specified for labelled data')
	elif fname_data_unlabel is None:
		print('Loading data ' + fname_data_label)
		data_train_label, data_val = load_class_data(
			fname_data_label,
			model_config['dim'],
			N_train = N_train,
			N_val = N_val
		)
		data_train_unlabel = None
	else:
		print('Loading data ' + fname_data_label + ' ' + fname_data_unlabel)
		data_train_label, data_train_unlabel, data_val = load_semi_data(
			fname_data_label,
			fname_data_unlabel,
			model_config['dim'],
			N_train = N_train,
			N_val = N_val
		)

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
	if data_train_unlabel is not None:
		class_semi_ml_train(model, opt, 
							data_train_label, 
							data_train_unlabel,
							data_val = data_val,
							n_epoch = n_ml,
							batch_size = batch_size_ml,
							f_const_samp = f_const_samp)
	else:
		class_ml_loss_train(model, opt, 
							data_train_label, 
							data_val = data_val, 
							n_epoch = n_ml,
							batch_size = batch_size_ml,
							f_const_samp = f_const_samp)
	t1 = time.time()
	print(f'Time: {t1-t0:.2f}')

	return model, data_train_label, data_train_unlabel

def run_parallel(dirname, fname, model_config, train_config, run_no):
	fname_save = fname + f'_run{run_no}'
	if model_config['model_seed'] is not None:
		model_config['model_seed'] += run_no
	else:
		rng_seed = np.random.default_rng()
		model_config['model_seed'] = int(rng_seed.integers(1e9))
	
	model, _, _ = run_config(
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
	labels = 'semisup' #semisup/fullysup/partsup
	
	if distname=='Metastable':
		dirname_data = os.path.join(PATH_TO_DATA,'Metastable')
		dim = 3
		
		fname_data_label = os.path.join(
			dirname_data,
			'traindata_Metastable_mu_0_0p1'
		)
		fname_data_unlabel = None

		class_dist = 'Uniform'
		class_weights = None
		class_min = np.zeros(2)
		class_max = np.ones(2)
		labels = 'fullysup'
		
	else:
		dirname_data = os.path.join(PATH_TO_DATA,'Labels')
		if distname=='Rastrigin':
			param_str = 'omega_3p5_4p5_theta_0_10_n7'
			dim = 4
			N_class = 3
		elif distname=='Schwefel':
			param_str = 'omega_1100_1800_theta_0_10_n7'
			dim = 4
			N_class = 3
		elif distname=='Whitley':
			param_str = 'omega_0p2_0p3_theta_0_10_n7'
			dim = 4
			N_class = 3
		elif distname=='CR5BP':
			param_str = 'mu_0p001_0p05'
			dim = 3
			N_class = 2
		
		class_dist = 'Categorical'
		class_weights = np.ones(N_class)/N_class
		class_min = None
		class_max = None
			
		if labels == 'fullysup':
			fname_data_label = os.path.join(
				dirname_data,
				f'traindata_fullysup_{distname}_{param_str}'
			)
		else:
			fname_data_label = os.path.join(
				dirname_data,
				f'traindata_label_{distname}_{param_str}'
			)
		if labels == 'semisup':
			fname_data_unlabel = os.path.join(
				dirname_data,
				f'traindata_unlabel_{distname}_{param_str}'
			)
		else:
			fname_data_unlabel = None
	
	
	model_config = dict(
		dim = dim,
		d = 2,
		N_hid_layers = [[512]*3]*4,
		pre_layers = [],
		network_layers = ['cont', 'cont', 'cont', 'cont'],
		couple_layers = ['ConditionalNice']*4,
		post_layers = ['Scale'],
		prior_type = 'Logistic',
		model_seed = None,
		fname_model_load = None,
		class_labels = True,
		class_dist = class_dist,
		class_weights = class_weights,
		class_min = class_min,
		class_max = class_max
	)
	train_config = dict(
		fname_data_label = fname_data_label,
		fname_data_unlabel = fname_data_unlabel,
		N_train = int(2**13),
		N_val = int(2**10),
		n_ml = 100,
		lr_ml = 1e-3,
		batch_size_ml = 256,
		sample_constraint = True,
		ind_constraint = [i for i in range(dim)],
		min_constraint = [-1.]*dim,
		max_constraint = [1.]*dim,
	)
	
	t_now = dt.now()
	dirname_all = 'Metastable' if distname=='Metastable' else 'Labels'
	fname_all = f"_{t_now.month}_{t_now.day}_{t_now.hour}h{t_now.minute}_{distname}_{labels}"

	f_run = partial(run_parallel, 
					dirname_all,
					fname_all,
					model_config,
					train_config)

	with Pool(processes=n_process) as p:
		done = p.map(f_run, range(n_run))
