"""Iterative retraining for finding equilbrium points"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentdir)

from tqdm import trange
from functools import partial
from inspect import signature
from multiprocessing import Pool
from datetime import datetime as dt
import time
import json
import pdb

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.optimize import root

from tfFlow.training import *
from tfFlow.utils import *

from potentialdist import get_crit_cr5bp, get_root_fns
from optdist import MinNormalDist


####
# Local optimisation

def run_single_root(x0, f_root, df_root,
				   r_t = 1e-6,
				   xmax = 1.,
				   tol = 1e-8,
				   min_tol = 1e-6,
				   min_update = [],
				   res_min = [],
				   res_nfev = [],
				   res_succ = [],
				   new_min = [],
				   method = 'lm',
				  ):
	"""Run local root optimisation starting from x0"""
	filt_bool = [np.linalg.norm(x0-m)>r_t for m in min_update]
	if not all(filt_bool):
		return
	res = root(f_root, x0, jac=df_root, tol=tol, method=method)
	res_min.append(res.x)
	res_nfev.append(res.nfev)
	res_succ.append(res.success)
	if not res.success or any(np.abs(res.x)>xmax):
		new_min.append(0)
	else:
		found_bool = [np.linalg.norm(res.x-m)>min_tol for m in min_update]
		new_min.append( int(all(found_bool)) )
		if all(found_bool):
			min_update.append(res.x)

####
# Sample minimisation

def random_nfev_potential_data(mu, r_t, min_true,
							   distname = 'CR5BP',
							   use_hess = True,
							   xmax = 1.,
							   tol = 1e-8,
							   min_tol = 1e-6,
							   N_seed = 32,
							   seed = None):
	"""Finding all roots from random uniform samples"""
	dist, f_root, df_root = get_root_fns(distname, mu, use_hess=use_hess)
	
	rng_seed = np.random.default_rng(seed=seed)
	data_all = []
	for seed_no in trange(N_seed):
		seed = rng_seed.integers(1e9)
		rng = np.random.default_rng(seed=seed)
		min_update = []
		res_min = []
		res_nfev = []
		res_succ = []
		new_min = []
		while len(min_update)<min_true.shape[0]:
			x0 = rng.uniform(low=[-xmax,-xmax],high=[xmax,xmax])
			run_single_root(x0, f_root, df_root,
							r_t = r_t,
							xmax = xmax,
							tol = tol,
							min_tol = min_tol,
							min_update = min_update,
							res_min = res_min,
							res_nfev = res_nfev,
							res_succ = res_succ,
							new_min = new_min,
							method = 'lm')
		data_all.append({
			'seed': seed,
			'min': res_min,
			'nfev': res_nfev,
			'success': res_succ,
			'newmin': new_min
		})
	return data_all

def model_samp_roots(model, N_samp, r_t, mu,
					 distname = 'CR5BP',
					 use_hess = True,
					 xmax = 1.,
					 min_update = [],
					 tol = 1e-8,
					 min_tol = 1e-6):
	"""Sample model and test distance in physical space"""
	dist, f_root, df_root = get_root_fns(distname, mu, use_hess=use_hess)
	
	x0_samp = model.sample_inv(N_samp).numpy()
	x0_filt = x0_samp[np.all(np.abs(x0_samp[:,:2])<=1.,axis=-1),:]
	
	res_min = []
	res_nfev = []
	res_succ = []
	new_min = []
	print(f'N xfilt: {x0_filt.shape[0]}')
	for x0 in x0_filt[:,:2]:
		run_single_root(x0, f_root, df_root,
						r_t = r_t,
						xmax = xmax,
						tol = tol,
						min_tol = min_tol,
						min_update = min_update,
						res_min = res_min,
						res_nfev = res_nfev,
						res_succ = res_succ,
						new_min = new_min,
						method = 'lm')
	del dist
	results_min = {
		'x0': x0_filt[:,:2],
		'min': res_min,
		'nfev': res_nfev,
		'success': res_succ,
		'newmin': new_min,
	}
	return results_min, min_update

####
# Creating training data

def get_root_training_data(min_retrain, mu,
						  distname = 'CR5BP',
						  func_vals = True,
						  N_train = 2**13,
						  sigma = None,
						  seed = None):
	"""Training data from roots"""
	rng = np.random.default_rng(seed=seed)
	if sigma is None:
		sigma = 1e-3

	if func_vals:
		dist, _, _ = get_root_fns(distname, mu)
		fval_norm = dist.fmin_norm(tf.constant(min_retrain,dtype=tf.float64)).numpy()
		min_fval = np.hstack([min_retrain,fval_norm.reshape(-1,1)])
		min_samp = rng.choice(min_fval, size=N_train)
		loc_samp = [0., 0., 0.]
		scale_samp = [sigma, sigma, 0.]
	else:
		min_samp = rng.choice(min_retrain, size=N_train)
		loc_samp = [0., 0.]
		scale_samp = [sigma, sigma]
	
	x_samp = min_samp + rng.normal(loc=loc_samp,
								   scale=scale_samp,
								   size=min_samp.shape)
	filt_x = np.any(np.abs(x_samp[:,:2])>1.,axis=-1)
	x_samp[filt_x,:] = rng.choice(x_samp[~filt_x,:], size=np.sum(filt_x))

	x_samp_tf = tf.constant(x_samp, dtype=tf.float32)
	data_train = tf.data.Dataset.from_tensor_slices(x_samp_tf)
	data_train = data_train.shuffle(buffer_size=data_train.cardinality(), 
									reshuffle_each_iteration=False)
	return data_train

def get_root_init_data(N_init, r_t, mu,
					  distname = 'CR5BP',
					  use_hess = True,
					  xmax = 1.,
					  func_vals = True,
					  sigma = None,
					  N_train = int(2**13),
					  seed = None,
					  tol = 1e-8,
					  min_tol = 1e-6):
	"""Training data from roots from initial random samples"""
	rng = np.random.default_rng(seed=seed)
	dist, f_root, df_root = get_root_fns(distname, mu, use_hess=use_hess)
	
	min_update = []
	x0_samp = []
	res_min = []
	res_nfev = []
	res_succ = []
	new_min = []
	while len(min_update)<N_init:
		x0 = rng.uniform(low=[-xmax,-xmax],high=[xmax,xmax])
		x0_samp.append(x0)
		run_single_root(x0, f_root, df_root,
						r_t = r_t,
						xmax = xmax,
						tol = tol,
						min_tol = min_tol,
						min_update = min_update,
						res_min = res_min,
						res_nfev = res_nfev,
						res_succ = res_succ,
						new_min = new_min,
						method = 'lm')
	results_init = {
		'seed': seed,
		'x0': x0_samp,
		'min': res_min,
		'nfev': res_nfev,
		'success': res_succ,
		'newmin': new_min,
	}

	min_retrain = np.array(min_update)
	data_train = get_root_training_data(
		min_retrain, mu,
		distname = distname,
		func_vals = func_vals,
		N_train = N_train,
		sigma = sigma,
		seed = seed)
	
	return data_train, min_retrain, results_init

####
# Model saving

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
	sig_train = signature(run_pot_retrain_config)
	config_save = {
		v.name: v.default for v in sig_train.parameters.values() if np.all(v.default != v.empty)
	}
	for p,v in train_config.items():
		config_save[p] = v
	config_save['model_config'] = mconfig_save

	dirname_save = os.path.join(PATH_TO_MODELS,dirname)
	save_model_json(model, config_save, dirname_save, fname)

####
# Retraining loop

def run_pot_retrain_config(
	dirname_save, 
	fname_save, 
	model_config = {},
	N_init = 8,
	seed_init = None,
	distname = 'CR5BP',
	mu = 0.02,
	xmax = 1.5,
	func_vals = False,
	N_train = int(2**13),
	N_val = int(2**8),
	N_samp = int(2**10),
	sigma_sample = 1e-3,
	r_t = 6e-3,
	opt_tol = 1e-6,
	min_tol = 1e-4,
	use_hess = True,
	nfev_max = 10000,
	max_iter = 20,
	n_ml = 16,
	lr_ml = 1e-3,
	batch_size_ml = int(2**8),
	dist_loss = True,
	save_models = False,
	sample_constraint = False,
	ind_constraint = [],
	min_constraint = [],
	max_constraint = [],
):
	####
	# Build model
	keras.backend.clear_session()
	model, _ = make_flow_model(**model_config)
	
	####
	# Create Training Data
	dirname_data = os.path.join(PATH_TO_DATA,dirname_all)
	data_train, min_init, results_init = get_root_init_data(
		N_init, r_t, mu,
		distname = distname,
		use_hess = use_hess,
		xmax = xmax,
		func_vals = func_vals,
		sigma = sigma_sample,
		N_train = N_train,
		seed = seed_init,
		tol = opt_tol,
		min_tol = min_tol,
	)
	fname_init = 'mindata' + fname_save + '_init.json'
	if not os.path.isdir(dirname_data):
		try:
			os.mkdir(dirname_data)
		except FileExistsError:
			pass
	data_init_save = results_init.copy()
	data_init_save['x0'] = [d.tolist() for d in data_init_save['x0']]
	data_init_save['min'] = [d.tolist() for d in data_init_save['min']]
	with open(os.path.join(dirname_data,fname_init),'w') as f:
		json.dump(data_init_save, f)
	
	if distname=='CR5BP':
		min_all = get_crit_cr5bp(1., mu_min=0., mu_max=mu)
	else:
		raise ValueError(f'Invalid distribution name: {distname}')
	N_min = min_all.shape[0]
	min_update = [m for m in min_init]
	
	if sample_constraint:
		obj_const_samp = SampleConstraint(
			ind_constraint, min_constraint, max_constraint
		)
		f_const_samp = obj_const_samp.sample_constraint
	else:
		f_const_samp = no_samp_constraint

	####
	# Initial Training
	print('Initial ML Training')
	opt = keras.optimizers.Adam(learning_rate=lr_ml)
	if dist_loss:
		dist = MinNormalDist(min_all, sigma=sigma_sample)
		t0 = time.time()
		ml_dist_loss_train(model, opt, data_train, dist.ux,
					  data_val = None, 
					  n_epoch = n_ml,
					  batch_size = batch_size_ml,
					  f_const_samp = f_const_samp)
		t1 = time.time()
	else:
		t0 = time.time()
		ml_loss_train(model, opt, data_train, 
					  data_val = None, 
					  n_epoch = n_ml,
					  batch_size = batch_size_ml,
					  f_const_samp = f_const_samp)
		t1 = time.time()
	print(f'Initial train time: {t1-t0:.2f}')
	del data_train

	if save_models:
		fname_init = fname_save + '_init'
		save_model_config(
			model, 
			model_config, 
			{}, 
			fname_init,
			dirname=dirname_save
		)

	####
	# Iterative Training
	data_min = {
		'opt': [],
		'min': [min_init],
	}
	n_prev = len(min_update)
	nfev_total = 0
	for i in range(max_iter):
		results_min, min_update = model_samp_roots(
			model, N_samp, r_t, mu,
			distname = distname,
			use_hess = use_hess,
			xmax = xmax,
			min_update = min_update,
			tol = opt_tol,
			min_tol = min_tol)
		min_retrain = np.array(min_update)
		print(f'Retrain iter {i} nfev {nfev_total} min prev: {n_prev} min new: {len(min_update)}')
		
		data_train =  get_root_training_data(
			min_retrain, mu,
			distname = distname,
			func_vals = func_vals,
			N_train = N_train,
			sigma = sigma_sample)
		
		data_min['opt'].append(results_min)
		data_min['min'].append(min_retrain)
		
		nfev_total += np.sum(results_min['nfev'])
		if nfev_total >= nfev_max:
			print('Exceeded max nfev')
			break
		elif len(min_update)>=N_min:
			print('Found all minima')
			break

		print('ML Retraining')
		if dist_loss:
			t0 = time.time()
			ml_dist_loss_train(model, opt, data_train, dist.ux,
						  data_val = None, 
						  n_epoch = n_ml,
						  batch_size = batch_size_ml,
						  f_const_samp = f_const_samp)
			t1 = time.time()
		else:
			t0 = time.time()
			ml_loss_train(model, opt, data_train, 
						  data_val = None, 
						  n_epoch = n_ml,
						  batch_size = batch_size_ml,
						  f_const_samp = f_const_samp)
			t1 = time.time()
		print(f'Training time: {t1-t0:.0f}s')
		del data_train
		n_prev = len(min_update)

		if save_models:
			fname_init = fname_save + f'_i{i}'
			save_model_config(
				model, 
				model_config, 
				{}, 
				fname_init,
				dirname=dirname_save
			)

	return model, data_min

def run_parallel(dirname_save, fname, model_config, train_config, run_no):
	dirname_data = os.path.join(PATH_TO_DATA,dirname_save)
	fname_save = fname + f'_run{run_no}'
	
	model, data_min = run_pot_retrain_config(
		dirname_save,
		fname_save,
		model_config=model_config,
		**train_config,
	)

	fname_data = 'mindata' + fname_save + '.json'
	if not os.path.isdir(dirname_data):
		try:
			os.mkdir(dirname_data)
		except FileExistsError:
			pass

	data_min_save = data_min.copy()
	data_min_save['min'] = [d.tolist() for d in data_min_save['min']]
	for d in data_min_save['opt']:
		d['x0'] = d['x0'].tolist()
		d['min'] = [m.tolist() for m in d['min']]
	with open(os.path.join(dirname_data,fname_data),'w') as f:
		json.dump(data_min_save, f)
	
	save_model_config(
		model,
		model_config,
		train_config,
		fname_save,
		dirname = dirname_save,
	)
	del model, data_min

	return True

if __name__=='__main__':
	n_run = 32
	n_process = np.min([8, n_run])

	PATH_TO_DATA = os.path.join(parentdir,'DataESGenepy')
	PATH_TO_MODELS = os.path.join(parentdir,'ModelsESGenepy')

	distname = 'CR5BP'
	mu = 0.992
	func_vals = True
	xmax = 1.3
	use_hess = True
	N_init = 4
	sigma_sample = 1e-3
	r_t = 6*sigma_sample

	if func_vals:
		ind_constraint = [0,1,2],
		min_constraint = [-xmax,-xmax,-1.],
		max_constraint = [xmax,xmax,1.]
	else:
		ind_constraint = [0,1],
		min_constraint = [-xmax,-xmax],
		max_constraint = [xmax,xmax]
	
	model_config = dict(
		dim = 2 + int(func_vals),
		d = 1 + int(func_vals),
		N_hid_layers = [[512]*3]*4,
		pre_layers = [],
		network_layers = ['cont', 'cont', 'cont', 'cont'],
		post_layers = ['Scale'],
		prior_type = 'Logistic',
		train_cov = True,
		model_seed = None,
		fname_model_load = None,
		class_labels = False,
	)
	train_config = dict(
		N_init = N_init,
		seed_init = None,
		max_iter = 100,
		nfev_max = int(1e6),
		distname = distname,
		mu = mu,
		xmax = xmax,
		func_vals = func_vals,
		use_hess = use_hess,
		opt_tol = 1e-8,
		min_tol = 1e-3,
		r_t = r_t,
		sigma_sample = sigma_sample,
		n_ml = 16,
		N_samp = 512,
		dist_loss = True,
		save_models = False,
		sample_constraint = True,
		ind_constraint = ind_constraint,
		min_constraint = min_constraint,
		max_constraint = max_constraint
	)
	t_now = dt.now()
	dirname_all = 'Retraining'

	mu_str = str(mu).replace('.','p').rstrip('0').rstrip('p')
	fval_str = 'fvals' if func_vals else 'xonly'
	fname_all = f'_{t_now.month}_{t_now.day}_{t_now.hour}h{t_now.minute}_{distname}_mu{mu_str}_{fval_str}'

	f_run = partial(run_parallel, 
					dirname_all,
					fname_all,
					model_config,
					train_config)

	with Pool(processes=n_process) as p:
		done = p.map(f_run, range(n_run))
