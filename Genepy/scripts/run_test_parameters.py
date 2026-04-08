"""Testing with uncertain parameters"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentdir)

from tqdm import trange
from functools import partial
from itertools import product
from multiprocessing import Pool
from datetime import datetime as dt
import json
import pdb

import numpy as np
import tensorflow as tf

from tfFlow.utils import *

from potentialdist import get_crit_cr5bp, get_root_fns
from optdist import get_min_rot_rast, get_min_rot_schwefel, get_min_rot_whitley, get_opt_fns

from run_potential_retrain import run_single_root, random_nfev_potential_data
from run_retrain import run_single_opt, random_nfev_rot_data

####
# Sample minimisation and roots

def model_test_samp_min(
	model,
	omega_norm, theta_norm, 
	omrange, thetarange,
	r_t, dx,
	min_true,
	distname = 'Rastrigin',
	use_hess = True,
	xmax = 1.,
	tol = 1e-8,
	min_tol = 1e-6,
	N_seed = 32
):
	"""Sample from model and run local minimisation until all minima found"""
	dx = np.zeros(2) if dx is None else dx
	omega = omrange[0] + 0.5*(omega_norm+1)*(omrange[1]-omrange[0])
	theta = thetarange[0] + 0.5*(theta_norm+1)*(thetarange[1]-thetarange[0])
	
	dist, f_min, df_min, ddf_min, method = get_opt_fns(
		distname, omega, theta*np.pi/180., dx, 
		use_hess=use_hess)
	c = tf.constant([[omega_norm, theta_norm]],dtype=tf.float32)
	data_all = []
	for seed_no in trange(N_seed):
		min_update = []
		res_min = []
		res_nfev = []
		res_succ = []
		new_min = []
		x0_all = []
		while len(min_update)<min_true.shape[0]:
			x0 = model.sample_inv_class(1, c).numpy().flatten()
			x0_all.append(x0)
			if np.any(np.abs(x0)>xmax):
				continue
			run_single_opt(x0, f_min, df_min, ddf_min,
						   r_t = r_t,
						   xmax = xmax,
						   tol = tol,
						   min_tol = min_tol,
						   min_update = min_update,
						   res_min = res_min,
						   res_nfev = res_nfev,
						   res_succ = res_succ,
						   new_min = new_min,
						   method = method,
						  )
		data_all.append({
			'min': res_min,
			'nfev': res_nfev,
			'success': res_succ,
			'newmin': new_min,
			'x0': x0_all
		})
	return data_all

def model_test_samp_cr5bp(
	model,
	mu_norm, murange,
	r_t,
	use_hess = True,
	xmax = 1.3,
	tol = 1e-8,
	min_tol = 1e-6,
	N_seed = 32
):
	"""Sample from model and run root until all equilibrium points found"""
	mu = murange[0] + 0.5*(mu_norm+1)*(murange[1]-murange[0])
	min_true = get_crit_cr5bp(mu_norm, mu_min=murange[0], mu_max=murange[1])
	
	dist, f_root, df_root = get_root_fns('CR5BP', mu, use_hess=use_hess)
	c = tf.constant([[mu_norm]],dtype=tf.float32)
	data_all = []
	for seed_no in trange(N_seed):
		min_update = []
		res_min = []
		res_nfev = []
		res_succ = []
		new_min = []
		x0_all = []
		while len(min_update)<min_true.shape[0]:
			x0 = model.sample_inv_class(1, c).numpy().flatten()
			x0_all.append(x0)
			if np.any(np.abs(x0)>xmax):
				continue
			run_single_root(x0, f_root, df_root,
					   r_t = r_t,
					   xmax = xmax,
					   tol = tol,
					   min_tol = min_tol,
					   min_update = min_update,
					   res_min = res_min,
					   res_nfev = res_nfev,
					   res_succ = res_succ,
					   new_min = new_min)
		data_all.append({
			'min': res_min,
			'nfev': res_nfev,
			'success': res_succ,
			'new_min': new_min,
			'x0': x0_all
		})
	return data_all

####
# Running sets of parameters

def run_dist_nfev_parallel(
	omega_theta_norm,
	distname='Rastrigin',
	run_model=True,
):
	omega_norm, theta_norm = omega_theta_norm
	dirname_all = 'Parameters/'

	thetarange = [0., 10.]
	if distname=='Rastrigin':
		omrange = [3.5,4.5]
		omega = omrange[0] + 0.5*(omega_norm+1)*(omrange[1]-omrange[0])
		theta = thetarange[0] + 0.5*(theta_norm+1)*(thetarange[1]-thetarange[0])
		min_true = get_min_rot_rast(omega, theta*np.pi/180.)
		r_t = 0.5/omega
		dx = None
		use_hess = True
		param_str = 'mu_3p5_4p5_theta_0_10'
		run_no = 7
	elif distname=='Schwefel':
		omrange = [1100,1800]
		omega = omrange[0] + 0.5*(omega_norm+1)*(omrange[1]-omrange[0])
		theta = thetarange[0] + 0.5*(theta_norm+1)*(thetarange[1]-thetarange[0])
		min_x = 10./omega
		dx = np.array([1.,-1.])
		min_true = get_min_rot_schwefel(omega, theta*np.pi/180., min_x=min_x, dx=dx)
		r_t = 40./omega
		use_hess = True
		param_str = 'mu_1100_1800_theta_0_10'
		run_no = 7
	elif distname=='Whitley':
		omrange = [0.2, 0.3]
		omega = omrange[0] + 0.5*(omega_norm+1)*(omrange[1]-omrange[0])
		theta = thetarange[0] + 0.5*(theta_norm+1)*(thetarange[1]-thetarange[0])
		min_true = get_min_rot_whitley(omega, theta*np.pi/180.)
		r_t = 6e-3
		dx = None
		use_hess = False
		param_str = 'mu_0p2_0p3_theta_0_10'
		run_no = 6

	if run_model:
		fname_load = f'_{distname}_parameters_run{run_no}'
		model, _ = load_model_json(
			os.path.join(PATH_TO_MODELS,dirname_all),
			fname_load,
			seed=12345678)
		data_test = model_test_samp_min(
			model,
			omega_norm, theta_norm, 
			omrange, thetarange,
			r_t, dx,
			min_true,
			distname = distname,
			use_hess = use_hess,
			tol = 1e-8,
			min_tol = 1e-2,
			N_seed = 32)
		nfev_tot = [sum(d['nfev']) for d in data_test]
	else:
		data_rand = random_nfev_rot_data(
			omega, theta*np.pi/180., 
			r_t, dx, 
			min_true,
			distname = distname,
			N_seed = 32,
			tol = 1e-8,
			min_tol = 1e-2,
			seed = 12345678)
		nfev_tot = [sum(d['nfev']) for d in data_rand]
	return nfev_tot

def run_cr5bp_nfev_parallel(mu_norm,run_model=True):
	murange = [0.005,0.995]
	dirname_all = 'Parameters/'
	
	mu = murange[0] + 0.5*(mu_norm+1)*(murange[1]-murange[0])
	r_t = 6e-3
	min_true = get_crit_cr5bp(mu_norm, mu_min=murange[0], mu_max=murange[1])
	if run_model:
		run_no = 6
		fname_load = f'_CR5BP_parameters_run{run_no}'
		model, _ = load_model_json(
			os.path.join(PATH_TO_MODELS,dirname_all),
			fname_load,
			seed=12345678)
		data_test = model_test_samp_cr5bp(
			model, mu_norm, murange, r_t, 
			xmax = 1.3,
			N_seed = 32)
		nfev_tot = [sum(d['nfev']) for d in data_test]
	else:
		data_rand = random_nfev_potential_data(
			mu, r_t, min_true,
			xmax = 1.3, 
			N_seed = 32, 
			seed = 12345678)
		nfev_tot = [sum(d['nfev']) for d in data_rand]
	return nfev_tot


if __name__=='__main__':
	n_process = 8
	
	distname = 'Rastrigin'
	run_model = True
	
	PATH_TO_DATA = os.path.join(parentdir,'DataESGenepy')
	PATH_TO_MODELS = os.path.join(parentdir,'ModelsESGenepy')

	if distname=='CR5BP':
		norm_test_all = [-0.9,-0.7,-0.6,-0.5,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.5,0.6,0.7,0.9,0.99]
		param_str = 'mu_0p005_0p995'
		f_run = partial(run_cr5bp_nfev_parallel,run_model=run_model)
		data_save = {'mu_norm':[float(n) for n in norm_test_all]}
	else:
		norm_test_omega = [-0.6,0.6]
		norm_test_theta = [-0.9,-0.7,-0.6,-0.5,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.5,0.6,0.7,0.9]
		norm_test_all = [(i,j) for i,j in product(norm_test_omega, norm_test_theta)]
		data_save = {
			'omega_norm':[float(n[0]) for n in norm_test_all],
			'theta_norm':[float(n[1]) for n in norm_test_all],
		}
		f_run = partial(run_dist_nfev_parallel,distname=distname,run_model=run_model)
		if distname=='Rastrigin':
			param_str = 'mu_3p5_4p5_theta_0_10'
		elif distname=='Schwefel':
			param_str = 'mu_1100_1800_theta_0_10'
		elif distname=='Whitley':
			param_str = 'mu_0p2_0p3_theta_0_10'
	
	t_now = dt.now()
	dirname_save = os.path.join(PATH_TO_DATA,'Parameters')
	model_str = 'model' if run_model else 'random'
	fname_save = f'nfev_{t_now.month}_{t_now.day}_{t_now.hour}h{t_now.minute}_{model_str}_{distname}_{param_str}.json'

	with Pool(processes=n_process) as p:
		data_nfev = p.map(f_run, norm_test_all)

	data_save['nfev'] = data_nfev
	with open(os.path.join(dirname_save,fname_save),'w') as f:
		json.dump(data_save, f)
