"""Testing with class labels"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentdir)

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from datetime import datetime as dt
import pdb

import numpy as np
import tensorflow as tf
from scipy.optimize import root

from tfFlow.utils import *

from optdist import get_min_rast, get_min_rot_rast, get_min_schwefel, get_min_rot_schwefel, get_min_whitley, get_min_rot_whitley, f_opt_rot_batch
from potentialdist import CR5BPDist, interp_min_cr5bp
from run_retrain import get_opt_fns

####
# Finding nearest minima/equilibria

def get_omega_theta(x, distname):
	"""Default values of omegarange and thetarange"""
	thetarange = [0., 10.]
	theta = thetarange[0] + 0.5*(x[:,3]+1)*(thetarange[1]-thetarange[0])
	if distname=='Rastrigin':
		omegarange = [3.5, 4.5]
		omega = omegarange[0] + 0.5*(x[:,2]+1)*(omegarange[1]-omegarange[0])
	elif distname=='Schwefel':
		omegarange = [1100., 1800.]
		omega = omegarange[0] + 0.5*(x[:,2]+1)*(omegarange[1]-omegarange[0])
	elif distname=='Whitley':
		omegarange = [0.2, 0.3]
		omega = omegarange[0] + 0.5*(x[:,2]+1)*(omegarange[1]-omegarange[0])
	else:
		raise ValueError('Invalid distname')
	return omega, theta

def calc_min_class_distance(x, omega, theta, distname):
	"""Minimum distance to minima of each class from x"""
	if distname=='Rastrigin':
		bins = [-0.1, 0.45, 0.9, 2.]
		dx = np.zeros(2)
		if np.abs(theta)>0.:
			min_all = get_min_rot_rast(omega, theta*np.pi/180.)
		else:
			min_all = get_min_rast(omega)
	elif distname=='Schwefel':
		bins = [-7e3, -2e3, -1e3, 1e3]
		dx = np.array([1.,-1.])
		min_x = 10./omega
		if np.abs(theta)>0.:
			min_all = get_min_rot_schwefel(omega, theta*np.pi/180., min_x=min_x, dx=dx)
		else:
			min_all = get_min_schwefel(omega, min_x=min_x)
	elif distname=='Whitley':
		bins = [-1, 0.15, 0.5, 5]
		dx = np.zeros(2)
		if np.abs(theta)>0.:
			min_all = get_min_rot_whitley(omega, theta*np.pi/180.)
		else:
			min_all = get_min_whitley(omega)
	else:
		raise ValueError('Invalid distname')

	dist, _, _, _, _ = get_opt_fns(distname, omega, theta*np.pi/180., dx)
	if np.abs(theta)>0:
		fval_all = f_opt_rot_batch(min_all, dist, theta*np.pi/180., dx).flatten()
	else:
		fval_all = dist.f_np(min_all).flatten()
			
	dist_min = []
	for b1, b2 in zip(bins[:-1],bins[1:]):
		min_c = min_all[(fval_all>b1)&(fval_all<=b2),:]
		if min_c.shape[0]>0:
			dist_c = np.linalg.norm(x.reshape(1,-1)-min_c,axis=-1)
			dist_min.append(np.min(dist_c))
		else:
			dist_min.append(-1.)
	return dist_min

def calc_eq_class_distance(x_samp, mindir='../mindata', show_progress=False):
	"""Minimum distance to equilbrium points of each class from x"""
	x_test = x_samp[:,:2]
	mu_test = x_samp[:,2]
	min_new, c_new = interp_min_cr5bp(x_test,mu_test,mindir=mindir)
	
	min_dist_all = []
	dist_min = CR5BPDist(mu_min=0.001, mu_max=0.05)
	if show_progress:
		t = tqdm(zip(x_test, min_new, c_new),total=x_test.shape[0])
	else:
		t = zip(x_test, min_new, c_new)
	for x, min_mu, c_mu in t:
		min_dist_c = []
		for c in range(2):
			min_test = min_mu[c_mu==c,:]
			min_diff = np.linalg.norm(x.reshape(1,-1)-min_test[:,:2],axis=-1)
			if min_diff.size==0:
				min_dist_c.append(-1.)
			else:
				x0 = min_test[np.argmin(min_diff),:]
				f_root = lambda x1: dist_min.df_np(x1.reshape(1,-1),x0[2])[0,:]
				jac_root = lambda x1: dist_min.H_np(x1.reshape(1,-1),x0[2])[0,...]
				res = root(f_root, x0[:2], jac=jac_root, tol=1e-10)
				min_dist_c.append(np.linalg.norm(x-res.x))
		min_dist_all.append(min_dist_c.copy())
	return min_dist_all, mu_test

####
# Model sampling within limits

def sample_model_limits(model, N, c):
	"""Generate N samples of class c from model with limits on samples"""
	x_samp = np.zeros((0,4))
	c_all = np.eye(3,dtype=np.float32)
	filt_c = np.sum(np.abs(c.reshape(1,-1)-c_all),axis=-1)>0
	ns = N
	while x_samp.shape[0]<N:
		cs = tf.constant([c]*ns,dtype=tf.float32)
		x = model.sample_inv_class(ns, cs).numpy()
		filt_lims = np.all(np.abs(x)<=1,axis=-1)
		
		ll_true = model.log_likelihood(x, cs).numpy()
		filt_ll = np.array([True]*ns)
		for cn in c_all[filt_c,:]:
			c_false = tf.constant([cn]*ns,dtype=tf.float32)
			ll_false = model.log_likelihood(x, c_false).numpy()
			filt_ll = filt_ll&(ll_true>ll_false)
		
		x_keep = x[filt_lims&filt_ll,:]
		x_samp = np.concatenate((x_samp,x_keep), axis=0)
		if x_keep.shape[0]==0:
			ns = ns*10
		else:
			ns = int((ns-x_keep.shape[0])*ns/x_keep.shape[0])+1
	return x_samp[:N,:]

def sample_cr5bp_model_limits(model, N, c):
	"""Generate N samples of class c from model with limits on mu"""
	x_samp = np.zeros((0,3))
	ns = N
	while x_samp.shape[0]<N:
		cs = tf.constant([c]*ns,dtype=tf.float32)
		x = model.sample_inv_class(ns, cs).numpy()
		filt_lims = np.abs(x[:,2])<=1
		
		ll_true = model.log_likelihood(x, cs).numpy()
		ll_false = model.log_likelihood(x, 1.-cs).numpy()
		filt_ll = ll_true>ll_false
		
		x_keep = x[filt_lims&filt_ll,:]
		x_samp = np.concatenate((x_samp,x_keep), axis=0)
		if x_keep.shape[0]==0:
			ns = ns*10
		else:
			ns = int((ns-x_keep.shape[0])*ns/x_keep.shape[0])+1
	return x_samp[:N,:]

####
# Running multiple runs in parallel

def run_parallel(fname, distname, run_no):
	dirname = os.path.join(PATH_TO_MODELS,'Labels')
	fname_load = fname + repr(run_no)
	model, _ = load_model_json(dirname,fname_load,seed=12345678)
	dist_all = []
	for c in range(3):
		c_test = np.zeros(3)
		c_test[c] = 1.
		x_samp = sample_model_limits(model, 1000, c_test)
		omega_samp, theta_samp = get_omega_theta(x_samp, distname)
		dist_min_c = []
		for x,omega,theta in zip(x_samp[:,:2],omega_samp,theta_samp):
			dist_min_c.append(calc_min_class_distance(x,omega,theta,distname))
		dist_all.append(np.array(dist_min_c))
	return dist_all

def run_parallel_cr5bp(fname, run_no):
	dirname = os.path.join(PATH_TO_MODELS,'Labels')
	fname_load = fname + repr(run_no)
	model, _ = load_model_json(dirname,fname_load,seed=12345678)
	dist_all = []
	for c in range(2):
		c_test = np.zeros(2)
		c_test[c] = 1.
		x_samp = sample_cr5bp_model_limits(model, 1000, c_test)
		dist_samp, mu_samp = calc_eq_class_distance(x_samp)
		dist_all.append(np.array(dist_samp))
	return dist_all

if __name__=='__main__':
	n_run = 8
	n_process = np.min([8, n_run])

	distname = 'Rastrigin'

	PATH_TO_DATA = os.path.join(parentdir,'DataESGenepy')
	PATH_TO_MODELS = os.path.join(parentdir,'ModelsESGenepy')
	
	scenario_all = ['semisup', 'fullysup', 'partsup']
	t_now = dt.now()
	for scenario in scenario_all:
		print(scenario)
		
		fname = f'_{distname}_{scenario}_run'
		if distname=='CR5BP':
			f_run = partial(run_parallel_cr5bp, fname)
		else:
			f_run = partial(run_parallel, fname, distname)
		
		with Pool(processes=n_process) as p:
			dist_all = p.map(f_run, range(n_run))
		
		fname_save = os.path.join(
			PATH_TO_DATA,'Labels',
			f'sampdist_{t_now.month}_{t_now.day}_{t_now.hour}h{t_now.minute}_{distname}_{scenario}.npy'
		)
		np.save(fname_save, dist_all)
