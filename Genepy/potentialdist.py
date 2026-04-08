"""Optimisation distribution defined in terms of partial derivatives"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from scipy.optimize import minimize, root

from itertools import product
from tqdm.notebook import tqdm

from multiprocessing import Pool

class PotentialDist(object):
	"""Distribution for optimisation of potential function"""
	def __init__(self, dim, 
				 A=1.):
		self.dim = tf.constant(dim,dtype=tf.int32)
		self.A_t = tf.constant(A,dtype=tf.float64)
		self.min_f = tf.constant(0.,dtype=tf.float64)
		self.max_f = tf.constant(1.,dtype=tf.float64)
	def fx(self, x, mu_norm):
		return x
	def dfdx(self, x, mu_norm):
		return tf.ones(x.shape)
	def d2fdx2(self, x, mu_norm):
		return tf.zeros(x.shape)
	def d2fdxy(self, x, mu_norm):
		return tf.zeros(x.shape[0])
	def f_np(self, x, mu_norm):
		x_tf = tf.constant(x, dtype=tf.float64)
		mu_tf = tf.constant(mu_norm, dtype=tf.float64)
		y_tf = self.fx(x_tf, mu_tf)
		y = y_tf.numpy()
		del x_tf, y_tf
		return y
	def f_2d_np(self, x):
		mu_norm = tf.ones(x.shape[0], dtype=tf.float64)
		return self.f_np(x, mu_norm)
	def df_np(self, x, mu_norm):
		x_tf = tf.constant(x, dtype=tf.float64)
		mu_tf = tf.constant(mu_norm, dtype=tf.float64)
		y_tf = self.dfdx(x_tf, mu_tf)
		y = y_tf.numpy()
		del x_tf, y_tf
		return y
	def df_2d_np(self, x):
		mu_norm = tf.ones(x.shape[0], dtype=tf.float64)
		return self.df_np(x, mu_norm)
	def ddf_np(self, x, mu_norm):
		x_tf = tf.constant(x, dtype=tf.float64)
		mu_tf = tf.constant(mu_norm, dtype=tf.float64)
		y_tf = self.d2fdx2(x_tf, mu_tf)
		y = y_tf.numpy()
		del x_tf, y_tf
		return y
	def ddf_2d_np(self, x):
		mu_norm = tf.ones(x.shape[0], dtype=tf.float64)
		return self.ddf_np(x, mu_norm)
	def f_df_np(self, x, mu_norm):
		return self.f_np(x), self.df_np(x)
	def fmin_norm(self, x):
		"""Function values of minima in range [-1,1] for mu_norm=1"""
		mu_norm = tf.ones(x.shape[0], dtype=tf.float64)
		ftrue = self.fx(x, mu_norm)
		return 2.*(ftrue-self.min_f)/(self.max_f-self.min_f)-1.
	def Hx(self, x, mu_norm):
		h_diag = self.d2fdx2(x,mu_norm)
		h_od = self.d2fdxy(x,mu_norm)
		h_od = tf.repeat(h_od,2*tf.ones(h_od.shape,dtype=tf.int32))
		ind_all = [[int(i/2), i%2, (i+1)%2] for i in range(h_od.shape[0])]
		H = tf.eye(2,batch_shape=[h_diag.shape[0]],dtype=tf.float64)*tf.reshape(h_diag,(-1,2,1))
		H = tf.tensor_scatter_nd_add(H,ind_all,h_od)
		return H
	def H_np(self, x, mu_norm):
		x_tf = tf.constant(x, dtype=tf.float64)
		mu_tf = tf.constant(mu_norm, dtype=tf.float64)
		H_tf = self.Hx(x_tf, mu_tf)
		H = H_tf.numpy()
		del x_tf, H_tf
		return H
	def ux(self, x, mu_norm):
		df = self.dfdx(x, mu_norm)
		return self.A_t * tf.norm(df,axis=-1)
	def ux_2d(self, x):
		mu_norm = tf.ones(x.shape[0], dtype=tf.float64)
		df = self.dfdx(x, mu_norm)
		return self.A_t * tf.norm(df,axis=-1)
	def u_mu_x(self, x_mu):
		x = x_mu[:,:self.dim]
		mu = tf.squeeze(x_mu[:,self.dim:])
		return self.ux(x, mu)
	def u_np(self, x, mu_norm):
		x_tf = tf.constant(x, dtype=tf.float64)
		mu_tf = tf.constant(mu_norm, dtype=tf.float64)
		u_tf = self.ux(x_tf, mu_tf)
		u = u_tf.numpy()
		del x_tf, u_tf
		return u
	def log_likelihood_mu_x(self, x_mu):
		return self.u_mu_x(x_mu) + tf.cast(tf.math.log(0.5),tf.float64)
	def sample(self, N_samp, mu_norm, xmin=-2.5, xmax=2.5, seed=None):
		rng = np.random.default_rng(seed=seed)
		x_samp = rng.uniform([xmin,xmin,0.],[xmax,xmax,1.],size=(N_samp,self.dim+1))
		mu_all = mu_norm*np.ones(N_samp,dtype=np.float64)
		E_samp = np.exp(-self.u_np(x_samp[:,:2],mu_all))
		return x_samp[E_samp>x_samp[:,-1],:-1]
	def sample_mu(self, N_samp, xmin=-1., xmax=1., seed=None):
		rng = np.random.default_rng(seed=seed)
		x_samp = rng.uniform([xmin,xmin,0.],[xmax,xmax,1.],size=(N_samp,self.dim+1))
		mu_samp = rng.uniform(low=-1.,high=1.,size=N_samp)
		E_samp = np.exp(-self.u_np(x_samp[:,:2],mu_samp))
		E_filt = E_samp>x_samp[:,-1]
		return x_samp[E_filt,:-1], mu_samp[E_filt]

class CR5BPDist(PotentialDist):
	def __init__(self, 
				 A=1.,
				 mu_min=1e-3,
				 mu_max=0.99999):
		super().__init__(2, A=A)
		self.mu_min = tf.constant(mu_min,dtype=tf.float64)
		self.mu_max = tf.constant(mu_max,dtype=tf.float64)
		self.X_i = tf.constant([
			[0., 0.],
			[1./np.sqrt(3), 0.],
			[-1./(2.*np.sqrt(3)), 1./2.],
			[-1./(2.*np.sqrt(3)), -1./2.]
		],dtype=tf.float64) #invariant with mu
		
		min_all = get_crit_cr5bp(1., mu_min=mu_min, mu_max=mu_max)
		f_all = self.f_2d_np(min_all)
		self.min_f = tf.constant(np.min(f_all),dtype=tf.float64)
		self.max_f = tf.constant(np.max(f_all),dtype=tf.float64)
	def get_params(self, mu_norm):
		"""Return physical parameters from normalised mu"""
		mu = (self.mu_max-self.mu_min)*(mu_norm+1.)/2.+self.mu_min
		beta = 1./mu-1.
		k = 1./(3.*(1.+beta*np.sqrt(3)))
		M_i = tf.stack(
			[beta] + [tf.ones(beta.shape,dtype=tf.float64)]*3,
			axis=-1,
		)
		return mu, k, M_i
	def get_delta_x(self, x):
		dx = tf.reshape(x,(-1,1,2))-self.X_i
		r_i = tf.norm(dx, axis=-1)
		return dx, r_i
	def normalise_mu(self, mu):
		return 2.*(mu-self.mu_min)/(self.mu_max-self.mu_min)-1.
	def dxdt(self, t, x, mu_norm):
		xdot = np.zeros(4)
		xdot[0] = x[2]
		xdot[1] = x[3]
		dom_dx = self.df_np(x[:2],mu)
		xdot[2] = dom_dx[0,0]+2.*x[3]
		xdot[3] = dom_dx[0,1]-2.*x[2]
		return xdot
	def fx(self, x, mu_norm):
		_, k, M_i = self.get_params(mu_norm)
		dx, r_i = self.get_delta_x(x)
		return k*tf.reduce_sum(M_i/r_i,axis=-1)+0.5*tf.reduce_sum(x**2,axis=-1)
	def dfdx(self, x, mu_norm):
		_, k, M_i = self.get_params(mu_norm)
		dx, r_i = self.get_delta_x(x)
		Mdx_r = tf.reshape(M_i,(-1,4,1))*dx/tf.reshape(r_i**3,(-1,4,1))
		return x - tf.reshape(k,(-1,1))*tf.reduce_sum(Mdx_r,axis=1)
	def d2fdx2(self, x, mu_norm):
		_, k, M_i = self.get_params(mu_norm)
		dx, r_i = self.get_delta_x(x)
		M3dx_r5 = tf.reshape(M_i,(-1,4,1))*(3*dx**2-tf.reshape(r_i**2,(-1,4,1)))/tf.reshape(r_i**5,(-1,4,1))
		return 1.+tf.reshape(k,(-1,1))*tf.reduce_sum(M3dx_r5,axis=1)
	def d2fdxy(self, x, mu_norm):
		_, k, M_i = self.get_params(mu_norm)
		dx, r_i = self.get_delta_x(x)
		Mxy_r5 = M_i*tf.reduce_prod(dx,axis=-1)/r_i**5
		return 3.*k*tf.reduce_sum(Mxy_r5,axis=-1)

def get_crit_cr5bp(mu_norm, mu_min=0.001, mu_max=0.05):
	"""Critical points in CR5BP for a given normalised mu value"""
	mu = (mu_max-mu_min)*(mu_norm+1.)/2.+mu_min
	beta = 1./mu-1.
	k = 1./(3.*(1.+beta*np.sqrt(3)))
	M_i = np.array([beta]+[1.]*3).reshape(-1,1)
	X_i = np.array([
			[0., 0.],
			[1./np.sqrt(3), 0.],
			[-1./(2.*np.sqrt(3)), 1./2.],
			[-1./(2.*np.sqrt(3)), -1./2.]
		],dtype=np.float64)
	def df_cr5bp(x):
		dx = x.reshape(1,2)-X_i
		r_i = np.linalg.norm(dx, axis=-1).reshape(-1,1)
		return x - k*np.sum(M_i*dx/r_i**3,axis=0)
	def H_cr5bp(x):
		dx = x.reshape(1,2)-X_i
		r_i = np.linalg.norm(dx, axis=-1).reshape(-1,1)
		d2fdx2 = 1.+k*np.sum(M_i*(3*dx**2-r_i**2)/(r_i**5))
		d2fdxy =  3.*k*np.sum(M_i*np.prod(dx,axis=-1)/(r_i**5))
		return np.array([
			[d2fdx2, d2fdxy],
			[d2fdxy, d2fdx2]
		])
	x0_all = []
	line_range = np.linspace(0.01,1.5,800,dtype=np.float64)
	for n in range(6):
		theta = n*np.pi/3.
		x_theta = np.stack([line_range*np.cos(theta,dtype=np.float64),
							line_range*np.sin(theta,dtype=np.float64)],
						   axis=-1,
						  )
		df_range = np.array([df_cr5bp(x) for x in x_theta])
		ind_x0 = (np.diff(np.sign(df_range[:,0]))!=0).nonzero()
		if n%2==0:
			x0_all.extend(x_theta[ind_x0][::2,:])
		else:
			x0_all.extend(x_theta[ind_x0])
	min_all = []
	for x0 in x0_all:
		res = root(df_cr5bp, x0, jac=H_cr5bp)
		min_all.append(res.x)
	return np.array(min_all)

def get_minima_classes_cr5bp(mu_norm_all, 
							 mu_min=0.001, mu_max=0.05, 
							 show_params=False):
	"""Critical points and stability class in CR5BP for different normalised mu"""
	dist_min = CR5BPDist(mu_min=mu_min, mu_max=mu_max)
	mu_all, _, _ = dist_min.get_params(mu_norm_all)
	beta_all = (1./mu_all.numpy())-1.
	if show_params:
		print(mu_all.numpy())
		print(beta_all)
	x0_all = []
	mu_x0 = []
	line_range = np.linspace(0.01,1.,500,dtype=np.float64)
	for mu, k in product(mu_norm_all,range(6)):
		theta = k*np.pi/3.
		x = np.stack([line_range*np.cos(theta),line_range*np.sin(theta)],axis=-1)
		df_range = dist_min.df_np(x,mu*np.ones(x.shape[0],dtype=np.float64))
		ind_x0 = (np.diff(np.sign(df_range[:,0]))!=0).nonzero()
		x0_new = x[ind_x0][::2,:]
		x0_all.extend(x[ind_x0][::2,:])
		mu_x0.extend([mu]*x0_new.shape[0])    
	min_all = []
	for x0, mu in zip(x0_all,mu_x0):
		f_root = lambda x: dist_min.df_np(x.reshape(1,-1),mu)[0,:]
		jac_root = lambda x: dist_min.H_np(x.reshape(1,-1),mu)[0,...]
		res = root(f_root, x0, jac=jac_root, tol=1e-10)
		min_all.append(res.x)
	min_all = np.array(min_all)
	min_all[np.abs(min_all[:,1])<1e-6,1] = 0.
	l_stable = [[1,0],[1,0],[0,1]]*3
	l_unstable = [[1,0]]*9
	labels_all = np.array(l_stable*np.sum(beta_all>43.18) + l_unstable*np.sum(beta_all<=43.18))
	
	return np.hstack([min_all, np.array(mu_x0).reshape(-1,1)]), labels_all

def get_root_fns(distname, mu, use_hess=True):
	if distname=='CR5BP':
		dist = CR5BPDist(mu_min=0., mu_max=mu)
	else:
		raise ValueError(f'Invalid distribution name: {distname}')
	f_root = lambda x: dist.df_np(x.reshape(1,-1), np.ones(1))[0,:]
	if use_hess:
		df_root = lambda x: dist.H_np(x.reshape(1,-1), np.ones(1))[0,...]
	else:
		df_root = None
	return dist, f_root, df_root

def interp_min_cr5bp(x_test,mu_test,mindir='mindata/'):
	"""Quicker approximation of equilibrium points via interpolation"""
	fname_load = os.path.join(os.path.dirname(__file__),
							  'mindata',
							  'minmu_CR5BP_mu0p001_0p05.npy')
	min_all = np.load(fname_load)
	min_new = np.zeros((x_test.shape[0],9,3))
	for i in range(9):
		xi = min_all[i::9,0]
		yi = min_all[i::9,1]
		mu = min_all[i::9,2]
	
		x_new = np.interp(mu_test,mu,xi)
		y_new = np.interp(mu_test,mu,yi)
		min_new[:,i,0] = x_new
		min_new[:,i,1] = y_new
		min_new[:,i,2] = mu_test
	c_new = np.zeros((x_test.shape[0],9))
	mu_filt = mu_test<2.*(1./44.18-0.001)/0.049-1.
	c_new[mu_filt,2::3] = 1.
	return min_new, c_new
