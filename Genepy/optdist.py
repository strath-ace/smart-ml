"""Optimisation distribution defined in terms of partial derivatives"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from scipy.optimize import minimize, root

from itertools import product

####
# Dist Classes

class OptDist(object):
	"""Distribution for optimisation"""
	def __init__(self, dim, 
				 A=1.,
				 H_scale=1.):
		self.dim = dim
		self.A_t = tf.constant(A,dtype=tf.float64)
		self.Hs = tf.constant(H_scale,dtype=tf.float64)
		self.min_f = tf.constant(0.,dtype=tf.float64)
		self.max_f = tf.constant(1.,dtype=tf.float64)
	def fx(self, x):
		return x
	def dfdx(self, x):
		return tf.ones(x.shape,dtype=x.dtype)
	def d2fdx2(self, x):
		return tf.zeros(x.shape,dtype=x.dtype)
	def d2fdxy(self, x):
		return tf.zeros(x.shape[0],dtype=x.dtype)
	def f_np(self, x):
		x_tf = tf.constant(x, dtype=tf.float64)
		y_tf = self.fx(x_tf)
		y = y_tf.numpy()
		del x_tf, y_tf
		return y
	def df_np(self, x):
		x_tf = tf.constant(x, dtype=tf.float64)
		y_tf = self.dfdx(x_tf)
		y = y_tf.numpy()
		del x_tf, y_tf
		return y
	def ddf_np(self, x):
		x_tf = tf.constant(x, dtype=tf.float64)
		y_tf = self.d2fdx2(x_tf)
		y = y_tf.numpy()
		del x_tf, y_tf
		return y
	def fmin_norm(self, x):
		"""Function values of minima in range [-1,1]"""
		ftrue = self.fx(x)
		return 2.*(ftrue-self.min_f)/(self.max_f-self.min_f)-1.
	def fmin_norm_np(self, x):
		x_tf = tf.constant(x, dtype=tf.float64)
		fmin_tf = self.fmin_norm(x_tf)
		fmin = fmin_tf.numpy()
		del x_tf, fmin_tf
		return fmin
	def f_df_np(self, x):
		return self.f_np(x), self.df_np(x)
	def Hx(self, x):
		h_diag = self.d2fdx2(x)
		h_od = self.d2fdxy(x)
		Hx_diag = tf.eye(self.dim,batch_shape=[x.shape[0]],dtype=tf.float64)*tf.reshape(h_diag,[-1,self.dim,1])
		Hx_od = (-tf.eye(self.dim,batch_shape=[x.shape[0]],dtype=tf.float64)+1.)*tf.reshape(h_od,[-1,1,1])
		return Hx_diag + Hx_od
	def ux(self, x):
		df = self.dfdx(x)
		H = self.Hx(x)
		e = tf.math.real(tf.linalg.eigvals(H))
		eig_cond = tf.reduce_sum((1.-tf.sign(e))*e**2,axis=-1)
		return self.A_t * (tf.norm(df,axis=-1) + self.Hs*eig_cond)
	def H_np(self, x):
		x = np.expand_dims(x,0)
		x_tf = tf.constant(x, dtype=tf.float64)
		H_tf = self.Hx(x_tf)
		H = H_tf.numpy()
		del x_tf, H_tf
		return H.reshape(self.dim,self.dim)
	def u_np(self, x):
		x_tf = tf.constant(x, dtype=tf.float64)
		u_tf = self.ux(x_tf)
		u = u_tf.numpy()
		del x_tf, u_tf
		return u
	def sample(self, N_samp, xmin=-2.5, xmax=2.5, seed=None):
		rng = np.random.default_rng(seed=seed)
		x_samp = rng.uniform([xmin,xmin,0.],[xmax,xmax,1.],size=(N_samp,self.dim+1))
		u_samp = np.exp(-self.u_np(x_samp[:,:2]))
		return x_samp[u_samp>x_samp[:,-1],:-1]
	def get_u_func(self, N):
		@tf.function(
			input_signature=[tf.TensorSpec(shape=(N,self.dim), dtype=tf.float64)])
		def u_func_shape(x):
			x_clip = tf.clip_by_value(x,
									  tf.constant(-1,dtype=tf.float64),
									  tf.constant(1,dtype=tf.float64),
									  name='x_clip')
			return self.ux(x_clip)
		return u_func_shape

class MinNormalDist(object):
	"""Normal Distribution at Every Minimum"""
	def __init__(self, min_all, sigma=1e-4):
		self.min_all = tf.constant(min_all,dtype=tf.float64)
		self.sigma = tf.constant(sigma,dtype=tf.float64)
		self.dist_tf = tfp.distributions.MultivariateNormalDiag(
			loc = tf.expand_dims(self.min_all,axis=0),
			scale_diag = tf.expand_dims(self.sigma*tf.ones(self.min_all.shape,dtype=tf.float64),axis=0)
		)
	def ux(self, x):
		return -tf.reduce_max(
			self.dist_tf.log_prob(tf.expand_dims(x,axis=1)),
			axis=-1)
	def u_np(self, x):
		x_tf = tf.constant(x, dtype=tf.float64)
		u_tf = self.ux(x_tf)
		u = u_tf.numpy()
		del x_tf, u_tf
		return u

####
# Rotated dist for optimisation

def f_opt_rot(x, dist, theta, dx):
	M_neg = np.array([
		[cos(theta), sin(theta)],
		[-sin(theta), cos(theta)]
	],dtype=np.float64)
	x_rot = np.matmul(M_neg, x-dx)+dx
	return dist.f_np(x_rot)

def f_norm_rot(x, dist, theta, dx):
	M_neg = np.array([
		[cos(theta), sin(theta)],
		[-sin(theta), cos(theta)]
	],dtype=np.float64)
	x_rot = np.matmul(M_neg, x-dx)+dx
	return dist.fmin_norm(tf.constant(x_rot,dtype=tf.float64)).numpy()

def f_opt_rot_batch(x, dist, theta, dx):
	M_neg = np.array([[
		[cos(theta), sin(theta)],
		[-sin(theta), cos(theta)]
	]],dtype=np.float64)
	x_rot = np.array([np.matmul(M_neg, xt-dx)+dx for xt in x])
	return dist.f_np(x_rot)

def df_opt_rot(x, dist, theta, dx):
	M_neg = np.array([
		[cos(theta), sin(theta)],
		[-sin(theta), cos(theta)]
	],dtype=np.float64)
	x_rot = np.matmul(M_neg, x-dx)+dx
	df_x = dist.df_np(x_rot)
	df_rot = np.matmul(M_neg.transpose(),df_x)
	return df_rot

def ddf_opt_rot(x, dist, theta, dx):
	M_neg = np.array([
		[cos(theta), sin(theta)],
		[-sin(theta), cos(theta)]
	],dtype=np.float64)
	x_rot = np.matmul(M_neg, x-dx)+dx
	df_x = dist.df_np(x_rot)
	ddf_x = dist.H_np(x_rot)
	ddf_rot = np.matmul(np.ones((2,2,2)),df_x) + np.matmul(
		np.matmul(
			M_neg.transpose(),
			ddf_x),
		M_neg)
	return ddf_rot

def get_opt_fns(distname, omega, theta, dx, use_hess=True):
	"""Return dist functions for running local optimisations"""
	if distname=='Rastrigin':
		dist = RastriginDist(omega=omega)
		method = 'Newton-CG'
	elif distname=='Schwefel':
		dist = SchwefelDist(omega=omega,shift=1.+10./omega)
		method = 'Newton-CG'
	elif distname=='Whitley':
		dist = WhitleyDist(omega=omega)
		method = 'BFGS'
		use_hess = False
	else:
		raise ValueError(f'Invalid distribution name: {distname}')
		
	if np.abs(theta)>0:
		f_min = lambda x: f_opt_rot(x, dist, theta, dx)
		df_min = lambda x: df_opt_rot(x, dist, theta, dx)
		if use_hess:
			ddf_min = lambda x: ddf_opt_rot(x, dist, theta, dx)
		else:
			ddf_min = None
	else:
		f_min = dist.f_np
		df_min = dist.df_np
		ddf_min = dist.H_np if use_hess else None
		
	return dist, f_min, df_min, ddf_min, method

####
# Dist classes for global opt

class RastriginDist(OptDist):
	def __init__(self, A=0.1, omega=1., dim=2):
		super().__init__(dim, A=A)
		self.omega = tf.constant(omega,dtype=tf.float64)
		self.pi = tf.constant(pi,dtype=tf.float64)

		min_all = get_min_rast(omega, dim=dim)
		f_all = self.f_np(min_all)
		self.min_f = tf.constant(np.min(f_all),dtype=tf.float64)
		self.max_f = tf.constant(np.max(f_all),dtype=tf.float64)
	def fx(self, x):
		return 10.*self.dim + tf.reduce_sum(x**2 - 10.*tf.cos(2.*self.pi*self.omega*x),axis=-1)
	def dfdx(self, x):
		return 2*x + 20.*self.pi*self.omega*tf.sin(2.*self.pi*self.omega*x)
	def d2fdx2(self, x):
		return 2 + 40.*(self.pi**2)*(self.omega**2)*tf.cos(2.*self.pi*self.omega*x)
	def d2fdxy(self, x):
		return tf.zeros(x.shape[0], dtype=x.dtype)

class SchwefelDist(OptDist):
	def __init__(self, A=0.001, omega=1., shift=None, dim=2):
		super().__init__(dim, A=A)
		if shift is None:
			shift = 1+20./omega
		self.omega = tf.constant(omega,dtype=tf.float64)
		self.shift = tf.constant(shift,dtype=tf.float64)
		
		min_all = get_min_schwefel(omega=omega,min_x=shift-1,dim=dim)
		f_all = self.f_np(min_all)
		self.min_f = tf.constant(np.min(f_all),dtype=tf.float64)
		self.max_f = tf.constant(np.max(f_all),dtype=tf.float64)
	def fx(self, x):
		x = x+self.shift
		return 418.9829*self.dim - tf.reduce_sum(
			self.omega*x*tf.sin(tf.sqrt(tf.abs(self.omega* x))),
			axis=-1
		)
	def dfdx(self, x):
		x = x+self.shift
		num_df = -( self.omega**3*x**2*tf.cos(tf.sqrt(tf.abs(self.omega*x))) )
		den_df = 2*tf.abs(self.omega*x)**1.5
		return num_df/den_df - self.omega*tf.sin(tf.sqrt(tf.abs(self.omega*x)))
	def d2fdx2(self, x):
		x = x+self.shift
		root_omx = tf.sqrt(tf.abs(self.omega*x))
		f1 = -( self.omega**4*x**2 * (root_omx*tf.sin(root_omx) + tf.cos(root_omx)) ) / ( 4.*tf.abs(self.omega*x)**3.5 )
		f2 = ( self.omega**2*x*tf.cos(root_omx) ) / ( tf.abs(self.omega*x)**1.5 )
		return -self.omega * (x*f1 + f2)
	def d2fdxy(self, x):
		return tf.zeros(x.shape[0], dtype=x.dtype)

class WhitleyDist(OptDist):
	def __init__(self, A=0.001, omega=0.4):
		dim = 2
		super().__init__(dim, A=A)

		self.shift = tf.ones(2,dtype=tf.float64)
		self.scale = tf.constant(omega,dtype=tf.float64)

		min_all = get_min_whitley(omega)
		f_all = self.f_np(min_all)
		self.min_f = tf.constant(np.min(f_all),dtype=tf.float64)
		self.max_f = tf.constant(np.max(f_all),dtype=tf.float64)
		
	def f_u(self, a, b):
		return 100.*(a**2-b)**2+(1.-a)**2
	def df1_da(self, a, b):
		return 1e-3*self.f_u(a,b)*(200.*a**3-200.*a*b+a-1.)
	def df1_db(self, a, b):
		return -0.1*self.f_u(a,b)*(a**2-b)
	def df2_da(self, a, b):
		return 2.*(200.*a**3-200.*a*b+a-1.)*tf.sin(self.f_u(a,b))
	def df2_db(self, a, b):
		return -200.*(a**2-b)*tf.sin(self.f_u(a,b))
	def fx(self, x):
		x = x*self.scale+self.shift
		y = 0.
		for i,j in zip([0,1],[1,0]):
			u = self.f_u(x[...,i],x[...,j])
			y += (u**2)/4e3-tf.cos(u)+1.
		return y
	def dfdx(self, x):
		x = x*self.scale+self.shift
		df = []
		for i,j in zip([0,1],[1,0]):
			df1_dxi = self.df1_da(x[...,i],x[...,j]) + self.df1_db(x[...,j],x[...,i])
			df2_dxi = self.df2_da(x[...,i],x[...,j]) + self.df2_db(x[...,j],x[...,i])
			df.append(df1_dxi + df2_dxi)
		return self.scale*tf.stack(df,axis=-1)
	def d2fdx2(self, xs):
		return None
	def d2fdxy(self, xs):
		return None

####
# Functions for returning minima

def get_min_rast(omega, dim=2):
	"""Return minima in range (-1,1) for Rastrigin function"""
	def df_rast_1d(x, omega=omega):
		return 2*x + 20*pi*omega*sin(2*pi*omega*x)
	def ddf_rast_1d(x, omega=omega):
		return 2. + 40.*(pi**2)*(omega**2)*cos(2*pi*omega*x)
	x0_all = np.arange(-np.floor(omega),omega)/omega
	min_all = []
	for x0 in x0_all:
		res = root(df_rast_1d, x0, jac=ddf_rast_1d)
		min_all.append(res.x[0])
	X_min_all = np.array([list(m) for m in product(*[min_all]*dim)])
	return X_min_all

def get_min_rot_rast(omega, theta,
					 fvals=False):
	"""Return minima in range (-1,1) for 2D rotated Rastrigin function"""
	def df_rast_1d(x, omega=omega):
		return 2*x + 20*pi*omega*sin(2*pi*omega*x)
	def ddf_rast_1d(x, omega=omega):
		return 2. + 40.*(pi**2)*(omega**2)*cos(2*pi*omega*x)
	x0_all = np.arange(-np.floor(1.5*omega),1.5*omega)/omega
	min_all = []
	for x0 in x0_all:
		res = root(df_rast_1d, x0, jac=ddf_rast_1d)
		min_all.append(res.x[0])
	X_min_all = np.array([[i,j] for i,j in product(min_all,min_all)])
	M = np.array([
		[cos(theta), -sin(theta)],
		[sin(theta), cos(theta)]
	])
	X_min_rot = np.matmul(
		M,
		np.expand_dims(X_min_all,axis=-1), 
		axes=[(-2,-1),(-2,-1),(-2,-1)]
	).reshape(X_min_all.shape)
	xmin_filt = np.all(X_min_rot>=-1,axis=-1)&np.all(X_min_rot<=1,axis=-1)
	X_rot_filt = X_min_rot[xmin_filt,:]
	if fvals:
		dist = RastriginDist(omega=omega)
		dx = np.zeros(2)
		fvals_min = f_opt_rot_batch(X_rot_filt, dist, theta, dx)
		return X_rot_filt, fvals_min
	return X_rot_filt

def get_min_schwefel(omega, min_x=1e-3, dim=2):
	"""Return shifted minima in range (min_x,2+min_x) for Schwefel function"""
	def df_schwefel_1d(x):
		num_df = -( omega**3*x**2*cos(np.sqrt(np.abs(omega*x))) )
		den_df = 2*np.abs(omega*x)**1.5
		return num_df/den_df - omega*sin(np.sqrt(np.abs(omega*x)))
	def ddf_schwefel_1d(x):
		root_omx = np.sqrt(np.abs(omega*x))
		f1 = -( omega**4*x**2 * (root_omx*sin(root_omx) + cos(root_omx)) ) / ( 4.*np.abs(omega*x)**3.5 )
		f2 = ( omega**2*x*cos(root_omx) ) / ( np.abs(omega*x)**1.5 )
		return -omega * (x*f1 + f2)
	kmax = 0.5+np.sqrt((2+min_x)*omega)/pi
	x0_all = (2*np.arange(1,kmax)-1)**2*pi**2/(4*omega)
	min_all = []
	for x0 in x0_all:
		res = root(df_schwefel_1d, x0, jac=ddf_schwefel_1d)
		ddf_x = ddf_schwefel_1d(res.x[0])
		if res.x[0]<min_x or ddf_x<0:
			continue
		min_all.append(res.x[0]-1-min_x)
	X_min_all = np.array([list(m) for m in product(*[min_all]*dim)])
	return X_min_all

def get_min_rot_schwefel(omega, theta, 
						 dx=np.array([1.,-1.]), 
						 min_x=1e-3, 
						 fvals=False):
	"""Return shifted minima in range (min_x,2+min_x) for Schwefel function rotated about dx"""
	def df_schwefel_1d(x):
		num_df = -( omega**3*x**2*cos(np.sqrt(np.abs(omega*x))) )
		den_df = 2*np.abs(omega*x)**1.5
		return num_df/den_df - omega*sin(np.sqrt(np.abs(omega*x)))
	def ddf_schwefel_1d(x):
		root_omx = np.sqrt(np.abs(omega*x))
		f1 = -( omega**4*x**2 * (root_omx*sin(root_omx) + cos(root_omx)) ) / ( 4.*np.abs(omega*x)**3.5 )
		f2 = ( omega**2*x*cos(root_omx) ) / ( np.abs(omega*x)**1.5 )
		return -omega * (x*f1 + f2)
	kmax = 0.5+1.5*np.sqrt((2+min_x)*omega)/pi
	x0_all = (2*np.arange(1,kmax)-1)**2*pi**2/(4*omega)
	min_all = []
	for x0 in x0_all:
		res = root(df_schwefel_1d, x0, jac=ddf_schwefel_1d)
		ddf_x = ddf_schwefel_1d(res.x[0])
		if res.x[0]<min_x or ddf_x<0:
			continue
		min_all.append(res.x[0]-1-min_x)
	X_min_all = np.array([[i,j] for i,j in product(min_all,min_all)])-dx.reshape(1,-1)
	M = np.array([
		[cos(theta), -sin(theta)],
		[sin(theta), cos(theta)]
	])
	X_min_rot = np.matmul(
		M,
		np.expand_dims(X_min_all,axis=-1), 
		axes=[(-2,-1),(-2,-1),(-2,-1)]
	).reshape(X_min_all.shape)+dx.reshape(1,-1)
	xmin_filt = np.all(np.abs(X_min_rot)<=1,axis=-1)
	X_rot_filt = X_min_rot[xmin_filt,:]
	if fvals:
		dist = SchwefelDist(omega=omega, shift=1.+min_x)
		fvals_min = f_opt_rot_batch(X_rot_filt, dist, theta, dx)
		return X_rot_filt, fvals_min
	return X_rot_filt

def get_min_whitley(omega):
	fname_load = os.path.join(os.path.dirname(__file__),
							  'mindata',
							  'whitley_minima_0p6_1p4.npy')
	min_all = np.load(fname_load)
	min_scale = (min_all-1.)/omega
	return min_scale[np.all(np.abs(min_scale)<1.,axis=-1),:]

def get_min_rot_whitley(omega, theta,
						fvals=False):
	fname_load = os.path.join(os.path.dirname(__file__),
							  'mindata',
							  'whitley_minima_0p6_1p4.npy')
	min_all = np.load(fname_load)
	min_scale = (min_all-1.)/omega
	M = np.array([
		[cos(theta), -sin(theta)],
		[sin(theta), cos(theta)]
	])
	X_min_rot = np.matmul(
		M,
		np.expand_dims(min_scale,axis=-1), 
		axes=[(-2,-1),(-2,-1),(-2,-1)]
	).reshape(min_scale.shape)
	xmin_filt = np.all(np.abs(X_min_rot)<=1,axis=-1)
	X_rot_filt = X_min_rot[xmin_filt,:]
	if fvals:
		dist = WhitleyDist(omega=omega)
		dx = np.zeros(2)
		fvals_min = f_opt_rot_batch(X_rot_filt, dist, theta, dx)
		return X_rot_filt, fvals_min
	return X_rot_filt
