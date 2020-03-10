import numpy as np
import os,sys
import env_lib


class LanderEnvironment(object):
	"""
	Environment wrapper for the AAS-18-290_6DOF_journal simulator
	"""
	dynamics_model = env_lib.Dynamics_model()
	attitude_parameterization = env_lib.attu.Quaternion_attitude()
	thruster_model = env_lib.Thruster_model()
	glideslope_constraint = env_lib.Glideslope_constraint(gs_limit=-1.0)
	shape_constraint = env_lib.Flat_constraint()
	def __init__(self):
		self.thruster_model.max_thrust = 5000
		self.thruster_model.min_thrust = 1000
		
		self.lander_model = env_lib.Lander_model(self.thruster_model, attitude_parameterization=self.attitude_parameterization,
                           apf_v0=70, apf_atarg=15., apf_tau2=100.)
		self.lander_model.get_state_agent = self.lander_model.get_state_agent_tgo_alt
		
		self.attitude_constraint = env_lib.Attitude_constraint(self.attitude_parameterization, 
                                          attitude_penalty=-100,attitude_coeff=-10,
                                         attitude_limit=(10*np.pi, np.pi/2-np.pi/16, np.pi/2-np.pi/16))
		self.reward_object = env_lib.Reward(tracking_bias=0.01,tracking_coeff=-0.01, 
											fuel_coeff=-0.05, debug=False, landing_coeff=10.)
		
		self.logger=Logger()
		self.main_env = env_lib.Env(self.lander_model,self.dynamics_model,self.logger,
									reward_object=self.reward_object,
									glideslope_constraint=self.glideslope_constraint,
									shape_constraint=self.shape_constraint,
									attitude_constraint=self.attitude_constraint,
									tf_limit=120.0,print_every=10)
		self.main_env.ic_gen = env_lib.Landing_icgen(mass_uncertainty=0.05, 
								   g_uncertainty=(0.0,0.0), 
								   attitude_parameterization=self.attitude_parameterization,
								   l_offset=0.,
								   adapt_apf_v0=True,
								   inertia_uncertainty_diag=100.0, 
								   inertia_uncertainty_offdiag=10.0,
								   downrange = (0,2000 , -70, -10), 
								   crossrange = (-1000,1000 , -30,30),  
								   altitude = (2300,2400,-90,-70),
								   yaw   = (-np.pi/8, np.pi/8, 0.0, 0.0) ,
								   pitch = (np.pi/4-np.pi/8, np.pi/4+np.pi/16, -0.01, 0.01),
								   roll  = (-np.pi/8, np.pi/8, -0.01, 0.01))
		
		self.state_size=12
		self.action_size=4**2
	
	def render(self):
		self.main_env.render()
	
	def reset(self):
		s = self.main_env.reset()
		return s.reshape(1,-1)
	
	def step(self,a,render=False):
		if isinstance(a,int):
			a = int_to_bin(a)
		s,r,d,info = self.main_env.step(a)
		if render:
			self.main_env.render()
		return s.reshape(1,-1),r,d,info
		
		
class Logger(object):
	""" Simple training logger: saves to file and optionally prints to stdout """
	def __init__(self):
		"""
		Args:
			logname: name for log (e.g. 'Hopper-v1')
			now: unique sub-directory name (e.g. date/time string)
		"""

		self.write_header = True
		self.log_entry = {}
		self.writer = None  # DictWriter created with first call to write() method
		self.scores = []

	def write(self, display=True):
		""" Write 1 log entry to file, and optionally to stdout
		Log fields preceded by '_' will not be printed to stdout

		Args:
			display: boolean, print to stdout
		"""
		if display:
			self.disp(self.log_entry)

	@staticmethod
	def disp(log):
		"""Print metrics to stdout"""
		log_keys = [k for k in log.keys()]
		log_keys.sort()
		print('***** Episode {}, Mean R = {:.1f}  Std R = {:.1f}  Min R = {:.1f}'.format(log['_Episode'],
															   log['_MeanReward'], log['_StdReward'], log['_MinReward']))
		for key in log_keys:
			if key[0] != '_':  # don't display log items with leading '_'
				#print(key, log[key])
				print('{:s}: {:.3g}'.format(key, log[key]))
		print('\n')

	def log(self, items):
		""" Update fields in log (does not write to file, used to collect updates.

		Args:
			items: dictionary of items to update
		"""
		self.log_entry.update(items)

	def close(self):
		pass


def bin_to_int(s_bin):
	s = 0
	s_bin.reverse()
	for i, b in enumerate(s_bin):
		s += (2**i) * b
	return int(s)

def int_to_bin(s,s_dim=4):
	s_str = bin(s)[2:]
	while len(s_str)<s_dim:
		s_str = '0'+s_str
	s_bin = np.array([int(s) for s in s_str])
	return s_bin