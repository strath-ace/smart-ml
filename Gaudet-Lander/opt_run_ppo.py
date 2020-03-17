import sys
import numpy as np
from tqdm import trange
import pickle
import gc

sys.path.append('RL_lib/Agents/PPO/')
sys.path.append('RL_lib/Utils/')

from env_lib.env_mdr import Env
from env_lib.reward_terminal_mdr  import Reward
import env_lib.env_utils as envu
import env_lib.attitude_utils as attu
from env_lib.dynamics_model import Dynamics_model
from env_lib.lander_model import Lander_model
from env_lib.ic_gen import Landing_icgen
from agent_mdr2 import Agent
from policy_ppo import Policy
from value_function import Value_function
from utils import Mapminmax,Logger,Scaler
import utils
from env_lib.flat_constraint import Flat_constraint
from env_lib.glideslope_constraint import Glideslope_constraint
from env_lib.attitude_constraint import Attitude_constraint
from env_lib.thruster_model import Thruster_model

logger = Logger()
dynamics_model = Dynamics_model()
attitude_parameterization = attu.Quaternion_attitude()

thruster_model = Thruster_model()
thruster_model.max_thrust = 5000
thruster_model.min_thrust = 1000

lander_model = Lander_model(thruster_model, attitude_parameterization=attitude_parameterization,
                           apf_v0=70, apf_atarg=15., apf_tau2=100.)
lander_model.get_state_agent = lander_model.get_state_agent_tgo_alt

reward_object = Reward(tracking_bias=0.01,tracking_coeff=-0.01, fuel_coeff=-0.05, debug=False, landing_coeff=10.)

glideslope_constraint = Glideslope_constraint(gs_limit=-1.0)
shape_constraint = Flat_constraint()
attitude_constraint = Attitude_constraint(attitude_parameterization, 
                                          attitude_penalty=-100,attitude_coeff=-10,
                                         attitude_limit=(10*np.pi, np.pi/2-np.pi/16, np.pi/2-np.pi/16))
env = Env(lander_model,dynamics_model,logger,
          reward_object=reward_object,
          glideslope_constraint=glideslope_constraint,
          shape_constraint=shape_constraint,
          attitude_constraint=attitude_constraint,
          tf_limit=120.0,print_every=10)


env.ic_gen = Landing_icgen(mass_uncertainty=0.05, 
                           g_uncertainty=(0.0,0.0), 
                           attitude_parameterization=attitude_parameterization,
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

env.ic_gen.show()


obs_dim = 12
act_dim = 4


input_normalizer = Scaler(obs_dim)


policy = Policy(obs_dim,act_dim,kl_targ=0.001,epochs=20, beta=0.1, shuffle=True, servo_kl=True, discretize=True)
value_function = Value_function(obs_dim,cliprange=0.5)
agent = Agent(policy,value_function,env,input_normalizer,logger,
              policy_episodes=120,policy_steps=12000,gamma1=0.95,gamma2=0.995, lam=0.98, monitor=env.rl_stats)

gc.enable()
agent.train(300000)
fname = "opt_discrete_4km"
utils.save_run(policy,input_normalizer,env.rl_stats.history,fname)
gc.disable()
print('Finished.')