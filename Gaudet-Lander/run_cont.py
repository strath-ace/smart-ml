import sys
import numpy as np
from tqdm import trange
import pickle

sys.path.append('RL_lib/Agents/PPO')
sys.path.append('RL_lib/Utils')
sys.path.append('../')

import QLearn
import env_lib
from policy_ppo import Policy
from value_function import Value_function
from utils import Mapminmax,Logger,Scaler
import utils
from env_gaudet_lander import LanderEnvironment, int_to_bin, bin_to_int

def ppo_policy(s):
    s_norm = input_normalizer.apply(s)
    a = policy.sample(s_norm.reshape(1,-1))
    return bin_to_int(list(a[1][0]))

def random_policy(s):
    return np.random.randint(16)

env = LanderEnvironment()
obs_dim = 12
act_dim = 4

policy = Policy(obs_dim,act_dim,kl_targ=0.001,epochs=20, beta=0.1, shuffle=True, servo_kl=True, discretize=True)

fname = "optimize_4km"
input_normalizer = utils.load_run(policy,fname)


mean_r = -100
N_ep_train = 50
N_ep_test = 10

policy.test_mode=True
agent = QLearn.QAgent(env, net_type='MLPQNet', f_heur=ppo_policy, n_heur=N_ep_train, 
					  alpha=1e-4, update_steps=50, hidden_layers=[150, 80], eps0=0.0, gamma=0.99, memory_size=100000, minibatch_size=25)

R_all = []
steps_all = []
save_name = 'test_qlearn_11_3.pkl'

while mean_r<-20:
	agent.ep_no=0
	R_train, steps, agent, env = QLearn.do_run(agent, env, N_ep_train, show_progress=True)
	agent.ep_no = N_ep_train+10
	R_test, step_test = QLearn.agent_demo(agent, env, N_ep_test)
	R_all.append(R_test)
	steps_all.append(step_test)
	mean_r = np.mean(R_test)
	
	print('Performance: ' + repr(mean_r))
	data = {'params':agent.nn.get_params(),'R':R_all,'step':steps_all}
	pickle.dump(data, open(save_name,'wb'))