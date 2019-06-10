import numpy as np
from eqlm_agent import EQLMAgent
from qnet_agent import DoubleQNet
from environment import Environment
from multiprocessing import Pool, cpu_count
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle
import gc
import time


class NetworkConfig():
	def __init__(self):
		self.alpha = 0.01
		self.clip_norm = []
		self.update_steps = 50
		self.gamma_reg = 0.03
		self.N_hid = 20
		self.timing = False


class AgentConfig():
	def __init__(self):
		self.gamma = 1.0
		self.eps0 = 0.9
		self.epsf = 0.0
		self.n_eps = 1400
		self.minib = 20
		self.max_mem = 10000
		self.prioritized = False
		self.printQ = False


def opt_function(hyper_params):
	# [gamma_r/alpha, update_steps, N_hid, gamma, eps0, n_eps, minib]

	n_process = cpu_count()
	n_run = 10
	if __name__ == '__main__':
		p = Pool(processes=n_process, initializer=init_configs, initargs=hyper_params)
		regret_vec = p.map_async(do_run_acro, range(n_run))
	print('Current params:' +
		  '\ngamma_r ' + repr(round(hyper_params[0], 5)) +
		  '\nupdate_steps ' + repr(hyper_params[1].__trunc__()) +
		  '\nN_hid ' + repr(hyper_params[2].__trunc__()) +
		  '\ngamma ' + repr(round(hyper_params[3], 3)) +
		  '\neps0 ' + repr(round(hyper_params[4], 3)) +
		  '\nn_eps ' + repr(hyper_params[5].__trunc__()) +
		  '\nminib ' + repr(hyper_params[6].__trunc__()))
	regret_runs = regret_vec.get()
	mean_regret = np.mean(regret_runs)
	print('Loss: ' + repr(mean_regret))

	return {
		'loss': mean_regret,
		'status': STATUS_OK,
		'loss_runs': regret_runs
	}


def run_trials(filename, space):
	trials_step = 1
	max_trials = 2

	try:
		trials = pickle.load(open(filename, "rb"))
		print('Loading trials...')
		max_trials = len(trials.trials) + trials_step
	except:
		trials = Trials()

	best = fmin(fn=opt_function, space=space, algo=tpe.suggest, max_evals=max_trials, trials=trials)
	print('Best: ', best)

	pickle.dump(trials, open(filename, "wb"))


def init_configs(gamma_r, update_steps, N_hid, gamma, eps0, n_eps, minib):
	# [gamma_r/alpha, update_steps, N_hid, gamma, eps0, n_eps, minib]
	global netcon, agentcon
	netcon = NetworkConfig()
	agentcon = AgentConfig()

	netcon.gamma_reg = gamma_r
	netcon.update_steps = int(update_steps)
	netcon.N_hid = int(N_hid)
	agentcon.gamma = gamma
	agentcon.eps0 = eps0
	agentcon.n_eps = int(n_eps)
	agentcon.minib = int(minib)


def do_run(run_no):
	global agentcon, netcon
	N_ep = 1000
	env = Environment('CartPole-v0')
	agent = EQLMAgent(agentcon, netcon, env)

	regret_ep = []
	for ep_no in range(N_ep):
		observation = env.reset()
		done = False
		r = 0
		while not done:
			action = agent.action_select(env, observation)
			observation, reward, done, info = env.step(action)
			agent.update_net(observation, reward, done)
			r += reward
		regret_ep.append(200-r)
	agent.sess.close()

	print('Run {} value {}'.format(run_no, np.mean(regret_ep[-100:])))
	return np.mean(regret_ep[-100:])


def do_run_acro(run_no):
	global agentcon, netcon
	N_ep = 2000
	env = Environment('Acrobot-v1')
	agent = EQLMAgent(agentcon, netcon, env)

	regret_ep = []
	for ep_no in range(N_ep):
		observation = env.reset()
		done = False
		r = 0
		while not done:
			action = agent.action_select(env, observation)
			observation, reward, done, info = env.step(action)
			agent.update_net(observation, reward, done)
			r += reward
		regret_ep.append(-50-r)
	agent.sess.close()

	print('Run {} value {}'.format(run_no, np.mean(regret_ep[-100:])))
	return np.mean(regret_ep[-100:])


# # [gamma_r/alpha, update_steps, N_hid, gamma, eps0, n_eps, minib]

gc.enable()
space = [hp.loguniform('gamma_reg', -4.6, -1.6),
		 hp.quniform('update_steps', 10, 50, 1),
		 hp.quniform('N_hid', 20, 30, 1),
		 hp.uniform('gamma', 0.5, 1.0),
		 hp.uniform('eps0', 0.6, 1.0),
		 hp.quniform('n_eps', 600, 1500, 20),
		 hp.quniform('minib', 5, 30, 1)]

# best = fmin(opt_function, space, tpe.suggest, 200)

filename = 'opt_acrobot_eqlm_1.p'
while True:
	run_trials(filename, space)
