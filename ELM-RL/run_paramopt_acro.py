# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
from eqlm_agent import EQLMAgent
from qnet_agent import QNetAgent
from environment import Environment
from multiprocessing import Pool, cpu_count
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import pickle
import gc
import time


def network_config():
	netcon = {}
	netcon['alpha'] = 0.01
	netcon['gamma_reg'] = 0.03
	netcon['clip_norm'] = 1.0
	netcon['update_steps'] = 15
	netcon['N_hid'] = 22
	return netcon


def agent_config():
	agentcon = {}
	agentcon['gamma'] = 0.95
	agentcon['eps0'] = 0.88
	agentcon['epsf'] = 0.0
	agentcon['n_eps'] = 1400
	agentcon['minib'] = 20
	agentcon['max_mem'] = 10000
	agentcon['max_exp'] = 500
	return agentcon


def opt_function(hyper_params):
	# [update_steps, N_hid, gamma, eps0]

	n_process = cpu_count()
	n_run = 8
	if __name__ == '__main__':
		p = Pool(processes=n_process, initializer=init_configs, initargs=hyper_params)
		R_ep_runs = p.map_async(do_run_acro, range(n_run))

	# print('Current params:' +
	# 	  '\nupdate_steps ' + repr(hyper_params[0].__trunc__()) +
	# 	  '\nN_hid ' + repr(hyper_params[1].__trunc__()) +
	# 	  '\ngamma ' + repr(round(hyper_params[2], 3)) +
	# 	  '\neps0 ' + repr(round(hyper_params[3], 3)))

	print('Current gamma_reg:' + repr(hyper_params[0]))

	R_runs = R_ep_runs.get()
	regret_runs = [-np.trapz(R_run) for R_run in R_runs]
	regret_bs = bs.bootstrap(np.array(regret_runs), stat_func=bs_stats.mean)
	low_regret = regret_bs.lower_bound
	print('Loss: ' + repr(low_regret))

	return {
		'loss': low_regret,
		'status': STATUS_OK,
		'loss_runs': regret_runs,
		'loss_episodes': R_runs
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


# def init_configs(update_steps, N_hid, gamma, eps0):
# 	# [update_steps, N_hid, gamma, eps0]
# 	global netcon, agentcon
# 	netcon = network_config()
# 	agentcon = agent_config()

# 	netcon['update_steps'] = int(update_steps)
# 	netcon['N_hid'] = int(N_hid)
# 	agentcon['gamma'] = gamma
# 	agentcon['eps0'] = eps0

def init_configs(gamma_reg):
	global netcon, agentcon
	netcon = network_config()
	agentcon = agent_config()

	netcon['gamma_reg'] = gamma_reg


def do_run_acro(run_no):
	global agentcon, netcon
	N_ep = 2000
	env = Environment('Acrobot-v1')
	agent = EQLMAgent(agentcon, netcon, env)

	R_ep = []
	for ep_no in range(N_ep):
		observation = env.reset()
		done = False
		r = 0
		while not done:
			action = agent.action_select(env, observation)
			observation, reward, done, info = env.step(action)
			agent.update_net(observation, reward, done)
			r += reward
		R_ep.append(r)
	agent.sess.close()

	# print('Run {} mean value {}'.format(run_no, np.mean(R_ep[-100:])))
	return R_ep


# # [update_steps, N_hid, gamma, eps0]

gc.enable()
# space = [hp.quniform('update_steps', 10, 40, 1),
# 		 hp.quniform('N_hid', 20, 30, 1),
# 		 hp.uniform('gamma', 0.8, 1.0),
# 		 hp.uniform('eps0', 0.8, 1.0)]
space = [hp.loguniform('gamma_reg', -4.6, -1.6)]

filename = 'opt_acrobot_gammar_12_6.p'
while True:
	run_trials(filename, space)
