# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
import pickle
import gc
import time
import pdb

from QLearn.q_agents import QAgent, HeuristicAgent
from QLearn.environment import Environment
from QLearn.run_agent import *
from lander_run import heuristic

from multiprocessing import Pool, cpu_count, set_start_method
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


def do_opt_run(run_no, **kwargs):
	env = Environment("gym_MarsLander:MarsLander-v0")
	N_layer = [int(kwargs['N_hid1']), int(kwargs['N_hid2'])]
	agent = HeuristicAgent(env,heuristic,hidden_layers=N_layer,**kwargs)
	R, _, _ = do_run(agent,env,2000)
	print('Run {} mean value {}'.format(run_no, np.mean(R[-100:])))
	R_out = np.array(R)
	return R_out.tolist()
	
def opt_function(hyper_params):
	global param_list
	param_dict={}
	for i, p in enumerate(param_list):
		param_dict[p] = hyper_params[i]

	n_run = 8
	n_process = np.min([cpu_count(),n_run])
	if __name__ == '__main__':
		p = Pool(processes=n_process)
		R_ep_runs = p.map_async(partial(do_opt_run, **param_dict), range(n_run))
	
	time.sleep(2)
	print('Current params:')
	print(param_dict)

	R_runs = R_ep_runs.get()
	p.close()
	p.join()
	
	regret_runs = [-np.trapz(R_run) for R_run in R_runs]
	regret_bs = bs.bootstrap(np.array(regret_runs), stat_func=bs_stats.mean)
	low_regret = regret_bs.upper_bound
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


if __name__ == '__main__':
	gc.enable()
	set_start_method('fork')
	space = [hp.uniform('alpha',0.0001,0.01),
			 hp.quniform('update_steps', 40, 100, 5),
			 hp.quniform('N_hid1', 100, 200, 5),
			 hp.quniform('N_hid2', 30, 80, 2),
			 hp.uniform('gamma', 0.5, 1.0),
			 hp.uniform('eps0', 0.5, 1.0),
			 hp.quniform('n_eps', 200, 1800, 50),
			 hp.quniform('n_xi', 200, 1800, 50),
			 hp.quniform('minibatch_size', 10, 50, 2)]
	global param_list
	param_list = ('alpha', 'update_steps', 'N_hid1', 'N_hid2', 'gamma', 'eps0', 'n_eps', 'n_xi', 'minibatch_size')

	filename = 'opt_lander_2_3.pkl'
	while True:
		run_trials(filename, space)
