# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
from QLearn.q_agents import QAgent
from QLearn.environment import Environment
from QLearn.run_agent import *
import gc

if __name__ == '__main__':
	gc.enable()
	for run_no in range(10):
		env = Environment("gym_MarsLander:MarsLander-v0")
		agent = QAgent(env,net_type='MLPQNet',hidden_layers=[80, 40],memory_size=50000,minibatch_size=20,alpha=0.005,clip_norm=1.0,eps0=0.9, n_eps=2000)
		R, _, _ = do_run(agent, env, 2500, save_name = 'data{}_18_2.pkl'.format(run_no))
		print('Mean reward: ' + repr(np.mean(R)))
		del agent, env, R
	gc.disable()