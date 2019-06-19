# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
import pickle
from tabulate import tabulate
import matplotlib.pyplot as plt
import pdb


def data_smooth(data,n_avg):
	ind_vec = np.arange(n_avg,len(data)+1,n_avg)
	data_avg = []
	for ind in ind_vec:
		data_avg.append(np.mean(data[ind-n_avg:ind]))
	return data_avg


def sort_loss(val):
	return val[1]


# fname = 'opt_acrobot_gammar_12_6.p'
fname = 'paramopt_data/opt_acrobot_qnet_area.p'
print('Loading trials...')
trials = pickle.load(open(fname,'rb'))

print('N trials: ' + repr(len(trials.trials)))

tab_data = [[i, trials.losses()[i], trials.trials[i]['result']['loss_runs']] for i in range(len(trials.trials))]
tab_data.sort(key = sort_loss, reverse = True)
print(tabulate(tab_data, headers = ['Run No', 'Loss', 'Loss Runs']))

tab_data = [[i, trials.losses()[i], trials.trials[i]['misc']['vals']] for i in range(len(trials.trials))]
tab_data.sort(key = sort_loss, reverse = True)
print(tabulate(tab_data))

p_vec = [trials.trials[i]['misc']['vals']['update_steps'] for i in range(len(trials.trials))]
loss_vec = [trials.losses()[i] for i in range(len(trials.trials))]

plt.plot(p_vec, loss_vec, 'k.')
plt.show()

N_avg = 50
while True:
	try:
		run_no = input('Enter run number: ')
		r_data = trials.trials[run_no]['result']['loss_episodes']
	except KeyboardInterrupt:
		print('\n')
		break
	except:
		continue
	R_mat = []
	for run_data in r_data:
		R_plot = data_smooth(run_data, N_avg)
		plt.plot((np.arange(len(R_plot)) + 1) * N_avg, R_plot)
		R_mat.append(R_plot)
	plt.xlim((0, 2100))
	plt.ylim((-550, 0))
	plt.show()

	R_mean = np.mean(R_mat, 0)
	plt.plot((np.arange(len(R_mean)) + 1) * N_avg, R_mean)
	plt.xlim((0, 2100))
	plt.ylim((-550, 0))
	plt.show()
