import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import stats
import bootstrapped.bootstrap as bs
import bootstrapped.power as bs_power
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import pdb


def data_smooth(data, n_avg):
	ind_vec = np.arange(n_avg,len(data)+1,n_avg)
	data_avg = []
	for ind in ind_vec:
		data_avg.append(np.mean(data[ind-n_avg:ind]))
	return data_avg


def plot_data(r_data, title_str=None):
	N_avg = 100
	for run_data in r_data:
		R_plot = data_smooth(run_data, N_avg)
		plt.plot(np.arange(len(R_plot)) * N_avg, R_plot)
	plt.xlabel('Episode', fontsize=12)
	plt.ylabel('Average Total Discounted Reward', fontsize=12)
	if title_str:
		plt.title(title_str + ' Learning Curves')
	plt.show()


def plot_times(t_data1, t_data2=None, leg_labels=None):
	plt.plot(t_data1['params'], np.mean(t_data1['times'], 1), color='m')
	if t_data2:
		plt.plot(t_data2['params'], np.mean(t_data2['times'], 1), color='c')
		plt.legend(leg_labels, fontsize=12)

		m1 = np.mean(np.diff(np.mean(t_data1['times'], 1)))/np.mean(np.diff(t_data1['params']))
		c1 = np.mean(np.mean(t_data1['times'], 1)-[m1*k for k in t_data1['params']])
		print('{}: t = {}k + {}'.format(leg_labels[0], m1, c1))
		print('k50hz: ' + repr((2.0 - c1) / m1))
		plt.plot(t_data1['params'], [m1*k+c1 for k in t_data1['params']], 'k.')

		m2 = np.mean(np.diff(np.mean(t_data2['times'], 1))) / np.mean(np.diff(t_data2['params']))
		c2 = np.mean(np.mean(t_data2['times'], 1) - [m2*k for k in t_data2['params']])
		print('{}: t = {}k + {}'.format(leg_labels[1], m2, c2))
		print('k50hz: ' + repr((2.0 - c2) / m2))
		plt.plot(t_data2['params'], [m2 * k + c2 for k in t_data2['params']], 'k.')
	plt.plot([0, 110], [2, 2], 'k--')

	plt.xlabel('Minibatch Size', fontsize=12)
	plt.ylabel('Time Per 100 Steps (s)', fontsize=12)
	plt.xlim([0, 110])
	plt.ylim([0, 8])
	plt.show()


def disp_data(r_data, title_str=None, hist_range=(0, 200), n_hist=10):
	N_avg = 100
	mean_end = []
	for run_data in r_data:
		mean_end.append(np.mean(run_data[-100:]))
		R_plot = data_smooth(run_data, N_avg)
		plt.plot(np.arange(len(R_plot)) * N_avg, R_plot)

	# mean_good = [i for i in mean_end if i > 150]
	# mean_bad = [i for i in mean_end if i < 50]
	# mean_loss = 200-np.mean(mean_end)
	# print('Good: ' + repr(len(mean_good)))
	# print('Bad: ' + repr(len(mean_bad)))
	# print('Mean Loss: ' + repr(mean_loss))

	plt.xlabel('Episode', fontsize=12)
	plt.ylabel('Average Total Discounted Reward', fontsize=12)
	plt.ylim((-550, 0))
	if title_str:
		plt.title(title_str + ' Learning Curves')
	plt.show()

	plt.hist(mean_end, n_hist, hist_range)
	if title_str:
		plt.title(title_str + ' Histogram')
	plt.show()


def disp_runs(r_data, title_str=None):
	if title_str:
		print(title_str)
	mean_end = []
	for i, run_data in enumerate(r_data):
		mean_end.append([i, np.mean(run_data[-100:])])
	print(tabulate(mean_end))


def data_stats(r_data, runs=None, title_str=None):
	N_avg = 100
	R_avg = [data_smooth(R, N_avg) for R in r_data]
	R_mean = np.mean(R_avg, 0)
	R_err = np.std(R_avg, 0) / np.sqrt(np.size(R_avg, 0))

	plt.errorbar(np.arange(len(R_mean)) * N_avg, R_mean, fmt='b', yerr=R_err, ecolor='k', capsize=3, label='Mean')
	plt.xlabel('Episode', fontsize=14)
	plt.ylabel('Average Total Reward', fontsize=14)
	plt.xlim([0, 1050])
	plt.ylim([0, 210])
	if runs:
		colours = [(0, 0.5, 0), (1, 0.6, 0), (1, 0, 0)]
		r_labels = ['Successful', 'Unsuccessful', 'Worst-case']
		# linestyles = ['--', '-.', ':']
		for i, run_no in enumerate(runs):
			R_plot = data_smooth(r_data[run_no], N_avg)
			plt.plot(np.arange(len(R_plot)) * N_avg, R_plot, color=colours[i], label=r_labels[i])
	plt.legend(fontsize=12)
	# if title_str:
	# 	plt.title(title_str + ' Learning Curves')
	plt.show()


def data_ttest(r_data1, r_data2):
	mean_end1 = [np.mean(run_data[-100:]) for run_data in r_data1]
	mean_end2 = [np.mean(run_data[-100:]) for run_data in r_data2]
	(t_stat, p_value) = stats.ttest_ind(mean_end1, mean_end2, equal_var=False)
	print('t: ' + repr(t_stat))
	print('p: ' + repr(p_value))


def data_abtest(r_data1, r_data2=None, alg_str=None):
	mean_end1 = np.array([np.mean(run_data[-100:]) for run_data in r_data1])
	if alg_str:
		print(alg_str[0])
	print('Mean ' + repr(bs.bootstrap(mean_end1, stat_func=bs_stats.mean)))
	print('STD ' + repr(bs.bootstrap(mean_end1, stat_func=bs_stats.std)))
	if r_data2:
		if alg_str:
			print(alg_str[1])
		mean_end2 = np.array([np.mean(run_data[-100:]) for run_data in r_data2])
		print('Mean ' + repr(bs.bootstrap(mean_end2, stat_func=bs_stats.mean)))
		print('STD ' + repr(bs.bootstrap(mean_end2, stat_func=bs_stats.std)))
		print('Percent change')
		print('Mean ' + repr(bs.bootstrap_ab(mean_end1, mean_end2, bs_stats.mean, bs_compare.percent_change)))
		print('STD ' + repr(bs.bootstrap_ab(mean_end1, mean_end2, bs_stats.std, bs_compare.percent_change)))

	n_bs = 3000
	results_list_bs = []
	print('Running power analysis...')
	for i in range(n_bs):
		result_bs = bs.bootstrap_ab(mean_end1, mean_end1*1.25, bs_stats.mean, bs_compare.percent_change)
		results_list_bs.append(result_bs)
	print(bs_power.power_stats(results_list_bs))


# Load Data #
# cartpole
# print('Loading cartpole...')
# fname = 'data/cartpole_qnet_27_3.npy'
# data_all_qnet = np.load(fname).tolist()
# data_qnet = data_all_qnet['R_run']
# fname = 'data/cartpole_eqlm_20_3.npy'
# data_all_eqlm = np.load(fname).tolist()
# data_eqlm = data_all_eqlm['R_run']
# # timings
# print('Loading timings...')
# fname = 'data/timings_qnet_minib_12_3_0.npy'
# tdata_qnet = np.load(fname).tolist()
# fname = 'data/timings_eqlm_minib_12_3_0.npy'
# tdata_eqlm = np.load(fname).tolist()
# # acrobot
# print('Loading acrobot...')
# fname = 'data/acrobot_eqlm_30_5_all.npy'
# data_all_acrobot = np.load(fname).tolist()
# fname = 'data/acrobot_eqlm_1_4.npy'
# data_old_acrobot = np.load(fname).tolist()

# # Process Data #
# # disp_runs(data_eqlm, 'EQLM')
# # disp_data(data_eqlm, 'EQLM')
# # data_stats(data_eqlm, [4, 68, 5], 'EQLM')
# #
# # disp_runs(data_qnet, 'QNet')
# # disp_data(data_qnet, 'QNet')
# # data_stats(data_qnet, [3, 0, 2], 'Q-Network')
# #
# # plot_times(tdata_qnet, tdata_eqlm, ['Q-Network', 'EQLM'])
# #
# # data_abtest(data_eqlm, data_qnet, alg_str=('EQLM', 'QNet'))
# # data_ttest(data_eqlm, data_qnet)

# data_acro = data_all_acrobot['R_run']
# disp_data(data_acro, title_str='Acrobot EQLM', hist_range=(-500, 0), n_hist=10)
# data_abtest(data_acro)
