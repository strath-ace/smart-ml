# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2021 University of Strathclyde and Author ------
# ---------------- Author: Francesco Marchetti ------------------------
# ----------- e-mail: francesco.marchetti@strath.ac.uk ----------------

# Alternatively, the contents of this file may be used under the terms
# of the GNU General Public License Version 3.0, as described below:

# This file is free software: you may copy, redistribute and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3.0 of the License, or (at your
# option) any later version.

# This file is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

'''

Script to visualize obtained results in a fashion similar to the paper

'''


import matplotlib.pyplot as plt
import numpy as np
from deap import creator, gp, base
from copy import copy
import matplotlib
import os


matplotlib.rcParams.update({'font.size': 20})

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
creator.create("SubIndividual", gp.PrimitiveTree)

normalgp_on = True
save = True
ntot = 100  # number of different iterations
Ngen = 300  #  number of generations in gp algorithms


benches = ['koza1']

fig, axs = plt.subplots(3, 3, sharex=False, sharey=False)
fig.set_size_inches(16, 23)
fig1, axs1 = plt.subplots(3, 3, sharex=False, sharey=False)
fig1.set_size_inches(16, 23)

fig2, axs2 = plt.subplots(3, 3, sharex=False, sharey=False)
fig2.set_size_inches(16, 23)

for k in range(len(benches)):
    bench = benches[k]

    gen_gp = np.linspace(0, Ngen, Ngen+1)

    res_tot_igp = np.zeros((ntot, len(gen_gp)))
    res_tot_sgp = np.zeros((ntot, len(gen_gp)))
    mean_len_tot_igp = np.zeros((ntot, len(gen_gp)))
    mean_len_tot_sgp = np.zeros((ntot, len(gen_gp)))
    ent_tot_igp = np.zeros((ntot, len(gen_gp)))
    ent_tot_sgp = np.zeros((ntot, len(gen_gp)))

    for i in range(ntot):

        res_igp = np.load("Results/IGP_{}/{}_IGP_{}_FIT.npy".format(bench, i, bench))
        res_tot_igp[i, :] = res_igp  # fitness of ith run from generation 0 to 300
        res_sgp = np.load("Results/SGP_{}/{}_SGP_{}_FIT.npy".format(bench, i, bench))
        res_tot_sgp[i, :] = res_sgp
        len_igp = np.load("Results/IGP_{}/{}_IGP_{}_IND_LENGTHS.npy".format(bench, i, bench))
        mean_len_tot_igp[i, :] = np.mean(len_igp, axis=1)
        len_sgp = np.load("Results/SGP_{}/{}_SGP_{}_IND_LENGTHS.npy".format(bench, i, bench))
        mean_len_tot_sgp[i, :] = np.mean(len_sgp, axis=1)
        ent_igp = np.load("Results/IGP_{}/{}_IGP_{}_POP_STATS.npy".format(bench, i, bench), allow_pickle=True)[1:, 2]
        ent_tot_igp[i, :] = ent_igp
        ent_sgp = np.load("Results/SGP_{}/{}_SGP_{}_POP_STATS.npy".format(bench, i, bench), allow_pickle=True)[1:, 2]
        ent_tot_sgp[i, :] = ent_sgp


    ################# RMSE ########################
    iters = np.linspace(0, ntot-1, ntot)

    data_igp = np.load("Results/IGP_{}/{}_IGP_{}_T_RMSE.npy".format(bench, ntot, bench))
    rmse_train_igp = np.asarray(data_igp[1:, 1], dtype=float)
    rmse_test_igp = np.asarray(data_igp[1:, 2], dtype=float)
    mean_rmse_train_igp = np.median(rmse_train_igp)
    std_rmse_train_igp= np.std(rmse_train_igp)
    mean_rmse_test_igp = np.median(rmse_test_igp)
    std_rmse_test_igp= np.std(rmse_test_igp)
    teval_igp = np.asarray(data_igp[1:, 0], dtype=float)
    mean_teval_igp = np.median(teval_igp)
    std_teval_igp = np.std(teval_igp)
    data_sgp = np.load("Results/SGP_{}/{}_SGP_{}_T_RMSE.npy".format(bench, ntot, bench))
    rmse_train_sgp = np.asarray(data_sgp[1:, 1], dtype=float)
    rmse_test_sgp = np.asarray(data_sgp[1:, 2], dtype=float)
    mean_rmse_train_sgp = np.median(rmse_train_sgp)
    std_rmse_train_sgp= np.std(rmse_train_sgp)
    mean_rmse_test_sgp = np.median(rmse_test_sgp)
    std_rmse_test_sgp= np.std(rmse_test_sgp)
    teval_sgp = np.asarray(data_sgp[1:, 0], dtype=float)
    mean_teval_sgp = np.median(teval_sgp)
    std_teval_sgp = np.std(teval_sgp)
    print("{} - Median RMSE Train IGP {} +- {}".format(bench, mean_rmse_train_igp, std_rmse_train_igp))
    print("{} - Median RMSE Test IGP {} +- {}".format(bench, mean_rmse_test_igp, std_rmse_test_igp))
    print("{} - Median RMSE Train SGP {} +- {}".format(bench, mean_rmse_train_sgp, std_rmse_train_sgp))
    print("{} - Median RMSE Test SGP {} +- {}".format(bench, mean_rmse_test_sgp, std_rmse_test_sgp))
    mean_igp = np.median(res_tot_igp, axis=0)
    std_igp = np.std(res_tot_igp, axis=0)
    mean_sgp = np.median(res_tot_sgp, axis=0)
    std_sgp = np.std(res_tot_sgp, axis=0)
    mean_ent_igp = np.median(ent_tot_igp, axis=0)
    std_ent_igp = np.std(ent_tot_igp, axis=0)
    mean_ent_sgp = np.median(ent_tot_sgp, axis=0)
    std_ent_sgp = np.std(ent_tot_sgp, axis=0)
    mean_len_igp = np.median(mean_len_tot_igp, axis=0)
    std_len_igp = np.std(mean_len_tot_igp, axis=0)
    mean_len_sgp = np.median(mean_len_tot_sgp, axis=0)
    std_len_sgp = np.std(mean_len_tot_sgp, axis=0)

    if k < 3:
        r = 0
        c = copy(k)
    elif 3 <= k < 6:
        r = 1
        c = k-3
    else:
        r = 2
        c = k-6

    axs[r,c].plot(gen_gp, mean_igp, 'b-', label='IGP')
    axs[r,c].fill_between(gen_gp, mean_igp - std_igp, mean_igp + std_igp, color='b', alpha=0.2)
    axs[r,c].plot(gen_gp, mean_sgp, 'r--', label='SGP')
    axs[r,c].fill_between(gen_gp, mean_sgp - std_sgp, mean_sgp + std_sgp, color='r', alpha=0.2)
    axs[r,c].set_title(bench)
    axs[r,c].legend()

    axs1[r,c].plot(gen_gp, mean_len_igp, 'b-', label='IGP')
    axs1[r,c].fill_between(gen_gp, mean_len_igp - std_len_igp, mean_len_igp + std_len_igp, color='b', alpha=0.2)
    axs1[r,c].plot(gen_gp, mean_len_sgp, 'r--', label='SGP')
    axs1[r,c].fill_between(gen_gp, mean_len_sgp - std_len_sgp, mean_len_sgp + std_len_sgp, color='r', alpha=0.2)
    axs1[r, c].set_title(bench)
    axs1[r, c].legend()

    axs2[r,c].plot(gen_gp, mean_ent_igp, 'b-', label='IGP')
    axs2[r,c].fill_between(gen_gp, mean_ent_igp - std_ent_igp, mean_ent_igp + std_ent_igp, color='b', alpha=0.2)
    axs2[r,c].plot(gen_gp, mean_ent_sgp, 'r--', label='SGP')
    axs2[r,c].fill_between(gen_gp, mean_ent_sgp - std_ent_sgp, mean_ent_sgp + std_ent_sgp, color='r', alpha=0.2)
    axs2[r, c].set_title(bench)
    axs2[r, c].legend()


    matplotlib.rcParams.update({'font.size': 22})
    figg = plt.figure(k+10)
    figg.set_size_inches(9, 9)
    plt.boxplot([rmse_train_igp, rmse_train_sgp, rmse_test_igp, rmse_test_sgp], notch=True, sym='')
    plt.grid()
    plt.xticks([1, 2, 3, 4], ['Train IGP', 'Train SGP', 'Test IGP', 'Test SGP'], rotation=15)
    plt.ylabel("RMSE")
    #plt.title("Train and Test RMSE on 100 iterations, Benchmark {}".format(bench))
    if save:
        try:
            os.mkdir('Results/Plots')
        except FileExistsError:
            pass
        plt.savefig('Results/Plots/rmse_{}.pdf'.format(bench), format='pdf')
    matplotlib.rcParams.update({'font.size': 20})


if save:
    fig.savefig('Results/Plots/Fitnesses.pdf', format='pdf')
    fig1.savefig('Results/Plots/Lengths.pdf', format='pdf')
    fig2.savefig('Results/Plots/Entropy.pdf', format='pdf')
