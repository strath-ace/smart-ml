# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2023 University of Strathclyde and Author ------
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

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""This script is used to plot the results produced for the paper [1]

[1] [SUBMITTED TO] F. Marchetti, G. Pietropolli, F.J. Camerota Verdù, M. Castelli, E. Minisci, Control Law Automatic 
Design Through Parametrized Genetic Programming with Adjoint State Method Gradient Evaluation. Applied Soft Computing. 2023
"""

if __name__ == '__main__':

    x_ref = np.load('x_ref.npy')
    u_ref = np.load('u_ref.npy')
    time = np.load('time_points.npy')

    n_sims = 30
    n_states = 2
    n_gens = 301
    fits = np.zeros(n_sims)
    times = np.zeros(n_sims)
    best_fit = 1000
    best_sim = 0
    fit_evols = np.zeros((n_sims, n_gens))
    all_states = np.zeros((n_sims, len(time), n_states))
    all_controls = np.zeros((n_sims, len(time)))
    fits_adj = np.zeros(n_sims)
    times_adj = np.zeros(n_sims)
    best_fit_adj = 1000
    best_sim_adj = 0
    fit_evols_adj = np.zeros((n_sims, n_gens))
    all_states_adj = np.zeros((n_sims, len(time), n_states))
    all_controls_adj = np.zeros((n_sims, len(time)))
    fits_opt = np.zeros(n_sims)
    for sim in range(n_sims):
        with open("../../Results/Oscillator/IGP/Sim{}/best_ind_structure_IGP.txt".format(sim)) as f:
            print('IGP Sim {}: '.format(sim), f.read())
        fit = np.load('../../Results/Oscillator/IGP/Sim{}/best_fitness_IGP.npy'.format(sim))

        fits[sim] = fit
        if fit < best_fit:
            best_sim = sim
            best_fit = fit
        comp_time = np.load('../../Results/Oscillator/IGP/Sim{}/computational_time_IGP.npy'.format(sim))
        times[sim] = comp_time
        fit_evol = np.load('../../Results/Oscillator/IGP/Sim{}/fitness_evol_IGP.npy'.format(sim))[:,0]
        fit_evols[sim,:] = fit_evol
        x = np.load('../../Results/Oscillator/IGP/Sim{}/x_IGP.npy'.format(sim))
        all_states[sim, : :] = x
        u = np.load('../../Results/Oscillator/IGP/Sim{}/u_IGP.npy'.format(sim))
        all_controls[sim,:] = u

        with open("../../Results/Oscillator/Adjoint/Sim{}/best_ind_structure_IGP_adj_opt_fdiff.txt".format(sim)) as f:
            print('IGP Adjoint Sim {}: '.format(sim), f.read())
        fit_adj = np.load('../../Results/Oscillator/Adjoint/Sim{}/best_fitness_IGP_adj_opt_fdiff.npy'.format(sim))
        fits_adj[sim] = fit_adj
        fits_opt[sim] = np.load('../../Results/Oscillator/Adjoint/Sim{}/best_fitness_IGP_adj_opt_fdiff.npy'.format(sim))
        if fits_opt[sim] < best_fit_adj:
            best_sim_adj = sim
            best_fit_adj = fits_opt[sim]
        comp_time_adj = np.load('../../Results/Oscillator/Adjoint/Sim{}/computational_time_IGP_adj.npy'.format(sim))
        times_adj[sim] = comp_time_adj
        fit_evol_adj = np.load('../../Results/Oscillator/Adjoint/Sim{}/fitness_evol_IGP_adj.npy'.format(sim))[:,0]
        fit_evols_adj[sim,:] = fit_evol_adj
        x_adj = np.load('../../Results/Oscillator/Adjoint/Sim{}/x_IGP_adj_opt_fdiff.npy'.format(sim))
        all_states_adj[sim, : :] = x_adj
        u_adj = np.load('../../Results/Oscillator/Adjoint/Sim{}/u_IGP_adj_opt_fdiff.npy'.format(sim))
        all_controls_adj[sim,:] = u_adj
    with open("../../Results/Oscillator/IGP/Sim{}/best_ind_structure_IGP.txt".format(best_sim)) as f:
        print('IGP Best Sim {}: '.format(best_sim), f.read())
    print('Best IGP fit ', best_fit)
    with open("../../Results/Oscillator/Adjoint/Sim{}/best_ind_structure_IGP_adj_opt_fdiff.txt".format(best_sim_adj)) as f:
        print('IGP Adjoint Best Sim {}: '.format(best_sim_adj), f.read())
    print('Best IGP adjoint fit ', best_fit_adj)


    fig = plt.figure(0, figsize=[11, 6])
    ax = fig.add_subplot(111)
    plt.plot(time, x_ref[:, 0], '--k', linewidth=2, label='Reference', zorder=8)
    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            plt.plot(time, all_states[i, :, 0], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            plt.plot(time, all_states[i, :, 0], '--', color=color, alpha=alpha, zorder=1)
        color = 'tab:orange'
        if i == best_sim_adj:
            alpha = 1
            plt.plot(time, all_states_adj[i, :, 0], color=color, label='Best OPGD-IGP', zorder=10, linewidth=2)
        else:
            alpha = 0.4
            plt.plot(time, all_states_adj[i, :, 0], '--', color=color, alpha=alpha, zorder=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')

    axins = inset_axes(ax, width=4, height=3, borderpad=1)
    axins.plot(time, x_ref[:, 0], '--k', linewidth=2, label='Reference', zorder=8)
    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            axins.plot(time, all_states[i, :, 0], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            axins.plot(time, all_states[i, :, 0], '--', color=color, alpha=alpha, zorder=1)
        color = 'tab:orange'
        if i == best_sim_adj:
            alpha = 1
            axins.plot(time, all_states_adj[i, :, 0], color=color, label='Best OPGD-IGP', zorder=10, linewidth=2)
        else:
            alpha = 0.4
            axins.plot(time, all_states_adj[i, :, 0], '--', color=color, alpha=alpha, zorder=1)
    axins.set(xlim=(2, 4.5), ylim=(-0.1, 0.4))
    plt.legend(loc='upper right')
    plt.savefig('../../Results/Oscillator/Plots/x_oscillator.pdf', format='pdf', bbox_inches='tight')


    fig = plt.figure(1, figsize=[11, 7])
    ax = fig.add_subplot(111)
    plt.plot(time, x_ref[:, 1], '--k', linewidth=2, label='Reference', zorder=8)
    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            plt.plot(time, all_states[i, :, 1], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            plt.plot(time, all_states[i, :, 1], '--', color=color, alpha=alpha, zorder=1)
        color = 'tab:orange'
        if i == best_sim_adj:
            alpha = 1
            plt.plot(time, all_states_adj[i, :, 1], color=color, label='Best OPGD-IGP', zorder=10, linewidth=2)
        else:
            alpha = 0.4
            plt.plot(time, all_states_adj[i, :, 1], '--', color=color, alpha=alpha, zorder=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [m/s]')
    plt.legend(loc=(0.6, 0.65))

    axins = inset_axes(ax, width=4, height=3, borderpad=1.5, loc=4)
    axins.plot(time, x_ref[:, 1], '--k', linewidth=2, label='Reference', zorder=8)
    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            axins.plot(time, all_states[i, :, 1], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            axins.plot(time, all_states[i, :, 1], '--', color=color, alpha=alpha, zorder=1)
        color = 'tab:orange'
        if i == best_sim_adj:
            alpha = 1
            axins.plot(time, all_states_adj[i, :, 1], color=color, label='Best OPGD-IGP', zorder=10, linewidth=2)
        else:
            alpha = 0.4
            axins.plot(time, all_states_adj[i, :, 1], '--', color=color, alpha=alpha, zorder=1)
    axins.set(xlim=(0.5, 4), ylim=(-2.3, 0.1))

    plt.savefig('../../Results/Oscillator/Plots/v_oscillator.pdf', format='pdf', bbox_inches='tight')

    fig = plt.figure(2, figsize=[11, 7])
    ax = fig.add_subplot(111)
    plt.plot(time, u_ref, '--k', linewidth=2, label='Reference', zorder=8)
    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            plt.plot(time, all_controls[i, :], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            plt.plot(time, all_controls[i, :], '--', color=color, alpha=alpha, zorder=1)
        color = 'tab:orange'
        if i == best_sim_adj:
            alpha = 1
            plt.plot(time, all_controls_adj[i, :], color=color, label='Best OPGD-IGP', zorder=10, linewidth=2)
        else:
            alpha = 0.4
            plt.plot(time, all_controls_adj[i, :], '--', color=color, alpha=alpha, zorder=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Control force [N]')
    plt.legend(loc='upper right')

    axins = inset_axes(ax, width=4, height=3, borderpad=1.5, loc=4)
    axins.plot(time, u_ref, '--k', linewidth=2, label='Reference', zorder=8)
    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            axins.plot(time, all_controls[i, :], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            axins.plot(time, all_controls[i, :], '--', color=color, alpha=alpha, zorder=1)
        color = 'tab:orange'
        if i == best_sim_adj:
            alpha = 1
            axins.plot(time, all_controls_adj[i, :], color=color, label='Best OPGD-IGP', zorder=10, linewidth=2)
        else:
            alpha = 0.4
            axins.plot(time, all_controls_adj[i, :], '--', color=color, alpha=alpha, zorder=1)
    axins.set(xlim=(0.5, 5), ylim=(-0.5, 3.7))

    plt.savefig('../../Results/Oscillator/Plots/u_oscillator.pdf', format='pdf', bbox_inches='tight')


    figg = plt.figure(6)
    figg.set_size_inches(11, 6)
    plt.boxplot([fits, fits_opt], notch=False, showfliers=False)
    plt.plot([0.5,1,2,2.5], np.ones(4) * 56.152032282666816, '--k', label='Reference')
    plt.grid()
    plt.xticks([1, 2], ['IGP', 'OPGD-IGP'])
    plt.ylabel("Fitness")
    plt.legend(loc='best')
    plt.savefig('../../Results/Oscillator/Plots/fitness_oscillator.pdf', format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=[11, 6])
    plt.plot(np.linspace(0, 300, n_gens), np.mean(fit_evols, axis=0), label='IGP', color='tab:blue')
    plt.fill_between(np.linspace(0, 300, n_gens), np.mean(fit_evols, axis=0) - np.std(fit_evols, axis=0),
                     np.mean(fit_evols, axis=0) + np.std(fit_evols, axis=0), alpha=0.2, color='tab:blue')
    plt.plot(np.linspace(0, 300, n_gens), np.mean(fit_evols_adj, axis=0), label='OPGD-IGP', color='tab:orange')
    plt.fill_between(np.linspace(0, 300, n_gens), np.mean(fit_evols_adj, axis=0) - np.std(fit_evols_adj, axis=0),
                     np.mean(fit_evols_adj, axis=0) + np.std(fit_evols_adj, axis=0), alpha=0.2, color='tab:orange')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    axins = inset_axes(ax, width=5, height=3.5, borderpad=1.5)
    axins.set(xlim=(-1, 100), ylim=(56, 56.4))
    axins.plot(np.linspace(0, 300, n_gens), np.mean(fit_evols, axis=0), label='IGP', color='tab:blue')
    axins.fill_between(np.linspace(0, 300, n_gens), np.mean(fit_evols, axis=0) - np.std(fit_evols, axis=0),
                     np.mean(fit_evols, axis=0) + np.std(fit_evols, axis=0), alpha=0.2, color='tab:blue')
    axins.plot(np.linspace(0, 300, n_gens), np.mean(fit_evols_adj, axis=0), label='OPGD-IGP', color='tab:orange')
    axins.fill_between(np.linspace(0, 300, n_gens), np.mean(fit_evols_adj, axis=0) - np.std(fit_evols_adj, axis=0),
                     np.mean(fit_evols_adj, axis=0) + np.std(fit_evols_adj, axis=0), alpha=0.2, color='tab:orange')
    plt.legend(loc='upper right')
    plt.savefig('../../Results/Oscillator/Plots/fit_evol_oscillator.pdf', format='pdf', bbox_inches='tight')

    plt.show()





