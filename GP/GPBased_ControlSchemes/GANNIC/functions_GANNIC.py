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

"""Functions used to run the GANNIC scheme"""


import GP.GP_Algorithms.IGP.IGP_Functions as gpfuns
import numpy as np
from scipy.integrate import simps
import GP.GPBased_ControlSchemes.utils.integration_schemes as int_schemes
from deap import tools
from copy import deepcopy, copy
import GP.GPBased_ControlSchemes.GANNIC.FESTIP_reentry.Plant_models as mods
import GP.GPBased_ControlSchemes.GANNIC.FESTIP_reentry.surrogate_models as surr_mods
import numpy.matlib
import matplotlib.pyplot as plt
import GP.GPBased_ControlSchemes.GANNIC.utils as utils
import GP.GPBased_ControlSchemes.GANNIC.utils_NN as utils_NN

def main(toolbox, load_population, pop_path, creator, pop_random_seed, size_pop_toPass, size_pop, size_pop_tot,
         hof_size, size_gen, Mu, Lambda, cxpb, mutpb, pset, obj, nbCPU, v_points, h_points, train_unc_profiles,
         uncertain, uncertain_atmo, uncertain_aero, save_gen, save_path, start_gen, fit_tol):
    """
    Main function that runs the GP evolutionary process in the GANNIC scheme

    Attributes:
        toolbox: class
            GP building blocks
        load_population: bool
            whether to load or not an old population
        pop_path: str
            path to old population
        creator: class
            GP building blocks
        pop_random_seed: bool
            whether initialize randomly a part of the population
        size_pop_toPass: int
            individuals to pass to new simulation
        size_pop: int
            population's size
        size_pop_tot: int
            population's size
        hof_size: int
            hall of fame size
        size_gen: int
            number of generations
        Mu: int
            number of individuals to select in tournament
        Lambda: int
            size of offspring
        cxpb: float
            crossover probability
        mutpb: float
            mutation probability
        pset: class
            primitive set
        obj: class
            plant's model
        nbCPU: int
            cores to use for multiprocessing
        v_points: array
            points used to evaluate uncertainty
        h_points: array
            points used to evaluate uncertainty
        train_unc_profiles: array
            training uncertainty profiles
        uncertain: bool
            whether to use uncertainty or not
        uncertain_atmo: bool
            whether to use atmospheric uncertainty or not
        uncertain_aero: bool
            whether to use aerodynamic uncertainty or not
        save_gen: int
            every how many generations to save data
        save_path: str
            path to save folder
        start_gen: int
            initial generation. Different from zero if resuming old simulation
        fit_tol: float
            fitness threshold below which evolution is stopped

    Return:
        pop: list
            final population
        hof: class
            hall of fame
        log: class
            logbook of evolution
    """
    #  Perform search for the initial population with the highest entropy
    if load_population is True:
        pop = gpfuns.POP(np.load(pop_path, allow_pickle=True), creator)
        best_pop = pop.items
        if pop_random_seed:
            hof_toPass = gpfuns.HallOfFame(size_pop_toPass)
            hof_toPass.update(best_pop, for_feasible=False)  # pick best individuals used to start GP if random seed is selected
            best_pop = copy(hof_toPass.shuffle())
    else:
        old_entropy = 0
        for i in range(200):
            pop = gpfuns.POP(toolbox.population(n=size_pop), creator)
            best_pop = pop.items
            if pop.entropy > old_entropy and len(pop.indexes) == len(pop.categories) - 1:
                best_pop = pop.items
                old_entropy = pop.entropy

    if pop_random_seed is True:
        for ind in best_pop:
            del ind.fitness.values
        pop_random = toolbox.population(n=size_pop_tot - len(best_pop))
        best_pop = pop_random + best_pop

    hof = gpfuns.HallOfFame(hof_size)

    print("INITIAL POP SIZE: %d" % size_pop)
    print("GEN SIZE: %d" % size_gen)
    print("\n")

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", gpfuns.Min)

    ####################################   EVOLUTIONARY ALGORITHM - EXECUTION   ###################################

    pop, log, pop_statistics, ind_lengths = gpfuns.eaMuPlusLambdaTol_GANNIC(best_pop, toolbox, Mu, Lambda, size_gen, cxpb,
                                                                     mutpb, pset, creator, stats=mstats,
                                                                     halloffame=hof, verbose=True, fit_tol=fit_tol,
                                                                     obj=obj, v_points=v_points, h_points=h_points,
                                                                     train_unc_profiles=train_unc_profiles,
                                                                     uncertain=uncertain, uncertain_aero=uncertain_aero,
                                                                     uncertain_atmo=uncertain_atmo, save_gen=save_gen,
                                                                     save_path=save_path, nbCPU=nbCPU,
                                                                     start_gen=start_gen)

    ####################################################################################################################

    return pop, log, hof


def evaluateLearningNodes(index_ind, pset, **k):
    """
    Wrapper function that calls the function used to evaluate the individuals. This wrapper is used to create the NN model

    Attributes:
        index_ind: int
            index of individual evaluated
        pset: class
            primitive set
        **k: kargs

    Return:
        fit: float
            fitness
        penalty: float
            constriant's violation
        fit_components: array
            fitness on every uncertainty scenario
        success: int
            number of successes on uncertainty scenarios
    """

    individual = deepcopy(index_ind[1])
    obj = k['kwargs']['obj']
    v_points = k['kwargs']['v_points']
    h_points = k['kwargs']['h_points']
    train_unc_profiles = k['kwargs']['train_unc_profiles']
    uncertain = k['kwargs']['uncertain']
    uncertain_aero = k['kwargs']['uncertain_aero']
    uncertain_atmo = k['kwargs']['uncertain_atmo']
    ubc = obj.initial_ubc
    ubd = obj.initial_ubd

    NNmodel, n_weights = utils_NN.create_NN_model(obj.n_inputNN, obj.n_hidden_layers, obj.nodes_per_hidden_layer,
                                                  obj.NNoutput, obj.activation)

    fit, penalty, successes, fit_components = evaluateIndividual(individual, v_points, h_points, train_unc_profiles,
                                                                 obj, NNmodel, uncertain, uncertain_aero,
                                                                 uncertain_atmo, pset, n_weights, ubc, ubd)

    del individual, obj, v_points, h_points, train_unc_profiles, uncertain, k, uncertain_aero, uncertain_atmo, \
        NNmodel

    return [fit, penalty, fit_components, successes]


def evaluateIndividual(GPind, v_points, h_points, train_unc_profiles, obj, NNmodel, uncertain, uncertain_aero,
                       uncertain_atmo, pset, n_weights, ubc, ubd):

    """
    This function is used to evaluate the fitness function of the GP individuals

    Attributes:
        GPind: list
            GP individuals
        v_points: array
            points to evaluate uncertainty
        h_points: array
            points to evaluate uncertainty
        train_unc_profiles: array
            training uncertainty profiles
        obj: class
            plant's model
        NNmodel: tensorflow mode
            NN model
        uncertain: bool
            whether to use or not uncertainty
        uncertain_aero: bool
            whether to use or not aerodynamic uncertainty
        uncertain_atmo: bool
            whether to use or not atmospheric uncertainty
        pset: class
            primitive set
        n_weights: int
            number of weights in NN model
        ubc: float
            upper bound on atmospheric uncertainty on c
        ubd: float
            upper bound on atmosheric uncertainty on rho

    Return:
        FIT_final: float
            fitness
        penalty_final: float
            constriant's violation
        FIT_TOT: array
            fitness on every uncertainty scenario
        all_success: int
            number of successes on uncertainty scenarios

    """
    penalty = []

    FIT_TOT = []
    penalty_tot = []

    successes = 1
    unc_indexes = np.load('uncert_indexes.npy', allow_pickle=True)
    all_successes = []

    for nt in unc_indexes:
        partial_success = 0
        uncertFuns = utils.select_uncertainty_profile(v_points, h_points, train_unc_profiles, nt)
        ### Direct problem solution
        init_conds = np.concatenate((obj.init_conds, np.array((obj.NNinitConds))))
        sol_forward = int_schemes.RK4(obj.t0, obj.tf, mods.forward_dynamicsNNUncert, obj.Npoints, init_conds,
                                      args=(obj, uncertain, uncertain_atmo, uncertain_aero, uncertFuns, NNmodel, GPind,
                                            pset, n_weights, ubc, ubd))

        z = sol_forward.T

        if np.isnan(z).any() or np.isinf(z).any():
            FIT_TOT.append(1e6)
            penalty_tot.append(1e6)
            all_successes.append(0)
            continue

        v = z[0, :]
        h = z[5, :]
        sts_tot = np.array(([z[0, :], z[1, :], z[2, :], z[3, :], z[4, :], z[5, :]]))
        refs_tot = np.array(([obj.states_refs_funs['vfun'](obj.t_points), obj.states_refs_funs['chifun'](obj.t_points),
                              obj.states_refs_funs['gammafun'](obj.t_points), obj.states_refs_funs['tetafun'](obj.t_points),
                              obj.states_refs_funs['lamfun'](obj.t_points), obj.states_refs_funs['hfun'](obj.t_points)])).transpose()
        LB = np.matlib.repmat(np.array(([obj.vmin, obj.chimin, obj.gammamin, obj.tetamin, obj.lammin, obj.hmin])),
                              len(obj.t_points), 1)
        UB = np.matlib.repmat(np.array(([obj.vmax, obj.chimax, obj.gammamax, obj.tetamax, obj.lammax, obj.hmax])),
                              len(obj.t_points), 1)
        norm_refs_tot = (refs_tot - LB) / (UB - LB)
        norm_states_tot = (sts_tot.T - LB) / (UB - LB)

        all_errs_tot = norm_states_tot - norm_refs_tot
        errs_tot = copy(all_errs_tot[:, :3])

        rho, c = utils.check_uncertainty('atmo', uncertain, uncertain_atmo, uncertain_aero, uncertFuns, obj, h, v, ubd,
                                         ubc, 0, 0)

        alfa = np.zeros(obj.Npoints)

        for j in range(obj.Npoints):
            NNmodel = utils_NN.update_NN_weights(NNmodel, z[6:, j], n_weights)
            input = np.array([[*errs_tot[j,:]]])
            NNoutput = np.ravel(NNmodel(input).numpy())
            NNoutput = np.nan_to_num(NNoutput)
            eps_cont = surr_mods.control_variation(abs(all_errs_tot[j,:]), obj.lb_contr, obj.ub_contr)
            alfa[j] = obj.controls_refs_funs['alfafun'](obj.t_points[j]) * \
                      (1 - eps_cont + (eps_cont / obj.tanh_param_out) *
                       (obj.tanh_param_out * np.tanh(obj.tanh_param_in * NNoutput[0]) + obj.tanh_param_out))

        alfa[alfa>obj.alfamax] = obj.alfamax
        alfa[alfa<obj.alfamin] = obj.alfamin
        alfa[np.isinf(alfa)] = obj.alfamax
        alfa[np.isnan(alfa)] = obj.alfamin

        M = v / c

        cL, cD = utils.check_uncertainty('aero', uncertain, uncertain_atmo, uncertain_aero, uncertFuns, obj, h, v, ubd,
                                         ubc, M, alfa)

        L = 0.5 * (v ** 2) * obj.wingSurf * rho * cL
        D = 0.5 * (v ** 2) * obj.wingSurf * rho * cD

        q = 0.5 * rho * (v ** 2)

        # Heat transfer
        Q = obj.Cfin * np.sqrt(rho) * (v ** 3) * (1 - 0.18 * (np.sin(obj.Lam) ** 2)) * np.cos(obj.Lam)

        # Load factor
        az = (D * np.sin(alfa) + L * np.cos(alfa)) / obj.m

        qq = obj.Maxq - q
        QQ = obj.MaxQ - Q
        AZ_plus = obj.MaxAz - az
        AZ_minus = az + obj.MaxAz
        penalty.extend(qq[qq < 0] / obj.Maxq)
        penalty.extend(QQ[QQ < 0] / obj.MaxQ)
        penalty.extend(AZ_plus[AZ_plus < 0])
        penalty.extend(AZ_minus[AZ_minus < 0])

        if abs(obj.tetaend - z[3,-1]) <= obj.tetaend_tol:
            all_errs_tot[-1, 3] = 0
            partial_success += 1

        if abs(obj.lamend - z[4,-1]) <= obj.lamend_tol:
            all_errs_tot[-1, 4] = 0
            partial_success += 1

        if abs(obj.hend - z[5,-1]) <= obj.hend_tol:
            all_errs_tot[-1, 5] = 0
            partial_success += 1
        if partial_success == 3:
            successes += 1
            all_successes.append(1)
        else:
            all_successes.append(0)

        final_points = int(obj.Npoints*0.3)
        fit1 = simps(abs(all_errs_tot[-final_points:, 0]), obj.t_points[-final_points:])
        fit2 = simps(abs(all_errs_tot[-final_points:, 1]), obj.t_points[-final_points:])
        fit3 = simps(abs(all_errs_tot[-final_points:, 2]), obj.t_points[-final_points:])
        fit4 = simps(abs(all_errs_tot[-final_points:, 3]), obj.t_points[-final_points:])
        fit5 = simps(abs(all_errs_tot[-final_points:, 4]), obj.t_points[-final_points:])
        fit6 = simps(abs(all_errs_tot[-final_points:, 5]), obj.t_points[-final_points:])

        globalFit = np.array(([fit1, fit2, fit3, fit4, fit5, fit6]))
        globalFit = np.hstack((globalFit/1e3, all_errs_tot[-1,:] * obj.fit_weights*1e3))

        FIT = (1 / len(globalFit)) * sum(globalFit ** 2)
        FIT_TOT.append(FIT)

        if penalty != []:
            pen = np.sqrt(1 / len(penalty) * sum(np.array(penalty) ** 2))
            penalty_tot.append(pen)

    if obj.n_train == 1:
        FIT_final = FIT_TOT[0]
    else:
        FIT_final = (np.max(np.nan_to_num(FIT_TOT)) + np.median(np.nan_to_num(FIT_TOT)))/max(sum(all_successes), 1)

    if penalty_tot != []:
        if obj.n_train == 1:
            penalty_final = penalty_tot[0]
        else:
            penalty_final = np.nan_to_num((1/len(penalty_tot))*sum(np.array((penalty_tot)) ** 2))
        del penalty_tot
        return [FIT_final, penalty_final, all_successes, FIT_TOT]
    else:
        return [FIT_final, 0.0, all_successes, FIT_TOT]


def plot_best_ind(obj, v_points, h_points, uncertain, uncertain_atmo, uncertain_aero, pset, savedata_file, best_ind,
                  test_unc_profiles, NNmodel, n_weights, NNinitName, toPlot, ubd, ubc):

    """
    Function used to plot the best individual at the end of the evolutionary process

    Attributes:
        obj: class
            plant's model
        v_points: array
            points used to evaluate uncertainty
        h_points: array
            points used to evaluate uncertainty
        uncertain: bool
            whether to use uncertainty or not
        uncertain_atmo: bool
            whether to use atmospheric uncertainty or not
        uncertain_aero: bool
            whether to use aerodynamic uncertainty or not
        pset: class
            primitive set
        savedata_file: str
            path to save folder
        best_ind: list
            GP individual
        test_unc_profiles: array
            test uncertainty profiles
        NNmodel: tensorflow model
            NN model
        n_weights: int
            number of weights in NN model
        NNinitName: str
            name of file containing NN model
        toPlot: bool
            whether to plot the results
        ubd: float
            upper bound on atmospheric uncertainty applied on rho
        ubc: float
            upper bound on atmospheric uncertainty applied on c

    Return:
        successes: int
            number of successes
    """
    obj.NNinitConds = np.load(NNinitName)

    successes = 0

    for l in range(obj.n_test):

        uncertFuns = utils.select_uncertainty_profile(v_points, h_points, test_unc_profiles, l)
        init_conds = np.concatenate((obj.init_conds, np.array((obj.NNinitConds))))

        res = int_schemes.RK4(obj.t0, obj.tf, mods.forward_dynamicsNNUncert, obj.Npoints, init_conds,
                              args=(obj, uncertain, uncertain_atmo, uncertain_aero, uncertFuns, NNmodel, best_ind,
                                    pset, n_weights, ubd, ubc))

        z = res.T

        states = z[:6,:]
        refs = np.array(([obj.states_refs_funs['vfun'](obj.t_points), obj.states_refs_funs['chifun'](obj.t_points),
                          obj.states_refs_funs['gammafun'](obj.t_points), obj.states_refs_funs['tetafun'](obj.t_points),
                          obj.states_refs_funs['lamfun'](obj.t_points), obj.states_refs_funs['hfun'](obj.t_points)])).T
        LB = np.matlib.repmat(np.array(([obj.vmin, obj.chimin, obj.gammamin, obj.tetamin, obj.lammin, obj.hmin])),
                              len(obj.t_points), 1)
        UB = np.matlib.repmat(np.array(([obj.vmax, obj.chimax, obj.gammamax, obj.tetamax, obj.lammax, obj.hmax])),
                              len(obj.t_points), 1)
        norm_refs = (refs - LB) / (UB - LB)
        norm_states = (states.T - LB) / (UB - LB)

        all_errs = norm_states.T - norm_refs.T
        errs = copy(all_errs[:3, :])

        az = []
        Q = []
        q = []
        alfa = []
        sigma = []
        alfaNN = []
        sigmaNN = []
        c_viol = 0
        delta_alfa = []
        delta_sigma = []
        alfaNNoutput = np.zeros(len(z[0,:]))
        sigmaNNoutput = np.zeros(len(z[0, :]))
        alfa_final = np.zeros(len(z[0, :]))
        sigma_final = np.zeros(len(z[0, :]))
        for i in range(len(z[0,:])):

            rho, c = utils.check_uncertainty('atmo', uncertain, uncertain_atmo, uncertain_aero, uncertFuns, obj, z[5,i],
                                             0, ubd, ubc, 0, 0)

            NNmodel = utils_NN.update_NN_weights(NNmodel, z[6:, i], n_weights)
            input = np.array([[*errs[:,i]]])
            NNoutput = np.ravel(NNmodel(input).numpy())
            NNoutput = np.nan_to_num(NNoutput)
            alfaNNoutput[i] = NNoutput[0]
            sigmaNNoutput[i] = NNoutput[1]
            eps_cont = surr_mods.control_variation(abs(all_errs[:,i]), obj.lb_contr, obj.ub_contr)
            alfaNN_part = 1 - eps_cont + (eps_cont / obj.tanh_param_out) * (obj.tanh_param_out * np.tanh(obj.tanh_param_in * NNoutput[0]) + obj.tanh_param_out)
            sigmaNN_part = 1 - eps_cont + (eps_cont / obj.tanh_param_out) * (obj.tanh_param_out * np.tanh(obj.tanh_param_in * NNoutput[1]) + obj.tanh_param_out)

            final_output_alfa = 1 + obj.ub_contr*sum(abs(all_errs[:,i]))*np.tanh(obj.tanh_param_in * NNoutput[0])
            final_output_sigma = 1 + obj.ub_contr * sum(abs(all_errs[:, i])) * np.tanh(obj.tanh_param_in * NNoutput[1])

            alfa_final[i] = final_output_alfa
            sigma_final[i] = final_output_sigma

            alfaNN.append(alfaNN_part)
            sigmaNN.append(sigmaNN_part)
            alfa_part = obj.controls_refs_funs['alfafun'](obj.t_points[i]) * alfaNN_part
            sigma_part = obj.controls_refs_funs['sigmafun'](obj.t_points[i]) * sigmaNN_part
            delta_alfa.append(alfa_part - obj.controls_refs_funs['alfafun'](obj.t_points[i]))
            delta_sigma.append(sigma_part - obj.controls_refs_funs['sigmafun'](obj.t_points[i]))
            if sigma_part > obj.sigmamax or np.isinf(sigma_part):
                sigma_part = obj.sigmamax
            elif sigma_part < obj.sigmamin or np.isnan(sigma_part):
                sigma_part = obj.sigmamin
            if alfa_part > obj.alfamax or np.isinf(alfa_part):
                alfa_part = obj.alfamax
            elif alfa_part < obj.alfamin or np.isnan(alfa_part):
                alfa_part = obj.alfamin
            alfa.append(alfa_part)
            sigma.append(sigma_part)
            if np.isnan(z[0, i]):
                M = 0
            elif np.isinf(z[0, i]):
                M = 1e6 / c
            else:
                M = z[0, i] / c

            cL, cD = utils.check_uncertainty('aero', uncertain, uncertain_atmo, uncertain_aero, uncertFuns, obj,
                                             z[5, i], z[0,i], ubd, ubc, M, alfa_part)

            L = 0.5 * (z[0,i] ** 2) * obj.wingSurf * rho * cL
            D = 0.5 * (z[0,i] ** 2) * obj.wingSurf * rho * cD

            qi = 0.5 * rho * (z[0,i] ** 2)

            q.append(qi)
            if qi > obj.Maxq:
                c_viol += 1
            # Heat transfer
            Q.append(obj.Cfin * np.sqrt(rho) * (z[0,i] ** 3) * (1 - 0.18 * (np.sin(obj.Lam) ** 2)) * np.cos(obj.Lam))

            # Load factor
            azi = (D * np.sin(alfa_part) + L * np.cos(alfa_part)) / obj.m
            az.append(azi)
            if azi > obj.MaxAz or azi < -obj.MaxAz:
                c_viol += 1

        if (obj.hend - obj.hend_tol <= z[5,-1] <= obj.hend + obj.hend_tol) and \
                (obj.lamend - obj.lamend_tol <= z[4,-1] <= obj.lamend + obj.lamend_tol) and \
                (obj.tetaend - obj.tetaend_tol <= z[3,-1] <= obj.tetaend + obj.tetaend_tol) and c_viol == 0:
            successes += 1

        if toPlot is True:

            plt.figure(1)
            plt.plot(obj.t_points, res[:,0])

            plt.figure(2)
            plt.plot(obj.t_points, np.rad2deg(res[:,1]))

            plt.figure(3)
            plt.plot(obj.t_points, np.rad2deg(res[:, 2]))

            plt.figure(4)
            plt.plot(obj.t_points, np.rad2deg(res[:, 3]))

            plt.figure(5)
            plt.plot(obj.t_points, np.rad2deg(res[:, 4]))

            plt.figure(6)
            plt.plot(obj.t_points, res[:, 5])

            plt.figure(7)
            plt.plot(obj.t_points, q)

            plt.figure(8)
            plt.plot(obj.t_points, Q)

            plt.figure(9)
            plt.plot(obj.t_points, az)

            plt.figure(10)
            plt.plot(obj.t_points, np.rad2deg(alfa))

            plt.figure(11)
            plt.plot(obj.t_points, np.rad2deg(sigma))

            plt.figure(12)
            plt.plot(obj.t_points, alfaNN)

            plt.figure(13)
            plt.plot(obj.t_points, sigmaNN)

            plt.figure(14)
            plt.plot(obj.t_points, np.rad2deg(delta_alfa))

            plt.figure(15)
            plt.plot(obj.t_points, np.rad2deg(delta_sigma))

            plt.figure(16)
            plt.plot(obj.t_points, alfaNNoutput)

            plt.figure(17)
            plt.plot(obj.t_points, sigmaNNoutput)

            for i in range(len(z[6:,0])):
                plt.figure(18+i)
                plt.plot(obj.t_points, z[6+i,:],)

    print('Successes {}/{}'.format(successes, obj.n_test))

    if toPlot is True:
        plt.figure(1)
        plt.plot(obj.t_points, obj.states_refs_funs['vfun'](obj.t_points), '--r', label='Ref')
        plt.xlabel('Time [s]')
        plt.ylabel('Speed [m]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'v', format='pdf')

        plt.figure(2)
        plt.plot(obj.t_points, np.rad2deg(obj.states_refs_funs['chifun'](obj.t_points)), '--r', label='Ref')
        plt.xlabel('Time [s]')
        plt.ylabel('Chi [deg]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'chi', format='pdf')

        plt.figure(3)
        plt.plot(obj.t_points, np.rad2deg(obj.states_refs_funs['gammafun'](obj.t_points)), '--r', label='Ref')
        plt.xlabel('Time [s]')
        plt.ylabel('Gamma [deg]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'gamma', format='pdf')

        plt.figure(4)
        plt.plot(obj.t_points, np.rad2deg(obj.states_refs_funs['tetafun'](obj.t_points)), '--r', label='Ref')
        plt.axhline(y=np.rad2deg(obj.tetaend + obj.tetaend_tol), color='k', linestyle='-')
        plt.axhline(y=np.rad2deg(obj.tetaend - obj.tetaend_tol), color='k', linestyle='-')
        plt.xlabel('Time [s]')
        plt.ylabel('Teta [deg]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'teta', format='pdf')

        plt.figure(5)
        plt.plot(obj.t_points, np.rad2deg(obj.states_refs_funs['lamfun'](obj.t_points)), '--r', label='Ref')
        plt.axhline(y=np.rad2deg(obj.lamend+obj.lamend_tol), color='k', linestyle='-')
        plt.axhline(y=np.rad2deg(obj.lamend-obj.lamend_tol), color='k', linestyle='-')
        plt.xlabel('Time [s]')
        plt.ylabel('Lambda [deg]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'lam', format='pdf')

        plt.figure(6)
        plt.plot(obj.t_points, obj.states_refs_funs['hfun'](obj.t_points), '--r', label='Ref')
        plt.axhline(y=obj.hend+obj.hend_tol, color='k', linestyle='-')
        plt.axhline(y=obj.hend-obj.hend_tol, color='k', linestyle='-')
        plt.xlabel('Time [s]')
        plt.ylabel('Altitude m')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'h', format='pdf')

        plt.figure(7)
        plt.axhline(y=obj.Maxq, color='r', linestyle='-')
        plt.xlabel('Time [s]')
        plt.ylabel('Dynamic Pressure [Pa]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'q', format='pdf')

        plt.figure(8)
        plt.axhline(y=obj.MaxQ, color='r', linestyle='-')
        plt.xlabel('Time [s]')
        plt.ylabel('Heat rate [W/m^2]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'Q', format='pdf')

        plt.figure(9)
        plt.axhline(y=obj.MaxAz, color='r', linestyle='-')
        plt.axhline(y=-obj.MaxAz, color='r', linestyle='-')
        plt.xlabel('Time [s]')
        plt.ylabel('Tangential Acceleration [m/s^2]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'az', format='pdf')

        plt.figure(10)
        plt.axhline(y=np.rad2deg(obj.alfamax), color='r', linestyle='-')
        plt.axhline(y=np.rad2deg(obj.alfamin), color='r', linestyle='-')
        plt.xlabel('Time [s]')
        plt.ylabel('Alfa [deg]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'alfa', format='pdf')

        plt.figure(11)
        plt.axhline(y=np.rad2deg(obj.sigmamax), color='r', linestyle='-')
        plt.axhline(y=np.rad2deg(obj.sigmamin), color='r', linestyle='-')
        plt.xlabel('Time [s]')
        plt.ylabel('Sigma [deg]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'sigma', format='pdf')

        plt.figure(12)
        plt.xlabel('Time [s]')
        plt.ylabel('Alfa NN with scaler [rad]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'alfa_nn_scaler.pdf', format='pdf')

        plt.figure(13)
        plt.xlabel('Time [s]')
        plt.ylabel('Sigma NN with scaler [rad]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'sigma_nn_scaler.pdf', format='pdf')

        plt.figure(14)
        plt.xlabel('Time [s]')
        plt.ylabel('Delta alfa [deg]')
        plt.legend(loc='best')

        plt.figure(15)
        plt.xlabel('Time [s]')
        plt.ylabel('Delta sigma [deg]')
        plt.legend(loc='best')

        plt.figure(16)
        plt.xlabel('Time [s]')
        plt.ylabel('NN output alfa [rad]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'alfa_nn.pdf', format='pdf')

        plt.figure(17)
        plt.xlabel('Time [s]')
        plt.ylabel('NN output sigma [rad]')
        plt.legend(loc='best')
        plt.savefig(savedata_file + 'sigma_nn.pdf', format='pdf')

        for i in range(len(z[6:,0])):
            plt.figure(18+i)
            plt.title('weight {}'.format(i+1))
            plt.savefig(savedata_file + 'w{}'.format(i+1), format='pdf')
        plt.show()


    return successes


'''def analyze_control_output(obj, v_points, h_points, uncertain, uncertain_atmo, uncertain_aero, pset, best_ind,
                           test_unc_profiles, NNmodel, n_weights, NNinitName, ubd, ubc):

    obj.NNinitConds = np.load(NNinitName)

    successes = 0
    alfa_output = np.zeros((obj.n_test, obj.Npoints))
    sigma_output = np.zeros((obj.n_test, obj.Npoints))
    for l in range(obj.n_test):
        uncertFuns = utils.select_uncertainty_profile(v_points, h_points, test_unc_profiles, l)
        init_conds = np.concatenate((obj.init_conds, np.array((obj.NNinitConds))))

        res = int_schemes.RK4(obj.t0, obj.tf, mods.forward_dynamicsNNUncert, obj.Npoints, init_conds,
                              args=(obj, uncertain, uncertain_atmo, uncertain_aero, uncertFuns, NNmodel, best_ind,
                                    pset, n_weights, ubd, ubc, None))

        z = res.T


        states = z[:6,:]
        refs = np.array(([obj.states_refs_funs['vfun'](obj.t_points), obj.states_refs_funs['chifun'](obj.t_points),
                          obj.states_refs_funs['gammafun'](obj.t_points), obj.states_refs_funs['tetafun'](obj.t_points),
                          obj.states_refs_funs['lamfun'](obj.t_points), obj.states_refs_funs['hfun'](obj.t_points)])).T
        LB = np.matlib.repmat(np.array(([obj.vmin, obj.chimin, obj.gammamin, obj.tetamin, obj.lammin, obj.hmin])),
                              len(obj.t_points), 1)
        UB = np.matlib.repmat(np.array(([obj.vmax, obj.chimax, obj.gammamax, obj.tetamax, obj.lammax, obj.hmax])),
                              len(obj.t_points), 1)
        norm_refs = (refs - LB) / (UB - LB)
        norm_states = (states.T - LB) / (UB - LB)

        all_errs = norm_states.T - norm_refs.T

        errs = copy(all_errs[:3, :])

        az = []
        Q = []
        q = []
        alfa = []
        sigma = []
        c_viol = 0
        delta_alfa = []
        delta_sigma = []
        for i in range(len(z[0,:])):

            rho, c = utils.check_uncertainty('atmo', uncertain, uncertain_atmo, uncertain_aero, uncertFuns, obj, z[5,i],
                                             0, ubd, ubc, 0, 0)

            NNmodel = utils_NN.update_NN_weights(NNmodel, z[6:, i], n_weights)
            input = np.array([[*errs[:,i]]])
            NNoutput = np.ravel(NNmodel(input).numpy())
            NNoutput = np.nan_to_num(NNoutput)
            eps_cont = surr_mods.control_variation(abs(all_errs[:,i]), obj.lb_contr, obj.ub_contr)
            alfaNN_part = 1 - eps_cont + (eps_cont / obj.tanh_param_out) * (obj.tanh_param_out * np.tanh(obj.tanh_param_in * NNoutput[0]) + obj.tanh_param_out)
            sigmaNN_part = 1 - eps_cont + (eps_cont / obj.tanh_param_out) * (obj.tanh_param_out * np.tanh(obj.tanh_param_in * NNoutput[1]) + obj.tanh_param_out)

            alfa_output[l, i] = NNoutput[0]
            sigma_output[l, i] = NNoutput[1]
            alfa_part = obj.controls_refs_funs['alfafun'](obj.t_points[i]) * alfaNN_part
            sigma_part = obj.controls_refs_funs['sigmafun'](obj.t_points[i]) * sigmaNN_part

            delta_alfa.append(alfa_part - obj.controls_refs_funs['alfafun'](obj.t_points[i]))
            delta_sigma.append(sigma_part - obj.controls_refs_funs['sigmafun'](obj.t_points[i]))


            if sigma_part > obj.sigmamax or np.isinf(sigma_part):
                sigma_part = obj.sigmamax
            elif sigma_part < obj.sigmamin or np.isnan(sigma_part):
                sigma_part = obj.sigmamin
            if alfa_part > obj.alfamax or np.isinf(alfa_part):
                alfa_part = obj.alfamax
            elif alfa_part < obj.alfamin or np.isnan(alfa_part):
                alfa_part = obj.alfamin
            alfa.append(alfa_part)
            sigma.append(sigma_part)
            if np.isnan(z[0, i]):
                M = 0
            elif np.isinf(z[0, i]):
                M = 1e6 / c
            else:
                M = z[0, i] / c

            cL, cD = utils.check_uncertainty('aero', uncertain, uncertain_atmo, uncertain_aero, uncertFuns, obj,
                                             z[5, i], z[0,i], ubd, ubc, M, alfa_part)

            L = 0.5 * (z[0,i] ** 2) * obj.wingSurf * rho * cL
            D = 0.5 * (z[0,i] ** 2) * obj.wingSurf * rho * cD

            qi = 0.5 * rho * (z[0,i] ** 2)

            q.append(qi)
            if qi > obj.Maxq:
                c_viol += 1
            # Heat transfer
            Q.append(obj.Cfin * np.sqrt(rho) * (z[0,i] ** 3) * (1 - 0.18 * (np.sin(obj.Lam) ** 2)) * np.cos(obj.Lam))

            # Load factor
            azi = (D * np.sin(alfa_part) + L * np.cos(alfa_part)) / obj.m
            az.append(azi)
            if azi > obj.MaxAz or azi < -obj.MaxAz:
                c_viol += 1

        if (obj.hend - obj.hend_tol <= z[5,-1] <= obj.hend + obj.hend_tol) and \
                (obj.lamend - obj.lamend_tol <= z[4,-1] <= obj.lamend + obj.lamend_tol) and \
                (obj.tetaend - obj.tetaend_tol <= z[3,-1] <= obj.tetaend + obj.tetaend_tol) and c_viol == 0:
            successes += 1

    return np.median(alfa_output, axis=0), np.median(sigma_output, axis=0), np.std(alfa_output, axis=0), np.std(sigma_output, axis=0)'''

