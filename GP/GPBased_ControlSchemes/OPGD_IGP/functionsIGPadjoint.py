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

"""

This script contains the functions used by the OPGD-IGP algorithm. The OPGD was impemented as presented in [1]

[1] F.-J. Camerota-Verdù, G. Pietropolli, L. Manzoni, M. Castelli, Parametrizing gp trees for better symbolic
regression performance through gradient descent, in: Proceedings of the Genetic and Evolutionary Computation Conference
Companion, 2023.
"""

import sys
import os
dirname = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(dirname, '..')))
sys.path.append(os.path.abspath(os.path.join(dirname, '..', '..', 'test_cases')))
import numpy as np
import GP.GP_Algorithms.IGP.IGP_Functions_V2 as gpfuns
from scipy.integrate import simps
from copy import deepcopy, copy
from deap import gp, creator, base, tools
import operator
import GP.GPBased_ControlSchemes.utils.integration_schemes as int_schemes
from GP.GPBased_ControlSchemes.OPGD_IGP.matrices_functions import ETA_function, fill_sensitivity_matrix, define_matrices
import re
import sympy

def main(nbCPU, toolbox, size_pop, size_gen, creator, Mu, Lambda, pset, obj, opt_steps, mul_fun, dynamics, symbols):
    """Main function used to run the OPGD-IGP evolutionary process

    Attributes:
        nbCPU: integer
            number of cores to use
        toolbox: class
            contains GP building blocks
        size_pop: integer
            number of individuals in the population
        size_gen: integer
            number of generations
        creator: class
            contains GP building blocks
        Mu: integer
            number of individuals selected in mu+lambda process
        Lambda: integer
            number of individuals in offspring
        pset: class
            contains primitive set
        obj: class
            containts plant's parameters
        opt_steps: integer
            optimization steps performed by Adam
        mul_fun: primitive function
            contains the multiplication function
        dynamics: function
            a function describing the plant's dynamics
        symbols: scipy symbols
            scipy symbols used to evaluate the symbolic derivatives
    Return:
        pop: list
            final population
        log: class
            logbook of evolution
        hof: list
            hall of fame of best individuals
        pop_statistics: class
            statistics of the evolutionary process
        ind_lenghts: list
            a list containing the lenghts of the individuals produced in the evolutionary process
    """

    mutpb = 0.7
    cxpb = 0.2

    old_entropy = 0
    for i in range(200):
        pop = gpfuns.POP(toolbox.population(n=size_pop), creator)
        best_pop = pop.items
        if pop.entropy > old_entropy and len(pop.indexes) == len(pop.categories) - 1:
            best_pop = pop.items
            old_entropy = pop.entropy

    hof = gpfuns.HallOfFame(10)

    print("INITIAL POP SIZE: %d" % size_pop)
    print("GEN SIZE: %d" % size_gen)
    print("\n")

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", gpfuns.Min)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   ###################################

    pop, log, pop_statistics, ind_lengths = gpfuns.eaMuPlusLambdaTol_loopInside(best_pop, toolbox, Mu, Lambda, size_gen,
                                                                                cxpb, mutpb, creator, stats=mstats,
                                                                                halloffame=hof, verbose=True, obj=obj,
                                                                                pset=pset, opt_steps=opt_steps,
                                                                                mul_fun=mul_fun, verbose_adam=False,
                                                                                dynamics=dynamics, symbols=symbols,
                                                                                nbCPU=nbCPU)
    ####################################################################################################################


    return pop, log, hof, pop_statistics, ind_lengths


def evaluate_individualIGP_adjoint(GPind, **k):
    """
    Function used to evaluate the individuals in the OPGD-IGP

    Attributes:
        GPind: list
            individual to be evaluated
        **k: problem dependant kwargs

    Return:
        FIT: individual's fitness
    """

    obj = k['kwargs']['obj']
    pset = k['kwargs']['pset']
    opt_steps = k['kwargs']['opt_steps']
    mul_fun = k['kwargs']['mul_fun']
    verbose = k['kwargs']['verbose_adam']
    dynamics = k['kwargs']['dynamics']
    symbols = k['kwargs']['symbols']
    pset.addTerminal('w')
    pset.terminals[pset.ret][-1].value = 1  # set initial value of weights to 1
    weight = deepcopy(pset.terminals[pset.ret][-1])  # store weight terminal


    if opt_steps == 0:
        traj, contr, failure = propagate_forward(obj, GPind, pset, dynamics)
        if failure is True:
            return [1e6, 0]
        else:
            g = np.zeros(obj.Npoints)
            for i in range(obj.Npoints):
                g[i] = 0.5 * ((traj[i,:]-obj.xf).T@obj.Qz@(traj[i,:]-obj.xf) + np.array(([[contr[i]]])).T @ obj.Qu @ np.array(([[contr[i]]])))
            int_g = simps(g, obj.t_points)
            h = 0.5 * ((traj[-1, :]-obj.xf).T @ obj.Qt @ (traj[-1, :]-obj.xf))
            FIT = int_g + h
            return [FIT, 0]
    else:
        updated_ind, opt_vars = insert_weights(deepcopy(GPind), mul_fun, weight)
        str_ind = str(deepcopy(updated_ind))
        count = 1
        for i in range(len(opt_vars)):
            str_ind = str_ind.replace('(1', '(w{}'.format(count), 1)
            count += 1
        explicit_eqs = []
        explicit_eq_g = []
        for eqs in obj.eq_f:
            for i in range(len(obj.cont_dict)):
                eqs = eqs.replace(obj.cont_dict['u{}'.format(i + 1)], '(' + str_ind + ')')
            for word, replacement in obj.GPinput_dict.items():
                eqs = eqs.replace(word, replacement)
            for i in range(len(obj.states_dict) - 1, -1, -1):
                eqs = eqs.replace(obj.states_dict['x{}'.format(i + 1)], 'x{}'.format(i + 1))
            explicit_eqs.append(eqs)
        for eqs in obj.eq_g:
            for i in range(len(obj.cont_dict)):
                eqs = eqs.replace(obj.cont_dict['u{}'.format(i + 1)], '(' + str_ind + ')')
            for word, replacement in obj.GPinput_dict.items():
                eqs = eqs.replace(word, replacement)
            for i in range(len(obj.states_dict) - 1, -1, -1):
                eqs = eqs.replace(obj.states_dict['x{}'.format(i + 1)], 'x{}'.format(i + 1))
            explicit_eq_g.append(eqs)
        A_list, B_list, PSI_list, PHI_list = define_matrices(explicit_eqs, explicit_eq_g, len(opt_vars), obj.n_states, symbols)

        opt_params, failure, fit = Adam(obj.alfa, obj.b1, obj.b2, obj.eps, opt_vars, opt_steps, obj, updated_ind, pset,
                                        A_list, B_list, PSI_list, PHI_list, verbose, dynamics)
        if failure is True:
            del pset.terminals[pset.ret][-1]
            return [1e6, 0]
        else:
            del pset.terminals[pset.ret][-1]
            return [fit, 0]


def insert_weights(ind, mul_fun, weight):
    """
    Function used to insert learnable parameters into GP individuals as proposed in [1].

    Attributes:
        ind: list
            individual
        mul_fun: primitive
            multiplication function
        weight: terminal
            a terminal node representing the learnable parameters

    Return:
        ind: list
            updated individual
        w: array
            numerical values of the learnable parameters
    """

    j = 0
    l = len(ind)
    stop = False
    w = []
    while not stop:
        ind.insert(j, mul_fun)
        ind.insert(j + 1, weight)
        w.append(1)
        j = j + 3
        l += 2
        if j == l:
            stop = True
    return ind, np.array((w))


def adjoint_grad(obj, GPind, pset, A_list, B_list, PSI_list, PHI_list, opt_vars, dynamics):
    """
    This function is used to evaluate the gradient and the fitness using the adjoint state method

    Attributes:
        obj: class
            a class containing the plant's paramters
        GPind: list
            the GP individual considered
        pset: class
            primitive set
        A_list: list
            the list of functions used in the A matrix evaluation
        B_list: list
            the list of functions used in the B matrix evaluation
        PSI_list: list
            the list of functions used in the PSI matrix evaluation
        PHI_list: list
            the list of functions used in the PHI matrix evaluation
        opt_vars: array
            the array of optimization variables
        dynamics: function
            the function describing the plant's dynamics
    Return:
        grad: array
            gradient
        success: bool
            states the outcome of the forward and backward propagations
        FIT: float
            fitness of the individual evaluate with the adjoint method
    """
    count = 0
    for i in range(len(GPind)):
        if type(GPind[i]) == gp.Terminal and GPind[i].name == 'w':
            GPind[i] = deepcopy(GPind[i])
            GPind[i].value = opt_vars[count]
            count += 1

    x, u, failure_for = propagate_forward(obj, GPind, pset, dynamics)
    if failure_for is False:
        adjoint_init = ETA_function(x[-1,:], obj.Qt, obj.xf)

        adjoint, failure_back = propagate_backward(obj, A_list, PHI_list, opt_vars, x, adjoint_init, obj.val_symbols)
        if failure_back is False:
            int_fun = np.zeros((obj.Npoints, len(opt_vars)))
            g = np.zeros(obj.Npoints)
            for i in range(obj.Npoints):
                B = fill_sensitivity_matrix(B_list, x[i,:], opt_vars, obj.val_symbols)
                PSI = fill_sensitivity_matrix(PSI_list, x[i,:], opt_vars, obj.val_symbols)
                int_fun[i, :] = B.T @ adjoint[i,:] + PSI.T
                g[i] = 0.5 * ((x[i, :]-obj.xf).T @ obj.Qz @ (x[i, :]-obj.xf) + np.array(([[u[i]]])).T @ obj.Qu @ np.array(([[u[i]]])))
            grad = simps(int_fun, obj.t_points, axis=0)
            int_g = simps(g, obj.t_points)
            h = 0.5 * ((x[-1, :]-obj.xf).T @ obj.Qt @ (x[-1, :]-obj.xf))
            FIT = int_g + h
            if np.isnan(grad).any() or np.isinf(grad).any():
                return 0, True, 1e6
            else:
                return grad, False, FIT
        else:
            return 0, True, 1e6
    else:
        return 0, True, 1e6

def adjoint_grad_noGP(obj, c_law, A_list, B_list, PSI_list, PHI_list, opt_vars, dynamics):
    """
        This function is used to evaluate the gradient and the fitness using the adjoint state method, when a control
        law obtained without GP is used

        Attributes:
            obj: class
                a class containing the plant's paramters
            c_law: function
                the control law
            pset: class
                primitive set
            A_list: list
                the list of functions used in the A matrix evaluation
            B_list: list
                the list of functions used in the B matrix evaluation
            PSI_list: list
                the list of functions used in the PSI matrix evaluation
            PHI_list: list
                the list of functions used in the PHI matrix evaluation
            opt_vars: array
                the array of optimization variables
            dynamics: function
                the function describing the plant's dynamics
        Return:
            grad: array
                gradient
            success: bool
                states the outcome of the forward and backward propagations
            FIT: float
                fitness of the individual evaluate with the adjoint method
        """
    x, u, failure_for = propagate_forward_noGP(obj, c_law, dynamics, opt_vars)
    if failure_for is False:
        adjoint_init = ETA_function(x[-1,:], obj.Qt, obj.xf)

        adjoint, failure_back = propagate_backward(obj, A_list, PHI_list, opt_vars, x, adjoint_init, obj.val_symbols)
        if failure_back is False:
            int_fun = np.zeros((obj.Npoints, len(opt_vars)))
            g = np.zeros(obj.Npoints)
            for i in range(obj.Npoints):
                B = fill_sensitivity_matrix(B_list, x[i,:], opt_vars, obj.val_symbols)
                PSI = fill_sensitivity_matrix(PSI_list, x[i,:], opt_vars, obj.val_symbols)
                int_fun[i, :] = B.T @ adjoint[i,:] + PSI.T
                g[i] = 0.5 * ((x[i, :]-obj.xf).T @ obj.Qz @ (x[i, :]-obj.xf) + np.array(([[u[i]]])).T @ obj.Qu @ np.array(([[u[i]]])))
            grad = simps(int_fun, obj.t_points, axis=0)
            int_g = simps(g, obj.t_points)
            h = 0.5 * ((x[-1, :]-obj.xf).T @ obj.Qt @ (x[-1, :]-obj.xf))
            FIT = int_g + h
            if np.isnan(grad).any() or np.isinf(grad).any():
                return 0, True, 1e6
            else:
                return grad, False, FIT
        else:
            return 0, True, 1e6
    else:
        return 0, True, 1e6


def Adam(alfa, b1, b2, eps, opt_vars, opt_steps, obj, GPind, pset, A_list, B_list, PSI_list, PHI_list, verbose, dynamics):
    """
    Function used to perform the optimization with the Adam algorithm

    Attributes:
        alfa: float
            learning rate
        b1: float
            adam parameter
        b2: float
            adam parameter
        eps: float
            adam parameter
        opt_vars: array
            optimization variables
        opt_steps: integer
            optimization steps to perform
        obj: class
            plant's model
        GPind: list
            GP individual
        pset: class
            ptimitive set
        A_list: list
            contains the functions to evaluate the A matrix
        B_list: list
            contains the functions to evaluate the B matrix
        PSI_list: list
            contains the functions to evaluate the PSI matrix
        PHI_list: list
            contains the functions to evaluate the PHI matrix
        verbose: bool
            to print or not the optimization outcome
        dynamics: function
            contains the plant's dynamics

    Return:
        best_vars: array
            optimized variables
        failure: bool
            tells if the adjoint propagations failed or not
        best_fit: float
            fitness corresponding to the optimized variables
    """
    m = 0
    v = 0
    t = 0
    best_fit = 1e6
    best_vars = copy(opt_vars)
    failure = False
    for i in range(opt_steps):
        t += 1
        grad, failure, fit = adjoint_grad(obj, GPind, pset, A_list, B_list, PSI_list, PHI_list, opt_vars, dynamics)
        if failure == False:
            if fit < best_fit:
                best_fit = copy(fit)
                best_vars = copy(opt_vars)
            m = b1 * m + (1-b1) * grad
            v = b2 * v + (1-b2) * grad**2
            m_bar = m/(1-b1**t)
            v_bar = v/(1-b2**t)
            opt_vars = copy(opt_vars - alfa*m_bar/(np.sqrt(v_bar)+eps))

            if verbose is True:
                print('Step {}, Fitness {}'.format(t, fit))
        else:
            return best_vars, failure, best_fit
    return best_vars, failure, best_fit

def Adam_noGP(alfa, b1, b2, eps, opt_vars, opt_steps, obj, c_law, A_list, B_list, PSI_list, PHI_list, verbose, dynamics):
    """
        Function used to perform the optimization with the Adam algorithm without the GP individual

        Attributes:
            alfa: float
                learning rate
            b1: float
                adam parameter
            b2: float
                adam parameter
            eps: float
                adam parameter
            opt_vars: array
                optimization variables
            opt_steps: integer
                optimization steps to perform
            obj: class
                plant's model
            c_law: function
                control law individual
            pset: class
                ptimitive set
            A_list: list
                contains the functions to evaluate the A matrix
            B_list: list
                contains the functions to evaluate the B matrix
            PSI_list: list
                contains the functions to evaluate the PSI matrix
            PHI_list: list
                contains the functions to evaluate the PHI matrix
            verbose: bool
                to print or not the optimization outcome
            dynamics: function
                contains the plant's dynamics

        Return:
            best_vars: array
                optimized variables
            failure: bool
                tells if the adjoint propagations failed or not
            best_fit: float
                fitness corresponding to the optimized variables
        """
    m = 0
    v = 0
    t = 0
    best_fit = 1e6
    best_vars = copy(opt_vars)
    failure = False
    for i in range(opt_steps):
        t += 1
        grad, failure, fit = adjoint_grad_noGP(obj, c_law, A_list, B_list, PSI_list, PHI_list, opt_vars, dynamics)
        if verbose is True:
            print('Step {}, Fitness {}, Opt vars {}'.format(t, fit, opt_vars))
        if failure == False:
            if fit < best_fit:
                best_fit = copy(fit)
                best_vars = copy(opt_vars)
            m = b1 * m + (1-b1) * grad
            v = b2 * v + (1-b2) * grad**2
            m_bar = m/(1-b1**t)
            v_bar = v/(1-b2**t)
            opt_vars = copy(opt_vars - alfa*m_bar/(np.sqrt(v_bar)+eps))
        else:
            return best_vars, failure, best_fit
    return best_vars, failure, best_fit

def propagate_forward(obj, GPind, pset, dynamics):
    """
    Function used to propagate the dyanmics forward

    Attributes:
        obj: class
            plant model
        GPind: list
            GP individual
        pset: class
            primitive set
        dynamics: function
            plant dynamics

    Return:
        x: array
            propagated state variables
        u: array
            propagated control variables
        failure: bool
            tells if the propagation failed or not
    """
    x = np.zeros((obj.Npoints, obj.n_states))
    u = np.zeros(obj.Npoints)

    x[0, :] = obj.x0
    ufuns = gp.compile(GPind, pset=pset)
    vv = x[0,:]-obj.xf
    u[0] = ufuns(*vv)
    failure = False
    for i in range(obj.Npoints - 1):
        sol_forward = int_schemes.RK4(obj.t_points[i], obj.t_points[i + 1], dynamics, 2, x[i, :],
                                      args=(obj, u[i]))
        if np.isnan(sol_forward[-1, :]).any() or np.isinf(sol_forward[-1, :]).any():
            failure = True
            break
        else:
            x[i + 1, :] = sol_forward[-1, :]
            vv = x[i+1, :] - obj.xf
            u[i + 1] = ufuns(*vv)
    return x, u, failure

def propagate_forward_noGP(obj, c_law, dynamics, opt_vars):
    """
        Function used to propagate the dynamics forward without the GP control law

        Attributes:
            obj: class
                plant model
            c_law: function
                control law
            dynamics: function
                plant dynamics
            opt_vars: array
                optimization variables

        Return:
            x: array
                propagated state variables
            u: array
                propagated control variables
            failure: bool
                tells if the propagation failed or not
        """
    x = np.zeros((obj.Npoints, obj.n_states))
    u = np.zeros(obj.Npoints)

    x[0, :] = obj.x0
    vv = x[0,:]-obj.xf
    try:
        u[0] = c_law(*vv, *opt_vars)
    except TypeError:
        return [],[],True
    failure = False
    for i in range(obj.Npoints - 1):
        sol_forward = int_schemes.RK4(obj.t_points[i], obj.t_points[i + 1], dynamics, 2, x[i, :],
                                      args=(obj, u[i]))
        if np.isnan(sol_forward[-1, :]).any() or np.isinf(sol_forward[-1, :]).any():
            failure = True
            break
        else:
            x[i + 1, :] = sol_forward[-1, :]
            vv = x[i+1, :] - obj.xf
            u[i + 1] = c_law(*vv, *opt_vars)
    return x, u, failure


def backward_dynamics(t, adjoint, A_list, PHI_list, x, opt_vars, val_symbols):
    """
    Function describing the backward dynamics using the adjoint state method

    Attributes:
        t: float
            time
        adjoint: array
            adjoint variables
        A_list: list
            contains the functions used to evaluate the A matrix
        PHI_list: list
            contains the functions used to evaluate the PHI matrix
        x: array
            state variables
        opt_vars: array
            optimization variables
        val_symbols: array
            values of the symbols used to evaluate the sensitivity matrices
    Return:
        adjoint: array
            updated adjoint variables
    """
    adjoint = np.array([adjoint])
    A = fill_sensitivity_matrix(A_list, x, opt_vars, val_symbols)
    PHI = fill_sensitivity_matrix(PHI_list, x, opt_vars, val_symbols)
    if PHI.ndim == 1:
        PHI = np.array([PHI])
    dadjoint_dt = -A.T @ adjoint.T - PHI.T
    return dadjoint_dt.T[0]

def propagate_backward(obj, A_list, PHI_list, opt_vars, x, adjoint_init, val_symbols):
    """
    Function used to propagate the dynamics backwards

    Attributes:
        obj: class
            plant model
        A_list: list
            contains the functions used to evaluate the A matrix.
        PHI_list: list
            contains the functions used to evaluate the PHI matrix
        opt_vars: array
            optimization variables
        x: array
            state variables
        adjoint_init: array
            initial values of adjoint variables
        val_symbols: array
            numerical values of the symbols used in the sensitivity matrices

    Return:
        adjoint: array
            propagated adjoint variables
        failure: bool
            tells if the propagation failed or not

    """
    adjoint = np.zeros((obj.Npoints, obj.n_states))
    adjoint[-1, :] = adjoint_init
    failure = False

    for i in range(obj.Npoints - 1, 0, -1):
        sol_backward = int_schemes.RK4(obj.t_points[i], obj.t_points[i-1], backward_dynamics, 2, adjoint[i,:],
                                      args=(A_list, PHI_list, x[i,:], opt_vars, val_symbols))
        if np.isnan(sol_backward[-1,:]).any() or np.isinf(sol_backward[-1,:]).any():
            failure = True
            break
        else:
            adjoint[i-1, :] = sol_backward[-1, :]

    return adjoint, failure



def final_optimization(GPind, obj, pset, opt_steps, mul_fun, verbose, dynamics, symbols):
    """
    Function used to perform the optimization on the best individual with Adam

    Attributes:
        GPind: list
            Gp individual
        obj: class
            plant model
        pset: class
            primitive set
        opt_steps: integer
            number of optimization steps to perform
        mul_fun: primitive
            multiplication function
        verbose: bool
            whether to print or not the optimization outcome
        dynamics: function
            contains the plant dynamics
        symbols: list
            contains the symbols used to evaluate the sensitivty matrices
    Return:
        fit: float
            fitness value
        opt_params: array
            optimized variables
    """
    pset.addTerminal('w')
    pset.terminals[pset.ret][-1].value = 1  # set initial value of weights to 1
    weight = deepcopy(pset.terminals[pset.ret][-1])  # store weight terminal

    updated_ind, opt_vars = insert_weights(deepcopy(GPind), mul_fun, weight)
    str_ind = str(deepcopy(updated_ind))
    count = 1
    for i in range(len(opt_vars)):
        str_ind = str_ind.replace('(1', '(w{}'.format(count), 1)
        count += 1
    explicit_eqs = []
    explicit_eq_g = []
    for eqs in obj.eq_f:
        for i in range(len(obj.cont_dict)):
            eqs = eqs.replace(obj.cont_dict['u{}'.format(i + 1)], '(' + str_ind + ')')
        for word, replacement in obj.GPinput_dict.items():
            eqs = eqs.replace(word, replacement)
        for i in range(len(obj.states_dict) - 1, -1, -1):
            eqs = eqs.replace(obj.states_dict['x{}'.format(i + 1)], 'x{}'.format(i + 1))
        explicit_eqs.append(eqs)
    for eqs in obj.eq_g:
        for i in range(len(obj.cont_dict)):
            eqs = eqs.replace(obj.cont_dict['u{}'.format(i + 1)], '(' + str_ind + ')')
        for word, replacement in obj.GPinput_dict.items():
            eqs = eqs.replace(word, replacement)
        for i in range(len(obj.states_dict) - 1, -1, -1):
            eqs = eqs.replace(obj.states_dict['x{}'.format(i + 1)], 'x{}'.format(i + 1))
        explicit_eq_g.append(eqs)
    A_list, B_list, PSI_list, PHI_list = define_matrices(explicit_eqs, explicit_eq_g, len(opt_vars), obj.n_states, symbols)

    opt_params, failure, fit = Adam(obj.alfa, obj.b1, obj.b2, obj.eps, opt_vars, opt_steps, obj, updated_ind, pset,
                                    A_list, B_list, PSI_list, PHI_list, verbose, dynamics)

    if failure is True:
        del pset.terminals[pset.ret][-1]
        return 1e6, opt_params
    else:
        del pset.terminals[pset.ret][-1]
        return fit, opt_params

def final_optimization_reducedLaw(c_law, c_law_string, init_conds, obj, pset, opt_steps, verbose, dynamics, symbols):
    """
    Function used to perform the optimization on the best individual with Adam, on the control law obtained by
    simplifying the GP individual

    Attributes:
        c_law: function
            control law
        c_law_string: string
            string representing the control law
        init_conds: array
            array of initial conditions
        obj: class
            plant model
        pset: class
            primitive set
        opt_steps: integer
            number of optimization steps to perform
        verbose: bool
            whether to print or not the optimization outcome
        dynamics: function
            contains the plant dynamics
        symbols: list
            contains the symbols used to evaluate the sensitivty matrices
    Return:
        fit: float
            fitness value
        opt_params: array
            optimized variables
    """
    explicit_eqs = []
    explicit_eq_g = []
    for eqs in obj.eq_f:
        for i in range(len(obj.cont_dict)):
            eqs = eqs.replace(obj.cont_dict['u{}'.format(i + 1)], '(' + c_law_string + ')')
        for word, replacement in obj.GPinput_dict.items():
            eqs = eqs.replace(word, replacement)
        for i in range(len(obj.states_dict) - 1, -1, -1):
            eqs = eqs.replace(obj.states_dict['x{}'.format(i + 1)], 'x{}'.format(i + 1))
        explicit_eqs.append(eqs)
    for eqs in obj.eq_g:
        for i in range(len(obj.cont_dict)):
            eqs = eqs.replace(obj.cont_dict['u{}'.format(i + 1)], '(' + c_law_string + ')')
        for word, replacement in obj.GPinput_dict.items():
            eqs = eqs.replace(word, replacement)
        for i in range(len(obj.states_dict) - 1, -1, -1):
            eqs = eqs.replace(obj.states_dict['x{}'.format(i + 1)], 'x{}'.format(i + 1))
        explicit_eq_g.append(eqs)
    A_list, B_list, PSI_list, PHI_list = define_matrices(explicit_eqs, explicit_eq_g, len(init_conds), obj.n_states, symbols)

    opt_params, failure, fit = Adam_noGP(obj.alfa, obj.b1, obj.b2, obj.eps, init_conds, opt_steps, obj, c_law, A_list,
                                         B_list, PSI_list, PHI_list, verbose, dynamics)

    if failure is True:
        del pset.terminals[pset.ret][-1]
        return 1e6, opt_params
    else:
        del pset.terminals[pset.ret][-1]
        return fit, opt_params


def parametrize_simplified_eq(simplified_eq, obj):
    """
    Function used to parametrize the simplified equation

    Attributes:
        simplified_eq: str
            simplified equation obtained from the GP individual
        obj: class
            plant model

    Return:
        s_fun: function
            function of the simplified equation
        s: string
            string of the simplified equation
        init_conds: array
            array of initial conditions
    """
    s = str(simplified_eq)
    p = r"([+-])?\s*(?:(\d+)\s*\*\s*)?([a-z]\w*)"
    coeffs = re.findall(p, str(simplified_eq))
    dict_opt_vars = {v: int(s + (c or '1')) for (s, c, v) in coeffs}
    index_map = {v: i for i, v in enumerate(obj.vars_order)}
    dict_opt_vars = dict(sorted(dict_opt_vars.items(), key=lambda pair: index_map[pair[0]]))
    s = s.strip()
    count = 1
    for key in dict_opt_vars:
        if dict_opt_vars[key] == 1 or dict_opt_vars[key] == -1:
            s = s.replace(key, 'w{}*'.format(count) + key)
        else:
            s = s.replace(str(abs(dict_opt_vars[key])) + '*' + key, '+w{}*'.format(count) + key)
        count += 1
    s = s.replace('-', '')
    symbols_vars = sympy.symbols([key for key in dict_opt_vars])
    symbols_weights = sympy.symbols(['w{}'.format(i) for i in range(1, count)])
    init_conds = [dict_opt_vars[key] for key in dict_opt_vars]
    s_fun = sympy.lambdify([*symbols_vars, *symbols_weights], s)
    return s_fun, s, np.array((init_conds))


def fitness_fun_finite_diff(opt_vars, obj, GPind, dynamics, pset):
    """
    Function used to evaluate the fitness with the scipy optimizers

    Attributes:
        opt_vars: array
            optimization variables
        obj: class
            plant model
        GPind: list
            GP individual
        dynamics: function
            describe the plant's dynamics
        pset: class
            primitve set

    Return:
        FIT: float
            fitness value
    """
    count = 0
    for i in range(len(GPind)):
        if type(GPind[i]) == gp.Terminal and GPind[i].name == 'w':
            GPind[i] = deepcopy(GPind[i])
            GPind[i].value = opt_vars[count]
            count += 1
    x, u, failure_for = propagate_forward(obj, GPind, pset, dynamics)
    if failure_for is False:
        g = np.zeros(obj.Npoints)
        for i in range(obj.Npoints):
            g[i] = 0.5 * ((x[i, :] - obj.xf).T @ obj.Qz @ (x[i, :] - obj.xf) + np.array(
                ([[u[i]]])).T @ obj.Qu @ np.array(([[u[i]]])))
        int_g = simps(g, obj.t_points)
        h = 0.5 * ((x[-1, :] - obj.xf).T @ obj.Qt @ (x[-1, :] - obj.xf))
        FIT = int_g + h
        return FIT
    else:
        return 1e6

def define_GP_model_pendulum(limit_height, limit_size, obj):
    """
    Function used to create the GP model in the pendulum test case

    Attributes:
        limit_height: integer
            max height of the GP tree
        limit_size: integer
            max sie of the GP tree
        obj: class
            plant model

    Return:
        pset: class
            primitive set
        creator: class
            GP building blocks
        toolbox: class
            GP building blocks
    """
    ####################################    P R I M I T I V E  -  S E T     ################################################

    pset = gp.PrimitiveSet("Main", obj.n_states)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.renameArguments(ARG0='eX')
    pset.renameArguments(ARG1='eV')
    pset.renameArguments(ARG2='eTheta')
    pset.renameArguments(ARG3='eOmega')

    ################################################## TOOLBOX #############################################################

    creator.create("Fitness", base.Fitness,
                   weights=(-1.0, -1.0))  # , -0.1, -0.08, -1.0))    # MINIMIZATION OF THE FITNESS FUNCTION
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate_individualIGP_adjoint)
    toolbox.register("select", gpfuns.InclusiveTournament, selected_individuals=1, fitness_size=2, parsimony_size=1.6,
                     creator=creator)
    toolbox.register("mate", gp.cxOnePoint)  ### NEW ##
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)
    toolbox.register("mutate", gpfuns.xmut, expr=toolbox.expr_mut, unipb=0.6, shrpb=0.1, inspb=0.3, pset=pset,
                     creator=creator)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))
    return pset, creator, toolbox

def define_GP_model_oscillator(limit_height, limit_size, obj):
    """
        Function used to create the GP model in the oscillator test case

        Attributes:
            limit_height: integer
                max height of the GP tree
            limit_size: integer
                max sie of the GP tree
            obj: class
                plant model

        Return:
            pset: class
                primitive set
            creator: class
                GP building blocks
            toolbox: class
                GP building blocks
        """
    ####################################    P R I M I T I V E  -  S E T     ################################################

    pset = gp.PrimitiveSet("Main", obj.n_states)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.renameArguments(ARG0='eX')
    pset.renameArguments(ARG1='eV')

    ################################################## TOOLBOX #############################################################

    creator.create("Fitness", base.Fitness,
                   weights=(-1.0, -1.0))  # , -0.1, -0.08, -1.0))    # MINIMIZATION OF THE FITNESS FUNCTION
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate_individualIGP_adjoint)
    toolbox.register("select", gpfuns.InclusiveTournament, selected_individuals=1, fitness_size=2, parsimony_size=1.6,
                     creator=creator)
    toolbox.register("mate", gp.cxOnePoint)  ### NEW ##
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)
    toolbox.register("mutate", gpfuns.xmut, expr=toolbox.expr_mut, unipb=0.6, shrpb=0.1, inspb=0.3, pset=pset,
                     creator=creator)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))
    return pset, creator, toolbox