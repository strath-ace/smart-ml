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
Main script to perform the evolutionary process with the OPGD-IGP on the pendulum test case
"""

import os
dirname = os.path.dirname(__file__)
import numpy as np
import functionsIGPadjoint as funs
from time import time
import multiprocessing
from deap import gp
from scipy.optimize import minimize
import operator
import _pickle as cPickle
from GP.GPBased_ControlSchemes.OPGD_IGP.test_cases.inverted_pendulum.Plant import Pendulum
from copy import deepcopy, copy
import sympy
import warnings
from GP.GPBased_ControlSchemes.OPGD_IGP.test_cases.inverted_pendulum.dynamics import dynamics
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    obj = Pendulum()

    limit_height = 10  # Max height (complexity) of the gp law
    limit_size = 30  # Max size (complexity) of the gp law

    size_pop = 300
    size_gen = 300  # Gen size

    fit_tol = 1e-10

    n_sims = 30

    Mu = int(size_pop)
    Lambda = int(size_pop * 1.2)

    symbols = [sympy.symbols(sym) for sym in obj.str_symbols]

    nbCPU = multiprocessing.cpu_count()

    intra_evo_opt_steps = 5
    obj.alfa = 0.01

    for sim in range(n_sims):
        pset, creator, toolbox = funs.define_GP_model_pendulum(limit_height, limit_size, obj)
        mul_fun = deepcopy(pset.primitives[pset.ret][2])  # store mul function primitive

        save_path = os.path.abspath(os.path.join(dirname, 'Results'))
        save_path = save_path + '/Pendulum/Adjoint/Sim{}/'.format(sim)
        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass

        start = time()
        pop, log, hof, pop_statistics, ind_lengths = funs.main(nbCPU, toolbox, size_pop, size_gen, creator, Mu, Lambda,
                                                               pset, obj, intra_evo_opt_steps, mul_fun, dynamics,
                                                               symbols)
        output = open(save_path+'Best_ind', "wb")
        cPickle.dump(hof[-1], output, -1)
        output.close()

        best_ind = copy(hof[-1])

        fitness_evol = []
        for i in range(len(log)):
            fitness_evol.append(log.chapters['fitness'][i]['min'])
        np.save(save_path+'fitness_evol_IGP_adj.npy', fitness_evol)

        pset.addTerminal('w')
        pset.terminals[pset.ret][-1].value = 1  # set initial value of weights to 1
        weight = deepcopy(pset.terminals[pset.ret][-1])  # store weight terminal
        updated_best_ind, opt_vars_init = funs.insert_weights(deepcopy(best_ind), mul_fun, weight)

        res = minimize(funs.fitness_fun_finite_diff, opt_vars_init, method='BFGS',
                       args=(obj, updated_best_ind, dynamics, pset),
                       options={'disp': True})
        opt_vars = res.x
        fitness = res.fun
        count = 0
        for i in range(len(updated_best_ind)):
            if type(updated_best_ind[i]) == gp.Terminal and updated_best_ind[i].name == 'w':
                updated_best_ind[i] = deepcopy(updated_best_ind[i])
                updated_best_ind[i].value = opt_vars[count]
                count += 1

        x_IGP, u_IGP, failure = funs.propagate_forward(obj, updated_best_ind, pset, dynamics)
        eX, eV, eTheta, eOmega = sympy.symbols('eX eV eTheta eOmega')
        sympy.init_printing(use_unicode=True)
        simplified_eq = sympy.sympify(str(updated_best_ind),
                                      locals={'add': operator.add, 'sub': operator.sub, 'mul': operator.mul,
                                              'cos': sympy.cos, 'sin': sympy.sin})

        print('Best individual: ', str(simplified_eq))
        end = time()
        t_offdesign = end - start
        np.save(save_path+'best_fitness_IGP_adj.npy', fitness)

        with open(save_path+'best_ind_structure_IGP_adj.txt', 'w') as f:
            f.write(str(simplified_eq))
        with open(save_path+'best_ind_original_structure_IGP_adj.txt', 'w') as f:
            f.write(str(updated_best_ind))

        np.save(save_path+'computational_time_IGP_adj.npy', t_offdesign)
        np.save(save_path+'best_fitness_IGP_adj.npy', fitness)
        np.save(save_path+'x_IGP_adj.npy', x_IGP)
        np.save(save_path+'u_IGP_adj.npy', u_IGP)

        del pset.terminals[pset.ret][-1]