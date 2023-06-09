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

"""This script is used to find the best individual in each of the performed simulations. The paths to save and load
must be modified by the user."""


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import GP.GP_Algorithms.IGP.IGP_Functions as gpfuns
import GP.GP_Algorithms.GP_PrimitiveSet as gpprim
import GP.GP_Algorithms.IGP.Recombination_operators as rop
import numpy as np
import operator
import GP.GPBased_ControlSchemes.GANNIC.functions_GANNIC as funs
from deap import gp, base, creator, tools
import GP.GPBased_ControlSchemes.GANNIC.FESTIP_reentry.Plant_models as mods
import warnings
import random
import GP.GPBased_ControlSchemes.GANNIC.utils_NN as utils_NN
from copy import copy
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    ###################### System parameters ###############################
    obj = mods.Spaceplane()
    obj.load_init_conds("init_conds.mat")
    ############################ GP PARAMETERS ######################################
    limit_height = 40  # Max height (complexity) of the controller law
    limit_size = 40  # Max size (complexity) of the controller law
    nEph = 5  # number of ephemeral constants

    flag_save = False  # set to true to save data
    flag_plot = False

    folder = 'DataMining'

    if flag_save:
        try:
            os.makedirs("Results/" + folder)
        except FileExistsError:
            pass

    savedata_file = "Results/" + folder + '/'


    ####################################    P R I M I T I V E  -  S E T     ############################################

    obj.n_inputGP = 8
    obj.n_inputNN = 3
    obj.n_inputTOT = 11

    NNmodel, n_weights = utils_NN.create_NN_model(obj.n_inputNN, obj.n_hidden_layers, obj.nodes_per_hidden_layer,
                                                  obj.NNoutput, obj.activation)
    pset = gp.PrimitiveSet("Main", obj.n_inputGP)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(gpprim.TriAdd, 3)
    pset.addPrimitive(np.tanh, 1)
    pset.addPrimitive(gpprim.ModSqrt, 1)
    pset.addPrimitive(gpprim.ModLog, 1)
    pset.addPrimitive(gpprim.ModExp, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
    for i in range(nEph):
        pset.addEphemeralConstant("rand{}".format(i), lambda: round(random.uniform(-10, 10), 4))

    pset.renameArguments(ARG0='v')
    pset.renameArguments(ARG1='chi')
    pset.renameArguments(ARG2='gamma')
    pset.renameArguments(ARG3='teta')
    pset.renameArguments(ARG4='lam')
    pset.renameArguments(ARG5='h')
    pset.renameArguments(ARG6='Dens')
    pset.renameArguments(ARG7='W')

    ################################################## TOOLBOX #############################################################

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness)
    creator.create("SubIndividual", gp.PrimitiveTree)

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genFull, pset=pset, type_=pset.ret, min_=1, max_=4)
    toolbox.register("leg", tools.initIterate, creator.SubIndividual, toolbox.expr)
    toolbox.register("legs", tools.initCycle, list, [toolbox.leg], n=n_weights)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.legs)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", funs.evaluateLearningNodes)
    toolbox.register("select", gpfuns.InclusiveTournamentV2, selected_individuals=1, fitness_size=2, parsimony_size=1.6,
                     creator=creator)
    toolbox.register("mate", rop.xmate)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)
    toolbox.register("mutate", rop.xmut, expr=toolbox.expr_mut, unipb=0.55, shrpb=0.05, inspb=0.25, pset=pset,
                     creator=creator)
    toolbox.decorate("mate", gpfuns.staticLimitMod(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mutate", gpfuns.staticLimitMod(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mate", gpfuns.staticLimitMod(key=len, max_value=limit_size))
    toolbox.decorate("mutate", gpfuns.staticLimitMod(key=len, max_value=limit_size))


    ########################## Uncertainty parameters ##############################
    uncertain_atmo = True
    uncertain_aero = False
    uncertain = True

    test_unc_profiles = np.load('Test_uncertSurfaces.npy')

    v_points = np.linspace(obj.vmin, obj.vmax, test_unc_profiles[0].shape[1])
    h_points = np.linspace(obj.hmin, obj.hmax, test_unc_profiles[0].shape[1])

    for sim in range(0, 21):
        min_fit_old = 1000
        gens_to_check = []
        for j in range(301):
            min_fit = np.load('Results/Simulation{}_stat/Min_fitness_gen_{}.npy'.format(sim, j), allow_pickle=True)[-1][0]
            if min_fit < min_fit_old:
                min_fit_old = copy(min_fit)
                gens_to_check.append(j)
                with open('Results/simulation_{}_analysis_laptop.txt'.format(sim), 'a') as f:
                    f.write('Sim {} Gen {}, fit {}'.format(sim, j, min_fit))
                    f.write('\n')
                print('Sim {} Gen {}, fit {}'.format(sim, j, min_fit))
        analysis_res = np.zeros((len(gens_to_check), 8))
        count = 0
        for gen in gens_to_check:
            analysis_res[count, 0] = gen
            hof = np.load('Results/Simulation{}_stat/Hof_gen_{}'.format(sim, gen), allow_pickle=True)
            NNmodel, n_weights = utils_NN.create_NN_model(obj.n_inputNN, obj.n_hidden_layers, obj.nodes_per_hidden_layer,
                                                          obj.NNoutput, obj.activation)
            obj.NNinitConds = []
            for i in range(len(NNmodel.weights)):
                obj.NNinitConds.extend(np.ravel(NNmodel.get_weights()[i]))
            NNinitName = 'NNinitconds_test.npy'
            np.save(NNinitName, obj.NNinitConds)
            best_ind = hof[-1]
            with open('Results/simulation_{}_analysis_laptop.txt'.format(sim), 'a') as f:
                f.write(str(best_ind.fitness.values))
                f.write('\n')
            print(best_ind.fitness.values)
            analysis_res[count, 1] = best_ind.fitness.values[0]
            ubd = 0.2
            ubc = 0.2
            for i in range(5):
                print('Uncertainty ', ubd)
                succs = funs.plot_best_ind(obj, v_points, h_points, uncertain, uncertain_atmo, uncertain_aero, pset, savedata_file,
                                best_ind, test_unc_profiles, NNmodel, n_weights, NNinitName, flag_plot, ubc, ubd)
                analysis_res[count, 2+i] = succs/100
                with open('Results/simulation_{}_analysis_laptop.txt'.format(sim), 'a') as f:
                    f.write('Uncertainty {}, Successes {}/{}'.format(ubd, succs, obj.n_test))
                    f.write('\n')
                analysis_res[count, -1] = np.mean(analysis_res[count, 2:7])
                ubc += 0.1
                ubd += 0.1
            count += 1
            np.save('analysis_res_sim{}_laptop.npy'.format(sim), analysis_res)
