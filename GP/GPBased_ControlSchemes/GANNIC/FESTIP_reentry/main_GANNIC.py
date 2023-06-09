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

"""Main function to run the GANNIC scheme"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import GP.GP_Algorithms.IGP.IGP_Functions as gpfuns
import GP.GP_Algorithms.GP_PrimitiveSet as gpprim
import GP.GP_Algorithms.IGP.Recombination_operators as rop
import numpy as np
import operator
from copy import copy
import GP.GPBased_ControlSchemes.GANNIC.functions_GANNIC as funs
import time
from deap import gp, base, creator, tools
import GP.GPBased_ControlSchemes.GANNIC.FESTIP_reentry.Plant_models as mods
import warnings
import random
import pandas as pd
import GP.GPBased_ControlSchemes.GANNIC.utils_NN as utils_NN
import multiprocess

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    obj = mods.Spaceplane() # load plant parameters

    obj.load_init_conds("init_conds.mat")


    ############################ GP PARAMETERS ######################################
    limit_height = 40  # Max height (complexity) of the controller law
    limit_size = 40  # Max size (complexity) of the controller law
    train_uncScen = obj.n_train  # how many individuals to use for training
    save_gen = None  # every how many generations saves the data
    size_gen = 300 # Gen size
    start_gen = 0 # change to start simulation from gen n
    size_pop_tot = 500 # population size
    nEph = 5  # number of ephemeral constants
    hof_size = int(size_pop_tot * 0.05) # size of hall of fame
    Mu = int(size_pop_tot)
    Lambda = int(size_pop_tot * 1.2)
    mutpb_start = 0.7  # mutation probability
    cxpb_start = 0.2 # crossover probability
    mutpb = copy(mutpb_start)
    cxpb = copy(cxpb_start)
    nbCPU = multiprocess.cpu_count()

    load_population = False  # set to True if you want to start GP from an previoulsy obtained population.
    # The size of the old population must be the same of size_pop_tot
    if load_population is False:
        pop_path = None
    else:
        pop_path = 'Results/LearningNodes/Population_gen_{}'.format(start_gen)
    pop_random_seed = False  # set to True if you want to use the best individuals from the old population + some individuals created randomly
    if pop_random_seed is False:
        perc_pop_toPass = None
        size_pop_toPass = None
    else:
        perc_pop_toPass = 0.3  # percentage of population used to start GP. The remaining part is initialized random. Used if pop_random_seed is True
        size_pop_toPass = int(size_pop_tot * perc_pop_toPass)

    if load_population is False:
        pop_random_seed = False

    flag_save = False  # set to true to save data
    flag_plot = False

    folder = 'Simulation1_stat'

    if flag_save:
        try:
            os.makedirs("Results/" + folder)
        except FileExistsError:
            pass

    savedata_file = "Results/" + folder + '/'


    obj.n_inputGP = 8
    obj.n_inputNN = 3
    obj.n_inputTOT = 11

    NNmodel, n_weights = utils_NN.create_NN_model(obj.n_inputNN, obj.n_hidden_layers, obj.nodes_per_hidden_layer,
                                                  obj.NNoutput, obj.activation)
    obj.NNinitConds = []
    for i in range(len(NNmodel.weights)):
        obj.NNinitConds.extend(np.ravel(NNmodel.get_weights()[i]))

    ####################################    P R I M I T I V E  -  S E T     ############################################


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
    creator.create("Fitness_Components", list)
    creator.create("Success_list", list)
    creator.create("Individual", list, fitness=creator.Fitness, fit_components=creator.Fitness_Components, successes=creator.Success_list)
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
    toolbox.register("mate", rop.xmateMultiple)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)
    toolbox.register("mutate", rop.xmutMultiple, expr=toolbox.expr_mut, unipb=0.55, shrpb=0.05, inspb=0.25, pset=pset,
                     creator=creator)
    toolbox.decorate("mate", gpfuns.staticLimitMod(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mutate", gpfuns.staticLimitMod(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mate", gpfuns.staticLimitMod(key=len, max_value=limit_size))
    toolbox.decorate("mutate", gpfuns.staticLimitMod(key=len, max_value=limit_size))



    NNinitName = 'NNinitconds.npy'
    np.save(NNinitName, obj.NNinitConds)

    ########################## Uncertainty parameters ##############################
    uncertain_atmo = True
    uncertain_aero = False
    uncertain = True

    test_unc_profiles = np.load('Test_uncertSurfaces.npy')
    train_unc_profiles = np.load('Train_uncertSurfaces.npy')

    v_points = np.linspace(obj.vmin, obj.vmax, train_unc_profiles[0].shape[1])
    h_points = np.linspace(obj.hmin, obj.hmax, train_unc_profiles[0].shape[1])

    ############################    Start GP ###############################################

    if start_gen == 0:
        # define indexes of uncertainty scenarios used and save them
        unc_data = pd.read_excel('results_uncertainty.xlsx')
        # Read the values of the file in the dataframe
        data = pd.DataFrame(unc_data, columns=['Index', 'Theta error', 'Lambda error', 'H error', 'Sum of errors'])
        data.sort_values('Sum of errors')
        available_indexes = data['Index'].values # stored indexes
        group_division = np.linspace(min(data['Sum of errors']), max(data['Sum of errors']), int(len(np.where(np.array((obj.change_steps))==0)[0]))+1) # find values of error for intervals division
        uncert_indexes = []
        for i in range(int(len(np.where(np.array((obj.change_steps))==0)[0]))-1):
            uncert_indexes.append(int(data['Index'][data['Sum of errors'].between(group_division[i], group_division[i+1])].values[0]))
        uncert_indexes.append(data['Index'].iloc[-1])
        available_indexes = [i for i in available_indexes if i not in uncert_indexes]
        uncert_indexes.extend(random.sample(available_indexes, int(len(np.where(np.array((obj.change_steps))!=0)[0]))))
        np.save('uncert_indexes.npy', uncert_indexes)
        np.save('available_indexes.npy', available_indexes)

    size_pop = size_pop_tot
    mutpb = copy(mutpb_start)
    cxpb = copy(cxpb_start)

    start = time.time()
    pop, log, hof = funs.main(toolbox, load_population, pop_path, creator, pop_random_seed, size_pop_toPass, size_pop,
                              size_pop_tot, hof_size, size_gen, Mu, Lambda, cxpb, mutpb, pset, obj, nbCPU, v_points,
                              h_points, train_unc_profiles, uncertain, uncertain_atmo, uncertain_aero, save_gen,
                              savedata_file, start_gen, None)

    end = time.time()
    t_offdesign = end - start
    print('Time elapsed: ', t_offdesign, ' s')
    with open('time_elapsed.txt', 'w') as f:
        f.write('Time elapsed {} s'.format(t_offdesign))
    best_ind = hof[-1]
    funs.plot_best_ind(obj, v_points, h_points, uncertain, uncertain_atmo, uncertain_aero, pset, savedata_file,
                       best_ind, test_unc_profiles, NNmodel, n_weights, NNinitName, flag_plot, obj.initial_ubd,
                       obj.initial_ubc)

