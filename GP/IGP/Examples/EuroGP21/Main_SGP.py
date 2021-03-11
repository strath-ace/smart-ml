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

Main script to run the standard genetic programming (SGP)

'''

import sys
import os
data_path = os.path.join(os.path.dirname( __file__ ), '..', 'Datasets')
gpfun_path = os.path.join(os.path.dirname( __file__ ), '..', 'EuroGP2020/GP_Functions')
sys.path.append(gpfun_path)
sys.path.append(data_path)
import numpy as np
import operator
import random
from copy import copy
from deap import gp, base, creator, tools
import multiprocessing
from time import time, strftime
import matplotlib
from SGP import SGP_Functions as funs
import IGP_Functions as ifuns
import GP_PrimitiveSet as gpprim
import Recombination_operators as rops
import Benchmarks


bench = 'koza1'  # select benchmark to run from [koza1, korns11, S1, S2, UB, ENC, ENH, CCS, ASN]
save = True
mod_hof = False  # True to use modified version of hall of fame. False to use standard DEAP hof
terminals, npoints = Benchmarks.out_terminals(bench)

matplotlib.rcParams.update({'font.size': 18})

nEph = 1  # number of ephemeral constants

limit_height = 15  # Max height (complexity) of the gp law
limit_size = 30  # Max size (complexity) of the gp law

size_pop_tot = 300
size_gen = 300  # Gen size

fit_tol = 1e-10 # stopping criteria on fitness value. I fitness gets below fit_tol, the process stops

Mu = int(size_pop_tot)
Lambda = int(size_pop_tot * 1.2)

mutpb = 0.2  # mutation rate
cxpb = 0.8  # crossover rate

nbCPU = multiprocessing.cpu_count()  # number of CPUs to use for multithreading

timestr = strftime("%Y%m%d-%H%M%S")


def main():

    pool = multiprocessing.Pool(nbCPU)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=size_pop_tot)

    hof = tools.HallOfFame(10)

    print("INITIAL POP SIZE: %d" % size_pop_tot)
    print("GEN SIZE: %d" % size_gen)
    print("\n")

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", ifuns.Min)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   ###################################

    pop, log, pop_statistics, ind_lengths = funs.eaMuPlusLambdaTolSimple(pop, toolbox, mu=Mu, lambda_=Lambda, cxpb=cxpb,
                                                                         mutpb=mutpb, ngen=size_gen, creator=creator,
                                                                         pset=pset, stats=mstats, halloffame=hof,
                                                                         verbose=True, mod_hof=mod_hof, fit_tol=fit_tol,
                                                                         check=False)
    ####################################################################################################################

    pool.close()
    pool.join()
    return pop, log, hof, pop_statistics, ind_lengths


##################################  F I T N E S S    F U N C T I O N    ##############################################


def evaluate(individual, pset, **kwargs):
    f_ind = gp.compile(individual, pset=pset)
    if terminals == 1:
        out = f_ind(copy(input_true))
    elif terminals == 2:
        out = f_ind(copy(input_true[0,:]), copy(input_true[1,:]))
    elif terminals == 5:
        out = f_ind(copy(input_true[0,:]), copy(input_true[1,:]), copy(input_true[2,:]), copy(input_true[3,:]), copy(input_true[4,:]))
    elif terminals == 8:
        out = f_ind(copy(input_true[0,:]), copy(input_true[1,:]), copy(input_true[2,:]), copy(input_true[3,:]),
                    copy(input_true[4,:]), copy(input_true[5,:]), copy(input_true[6,:]), copy(input_true[7,:]))
    err = copy(output_true) - out
    fitness = np.sqrt(sum(err**2)/(len(err)))
    return [fitness, 0.0]


####################################    P R I M I T I V E  -  S E T     ################################################

pset = gp.PrimitiveSet("Main", terminals)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(gpprim.TriAdd, 3)
pset.addPrimitive(gpprim.TriMul, 3)
pset.addPrimitive(np.tanh, 1)
pset.addPrimitive(gpprim.Square, 1)
pset.addPrimitive(gpprim.ModLog, 1)
pset.addPrimitive(gpprim.ModExp, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)


for i in range(nEph):
    pset.addEphemeralConstant("rand{}".format(i), lambda: round(random.uniform(-10, 10), 4))

if terminals == 1:
    pset.renameArguments(ARG0='x')
elif terminals == 2:
    pset.renameArguments(ARG0='x1')
    pset.renameArguments(ARG1='x2')
elif terminals == 5:
    pset.renameArguments(ARG0='x1')
    pset.renameArguments(ARG1='x2')
    pset.renameArguments(ARG2='x3')
    pset.renameArguments(ARG3='x4')
    pset.renameArguments(ARG4='x5')
elif terminals == 8:
    pset.renameArguments(ARG0='x1')
    pset.renameArguments(ARG1='x2')
    pset.renameArguments(ARG2='x3')
    pset.renameArguments(ARG3='x4')
    pset.renameArguments(ARG4='x5')
    pset.renameArguments(ARG5='x6')
    pset.renameArguments(ARG6='x7')
    pset.renameArguments(ARG7='x8')

################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))  # , -0.1, -0.08, -1.0))    # MINIMIZATION OF THE FITNESS FUNCTION
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)  ### NEW ###
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  ### NEW ###
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selDoubleTournament, fitness_size=2, parsimony_size=1.2, fitness_first=True)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)
toolbox.register("mutate", rops.xmut, expr=toolbox.expr_mut, unipb=0.5, shrpb=0.05, inspb=0.25, pset=pset, creator=creator)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

########################################################################################################################

if __name__ == '__main__':
    if save is True:
        try:
            os.mkdir('Results/SGP_{}'.format(bench))
        except FileExistsError:
            pass
    n = 0
    ntot = 300

    if save is True:
        to_save = np.array(['t_evaluate', 'RMSE_train', 'RMSE_test'])
    while n < ntot:
        f, input_true, output_true, input_test, output_test = Benchmarks.select_testcase(bench)
        print("----------------------- Iteration {} ---------------------------".format(n))
        start = time()
        pop, log, hof, pop_statistics, ind_lengths = main()
        end = time()
        t_offdesign = end - start
        best_ind = hof[0]
        print(best_ind)
        print(best_ind.fitness)

        f_best = gp.compile(best_ind, pset=pset)
        if terminals == 1:
            y_best_train = f_best(copy(input_true))
            y_best_test = f_best(copy(input_test))
        elif terminals == 2:
            y_best_train = f_best(copy(input_true[0,:]), copy(input_true[1,:]))
            y_best_test = f_best(copy(input_test[0, :]), copy(input_test[1, :]))
        elif terminals == 5:
            y_best_train = f_best(copy(input_true[0,:]), copy(input_true[1,:]), copy(input_true[2,:]), copy(input_true[3,:]), copy(input_true[4,:]))
            y_best_test = f_best(copy(input_test[0,:]), copy(input_test[1,:]), copy(input_test[2,:]), copy(input_test[3,:]), copy(input_test[4,:]))
        elif terminals == 8:
            y_best_train = f_best(copy(input_true[0,:]), copy(input_true[1,:]), copy(input_true[2,:]), copy(input_true[3,:]),
                                  copy(input_true[4,:]), copy(input_true[5,:]), copy(input_true[6,:]), copy(input_true[7,:]))
            y_best_test  = f_best(copy(input_test[0,:]), copy(input_test[1,:]), copy(input_test[2,:]), copy(input_test[3,:]),
                                  copy(input_test[4,:]), copy(input_test[5,:]), copy(input_test[6,:]), copy(input_test[7,:]))
        err_train = copy(output_true) - y_best_train
        RMSE_train = np.sqrt(sum(err_train ** 2) / (len(err_train)))
        err_test = copy(output_test) - y_best_test
        RMSE_test = np.sqrt(sum(err_test ** 2) / (len(err_test)))

        if save is True:
            np.save("Results/SGP_{}/{}_SGP_{}_POP_STATS".format(bench, n, bench), pop_statistics)
            np.save("Results/SGP_{}/{}_SGP_{}_IND_LENGTHS".format(bench, n, bench), ind_lengths)
            np.save("Results/SGP_{}/{}_SGP_{}_GEN".format(bench, n, bench), log.select("gen"))
            np.save("Results/SGP_{}/{}_SGP_{}_FIT".format(bench, n, bench), np.array(log.chapters['fitness'].select('min'))[:,0])
            to_save = np.vstack((to_save, [t_offdesign, RMSE_train, RMSE_test]))
        n += 1

    if save is True:
        np.save("Results/SGP_{}/{}_SGP_{}_T_RMSE".format(bench, n, bench), to_save)











