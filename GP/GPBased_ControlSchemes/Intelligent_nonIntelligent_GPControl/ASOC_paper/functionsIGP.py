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

import sys
import os
dirname = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(dirname, '..')))
sys.path.append(os.path.abspath(os.path.join(dirname, '..', '..', 'test_cases')))
sys.path.append(os.path.abspath(os.path.join(dirname, '..', 'IGP_adjoint')))
import numpy as np
import GP.GP_Algorithms.IGP.IGP_Functions as gpfuns
import GP.GP_Algorithms.IGP.Recombination_operators as rop
from GP.GPBased_ControlSchemes.OPGD_IGP.functionsIGPadjoint import propagate_forward
from scipy.integrate import simps
from deap import gp, tools, creator, base
import operator
import random

def main(nbCPU, toolbox, size_pop, size_gen, creator, Mu, Lambda, pset, obj, dynamics):
    """Main function used to run the IGP evolutionary process

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
            dynamics: function
                a function describing the plant's dynamics
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

    pop, log, pop_statistics, ind_lengths = gpfuns.eaMuPlusLambdaTol_OPGD_IGP(best_pop, toolbox, Mu, Lambda, size_gen,
                                                                                cxpb, mutpb, creator, stats=mstats,
                                                                                halloffame=hof, verbose=True, obj=obj,
                                                                                pset=pset, dynamics=dynamics,
                                                                                nbCPU=nbCPU)
    ####################################################################################################################

    return pop, log, hof, pop_statistics, ind_lengths


def evaluate_individualIGP(GPind, **k):
    """
    Function used to evaluate the fitness of the individual

    Attributes:
        GPind: list
            GP individual
        **k: kwargs

    Return:
        FIT: fitness of the individual
    """
    obj = k['kwargs']['obj']
    pset = k['kwargs']['pset']
    dynamics = k['kwargs']['dynamics']
    traj, contr, failure = propagate_forward(obj, GPind, pset, dynamics)
    if failure is True:
        return [1e6, 0]
    else:
        g = np.zeros(obj.Npoints)
        for i in range(obj.Npoints):
            g[i] = 0.5 * ((traj[i, :]-obj.xf).T @ obj.Qz @ (traj[i, :]-obj.xf) + np.array(([[contr[i]]])).T @ obj.Qu @ np.array(([[contr[i]]])))
        int_g = simps(g, obj.t_points)
        h = 0.5 * ((traj[-1, :]-obj.xf).T @ obj.Qt @ (traj[-1, :]-obj.xf))
        FIT = int_g + h
        return [FIT, 0]



def define_GP_model_oscillator(nEph, limit_height, limit_size, sim):
    """
    Function used to define the GP model in the oscillator test case

    Attributes:
        nEph: integer
            number of ephemeral constants
        limit_height: integer
            max height of GP tree
        limit_size: integer
            max size of GP tree
        sim: integer
            simulation id

    Return:
        pset: class
            primitive set
        toolbox: class
            GP building blocks
        creator: class
            GP building blocks
    """
    ####################################    P R I M I T I V E  -  S E T     ################################################

    pset = gp.PrimitiveSet("Main", 2)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    for i in range(nEph):
        pset.addEphemeralConstant("rand{}_{}".format(i, sim), lambda: round(random.uniform(-5, 5), 3))

    pset.renameArguments(ARG0='eX')
    pset.renameArguments(ARG1='eV')

    ################################################## TOOLBOX #############################################################

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate_individualIGP)
    toolbox.register("select", gpfuns.InclusiveTournamentV2, selected_individuals=1, fitness_size=2, parsimony_size=1.6,
                     creator=creator)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)
    toolbox.register("mutate", rop.xmut, expr=toolbox.expr_mut, unipb=0.5, shrpb=0.05, inspb=0.25, pset=pset,
                     creator=creator)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

    return pset, toolbox, creator

def define_GP_model_pendulum(nEph, limit_height, limit_size, sim, obj):
    """
        Function used to define the GP model in the pendulum test case

        Attributes:
            nEph: integer
                number of ephemeral constants
            limit_height: integer
                max height of GP tree
            limit_size: integer
                max size of GP tree
            sim: integer
                simulation id
            obj: class
                plant model

        Return:
            pset: class
                primitive set
            toolbox: class
                GP building blocks
            creator: class
                GP building blocks
        """
    ####################################    P R I M I T I V E  -  S E T     ################################################

    pset = gp.PrimitiveSet("Main", obj.n_states)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    for i in range(nEph):
        pset.addEphemeralConstant("rand{}_{}".format(i, sim), lambda: round(random.uniform(-10, 10), 3))

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
    toolbox.register("evaluate", evaluate_individualIGP)
    toolbox.register("select", gpfuns.InclusiveTournamentV2, selected_individuals=1, fitness_size=2, parsimony_size=1.6,
                     creator=creator)
    toolbox.register("mate", gp.cxOnePoint)  ### NEW ##
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)
    toolbox.register("mutate", rop.xmut, expr=toolbox.expr_mut, unipb=0.5, shrpb=0.05, inspb=0.25, pset=pset,
                     creator=creator)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

    return pset, toolbox, creator
