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

"""
This script creates a control law using the Inclusive Genetic Programming (IGP). 10 different control laws
are created on 10 different disturbance scenarios contained in the dataset GP_creationSet.npy
"""

import sys
import os
models_path = os.path.join(os.path.dirname( __file__ ), '..', 'FESTIP_Models')
gpfun_path = os.path.join(os.path.dirname( __file__ ), '..', 'GP_Functions')
data_path = os.path.join(os.path.dirname( __file__ ), '..', 'Datasets')
sys.path.append(models_path)
sys.path.append(gpfun_path)
sys.path.append(data_path)
import numpy as np
import operator
import random
from copy import copy
from deap import gp, base, creator, tools
import multiprocessing
from scipy.interpolate import PchipInterpolator
from time import time, strftime
import _pickle as cPickle
import matplotlib
import GP.GP_Algorithms.IGP.IGP_Functions as funs
import GP.GPBased_ControlSchemes.Intelligent_nonIntelligent_GPControl.FESTIP.FESTIP_Models.models_FESTIP as mods
import scipy.io as sio
import GP.GP_Algorithms.GP_PrimitiveSet as gpprim
import GP.GP_Algorithms.IGP.Recombination_operators as rop

matplotlib.rcParams.update({'font.size': 22})
timestr = strftime("%Y%m%d-%H%M%S")
random.seed(135)

############################ GP PARAMETERS ######################################
flag_save = False  # set to true to save data
learning = False
nEph = 2  # number of ephemeral constants
limit_height = 30  # Max height (complexity) of the controller law
limit_size = 50  # Max size (complexity) of the controller law
size_pop_tot = 300 # population size
hof_size = int(size_pop_tot*0.05)
if learning:
    old_hof = funs.HallOfFame(hof_size)
size_gen = 150  # Gen size
Mu = int(size_pop_tot)
Lambda = int(size_pop_tot * 1.2)
mutpb_start = 0.7  # mutation probability
cxpb_start = 0.2 # crossover probability
mutpb = copy(mutpb_start)
cxpb = copy(cxpb_start)
Npoints = 400 # number of integration points
t_max_int = 200 # maximum integration time
th = 0.01 # range outside which the first propagation is stopped
t_change = 100  # time at which the uncertainty is applied
######################### LOAD DATA ####################à
obj = mods.Spaceplane_Reentry()


ref_traj = sio.loadmat(models_path + "/reference_trajectory_reentry_GPOnly.mat")
tref = ref_traj['timetot'][0]
total_time_simulation = tref[-1]
tfin = tref[-1]

vref = ref_traj['vtot'][0]
chiref = ref_traj['chitot'][0]
gammaref = ref_traj['gammatot'][0]
tetaref = ref_traj['tetatot'][0]
lamref = ref_traj['lamtot'][0]
href = ref_traj['htot'][0]
alfaref = ref_traj['alfatot'][0]
sigmaref = ref_traj['sigmatot'][0]

indexes = np.where(np.diff(tref) == 0)
vref = np.delete(vref, indexes)
chiref = np.delete(chiref, indexes)
gammaref = np.delete(gammaref, indexes)
tetaref = np.delete(tetaref, indexes)
lamref = np.delete(lamref, indexes)
href = np.delete(href, indexes)
alfaref = np.delete(alfaref, indexes)
sigmaref = np.delete(sigmaref, indexes)
tref = np.delete(tref, indexes)

vfun = PchipInterpolator(tref, vref)
chifun = PchipInterpolator(tref, chiref)
gammafun = PchipInterpolator(tref, gammaref)
tetafun = PchipInterpolator(tref, tetaref)
lamfun = PchipInterpolator(tref, lamref)
hfun = PchipInterpolator(tref, href)
alfafun = PchipInterpolator(tref, alfaref)
sigmafun = PchipInterpolator(tref, sigmaref)


cl = sio.loadmat(models_path + "/crowd_cl.mat")['cl']
cd = sio.loadmat(models_path + "/crowd_cd.mat")['cd']
obj.angAttack = sio.loadmat(models_path + "/crowd_alpha.mat")['newAngAttack'][0]
obj.mach = sio.loadmat(models_path + "/crowd_mach.mat")['newMach'][0]


################################# M A I N ###############################################
def initPOP1():
    global old_hof
    # this function outputs the old hall of fame shuffling the order of the individuals
    res = old_hof.shuffle()
    return res


def main(lbt, ubt, lbp, ubp, lb, ub, hc, Mc, x_ini_h, t_startGP, nt, uncertain=True,
         uncertain_atmo=True, uncertain_aero=True, extend=False, uncert_Funs=None):
    nbCPU = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(nbCPU)
    toolbox.register("map", pool.map)

    #  Perform search for the initial population with the highest entropy
    old_entropy = 0
    for i in range(1):
        pop = funs.POP(toolbox.population(n=size_pop), creator)
        best_pop = pop.items
        if pop.entropy > old_entropy and len(pop.indexes) == len(pop.categories) - 1:
            best_pop = pop.items
            old_entropy = pop.entropy
    if learning and nt > 0:
        pop2 = toolbox.popx()
        for ind in pop2:
            del ind.fitness.values
        best_pop = pop2 + best_pop

    hof = funs.HallOfFame(hof_size)

    print("INITIAL POP SIZE: %d" % size_pop)
    print("GEN SIZE: %d" % size_gen)
    print("\n")

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", funs.Min)

    ####################################   EVOLUTIONARY ALGORITHM - EXECUTION   ###################################

    pop, log, pop_statistics, ind_lengths = funs.eaMuPlusLambdaTol_FestipReentry(best_pop, toolbox, Mu, Lambda, size_gen, cxpb, mutpb,
                                                                   pset, creator, stats=mstats, halloffame=hof,
                                                                   verbose=True, fit_tol=None, cl=cl, cd=cd, obj=obj,
                                                                   vfun=vfun, chifun=chifun, gammafun=gammafun, tetafun=tetafun,
                                                                   lamfun=lamfun, hfun=hfun, alfafun=alfafun, sigmafun=sigmafun, tfin=tfin,
                                                                   x_ini_h=x_ini_h, Npoints=Npoints, t_max_int=t_max_int, lbt=lbt, ubt=ubt,
                                                                   lbp=lbp, ubp=ubp, lb=lb, ub=ub, hc=hc, Mc=Mc, t_startGP=t_startGP,
                                                                   uncertain=uncertain, uncertain_atmo=uncertain_atmo, uncertain_aero=uncertain_aero,
                                                                   extend=extend, uncert_Funs=uncert_Funs)

    ####################################################################################################################

    pool.close()
    pool.join()
    return pop, log, hof


####################################    P R I M I T I V E  -  S E T     ################################################

pset = gp.PrimitiveSet("Main", 6)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(operator.mul, 2, name='Mul')
#pset.addPrimitive(gpprim.Div, 2, name='Div')
pset.addPrimitive(gpprim.TriAdd, 3)
pset.addPrimitive(np.tanh, 1, name="Tanh")
pset.addPrimitive(gpprim.ModSqrt, 1)
pset.addPrimitive(gpprim.ModLog, 1)
pset.addPrimitive(gpprim.ModExp, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)

for i in range(nEph):
    pset.addEphemeralConstant("rand{}".format(i), lambda: round(random.uniform(-10, 10), 4))

pset.renameArguments(ARG0='errV')
pset.renameArguments(ARG1='errChi')
pset.renameArguments(ARG2='errGamma')
pset.renameArguments(ARG3='errTeta')
pset.renameArguments(ARG4='errLam')
pset.renameArguments(ARG5='errH')


################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)
creator.create("SubIndividual", gp.PrimitiveTree)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genFull, pset=pset, type_=pset.ret, min_=1, max_=4)
toolbox.register("leg", tools.initIterate, creator.SubIndividual, toolbox.expr)
toolbox.register("legs", tools.initCycle, list, [toolbox.leg, toolbox.leg], n=1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.legs)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("popx", tools.initIterate, list, initPOP1)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", mods.evaluate_reentry)
toolbox.register("select", funs.InclusiveTournamentV2, selected_individuals=1, fitness_size=2, parsimony_size=1.6, creator=creator)
toolbox.register("mate", rop.xmate)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)
toolbox.register("mutate", rop.xmut, expr=toolbox.expr_mut, unipb=0.5, shrpb=0.05, inspb=0.25, pset=pset, creator=creator)
toolbox.decorate("mate", funs.staticLimitMod(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", funs.staticLimitMod(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mate", funs.staticLimitMod(key=len, max_value=limit_size))
toolbox.decorate("mutate", funs.staticLimitMod(key=len, max_value=limit_size))

########################################################################################################################

if __name__ == "__main__":
    hc = obj.hmax
    ranges = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    lbt_ref = 0.1
    ubt_ref = 0.5
    lbp_ref = 0.01
    ubp_ref = 0.5
    lb_ref  = 0.1
    ub_ref  = 0.2
    Mc = 10
    t_start = 0
    uncertain_atmo = True
    uncertain_aero = True
    uncertain = True
    extend = False
    x_ini = [vref[0], chiref[0], gammaref[0], tetaref[0], lamref[0], href[0]]

    for i in range(0, len(ranges)):
        print("-----------------------------------------------------------------------------------------------------------Range {}".format(ranges[i]))
        if flag_save:
            try:
                os.makedirs("Results/GPONLY_journal_results{}".format(i))
            except FileExistsError:
                pass
        savedata_file = "Results/GPONLY_journal_results{}/".format(i)
        uncert_profiles = np.load(data_path + "/uncertProfiles_GP{}.npy".format(i))
        ntot = len(uncert_profiles)
        success_range = 0
        nt = 0
        t_evals = []
        while nt < ntot:
            lbt = lbt_ref * ranges[i] / 100
            ubt = ubt_ref * ranges[i] / 100
            lbp = lbp_ref * ranges[i] / 100
            ubp = ubp_ref * ranges[i] / 100
            lb = lb_ref * ranges[i] / 100
            ub = ub_ref * ranges[i] / 100
            print("\n")
            print("--------------------------------------------------------------------------------------------------------- {}".format(nt))
            if learning:
                size_pop = size_pop_tot - len(old_hof)  # Pop size
            else:
                size_pop = size_pop_tot
            mutpb = copy(mutpb_start)
            cxpb = copy(cxpb_start)

            t_points = np.linspace(0, tfin, uncert_profiles[nt].shape[0])
            uncert_funP = PchipInterpolator(t_points, uncert_profiles[nt][:, 0])
            uncert_funT = PchipInterpolator(t_points, uncert_profiles[nt][:, 1])
            uncert_funA = PchipInterpolator(t_points, uncert_profiles[nt][:, 2])
            uncert_Funs = [uncert_funP, uncert_funT, uncert_funA]

            ###### FIND INITIAL CONDITIONS WHERE UNCERTAINTY IS APPLIED #####
            x_ini_h, t_startGP = mods.prop_untilCviol(tfin, Npoints, x_ini, vfun, chifun, gammafun, tetafun, lamfun,
                                                      hfun, th, cl, cd, alfafun, sigmafun, obj, hc, lbt, ubt, lbp, ubp,
                                                      lb, ub, Mc, uncertain_atmo, uncertain_aero, t_change, t_start, False,
                                                      uncert_Funs)

            ########### Start of GP ####################

            t_points = np.linspace(t_startGP, tfin, uncert_profiles[nt].shape[0])
            uncert_funP = PchipInterpolator(t_points, uncert_profiles[nt][:, 0])
            uncert_funT = PchipInterpolator(t_points, uncert_profiles[nt][:, 1])
            uncert_funA = PchipInterpolator(t_points, uncert_profiles[nt][:, 2])
            uncert_Funs = [uncert_funP, uncert_funT, uncert_funA]

            start = time()
            pop, log, hof = main(lbt, ubt, lbp, ubp, lb, ub, hc, Mc, x_ini_h, t_startGP, nt, uncertain,
                                 uncertain_atmo, uncertain_aero, extend, uncert_Funs)
            end = time()
            if learning:
                old_hof.update(hof, for_feasible=True)
            ############ End of GP ####################

            t_offdesign = end - start
            t_evals.append(t_offdesign)
            print("Time elapsed: {}".format(t_offdesign))
            if flag_save:  # save hall of fame
                output = open(savedata_file + "IGP_hof_{}.pkl".format(nt), "wb")
                cPickle.dump(hof, output, -1)
                output.close()
            print(hof[-1][0])
            print(hof[-1][1])
            falfa = toolbox.compile(hof[-1][0])
            fsigma = toolbox.compile(hof[-1][1])
            #####  Test obtained control law  #####

            solgp, tend, end_index = mods.RK4(t_startGP, tfin, mods.sys2GP_uncert, Npoints, x_ini_h, 200,
                                              args=(falfa, fsigma, cl, cd, vfun, chifun, gammafun, tetafun, lamfun, hfun,
                                                    alfafun, sigmafun, obj, lbt, ubt, lbp, ubp, lb, ub, hc, Mc, uncertain,
                                                    uncertain_atmo, uncertain_aero, False, uncert_Funs))

            vout = solgp[:, 0]
            chiout = solgp[:, 1]
            gammaout = solgp[:, 2]
            tetaout = solgp[:, 3]
            lamout = solgp[:, 4]
            hout = solgp[:, 5]
            ttgp = np.linspace(t_startGP, tfin, Npoints)

            if (obj.hend - obj.hend_tol <= hout[-1] and hout[-1] <= obj.hend + obj.hend_tol) and \
                    (obj.lamend - obj.lamend_tol <= lamout[-1] and lamout[-1] <= obj.lamend + obj.lamend_tol) and \
                    (obj.tetaend - obj.tetaend_tol <= tetaout[-1] and tetaout[-1] <= obj.tetaend + obj.tetaend_tol):
                success_range += 1
                print("Success")

            if flag_save:
                np.save(savedata_file + "{}_v_out".format(nt), vout)
                np.save(savedata_file + "{}_chi_out".format(nt), chiout)
                np.save(savedata_file + "{}_gamma_out".format(nt), gammaout)
                np.save(savedata_file + "{}_teta_out".format(nt), tetaout)
                np.save(savedata_file + "{}_lam_out".format(nt), lamout)
                np.save(savedata_file + "{}_h_out".format(nt), hout)
                np.save(savedata_file + "{}_t_out".format(nt), ttgp)

            nt += 1

        print("Success Rate {}/{}".format(success_range, ntot))

        if flag_save:
            np.save(savedata_file + "Success_range", success_range)
            np.save(savedata_file + "Evaluation_time", t_evals)










