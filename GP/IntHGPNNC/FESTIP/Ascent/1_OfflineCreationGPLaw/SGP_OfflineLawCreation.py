# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2020 University of Strathclyde and Author ------
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
This script creates a control law using the standard implementation of Genetic Programming. 10 different control laws
are created on 10 different disturbance scenarios contained in the dataset GP_creationSet.npy
"""

import sys
import os
models_path = os.path.join(os.path.dirname( __file__ ), '../..', 'FESTIP_Models')
igpfun_path = os.path.join(os.path.dirname( __file__ ), '../../../../IGP')
sgpfun_path = os.path.join(os.path.dirname( __file__ ), '../../../../SGP')
data_path = os.path.join(os.path.dirname( __file__ ), '..', 'Datasets')
sys.path.append(models_path)
sys.path.append(igpfun_path)
sys.path.append(sgpfun_path)
sys.path.append(data_path)
from scipy.integrate import solve_ivp
import numpy as np
import operator
import random
from copy import copy
from deap import gp, base, creator, tools
import multiprocessing
from scipy.interpolate import PchipInterpolator
from time import time, strftime
import os
import scipy.io as sio
import _pickle as cPickle
import matplotlib
import SGP_Functions as sgpfuns
import IGP_Functions as igpfuns
import models_FESTIP as mods
import GP_PrimitiveSet as gpprim
import Recombination_operators as rops


matplotlib.rcParams.update({'font.size': 22})
timestr = strftime("%Y%m%d-%H%M%S")

############################ GP PARAMETERS ######################################
flag_save = False # set to true to save data
nEph = 2  # number of ephemeral constants used
limit_height = 25  # Max height (complexity) of the controller law
limit_size = 50  # Max size (complexity) of the controller law
fit_tol = 0.5  # threshodld of the first fitness function below which the evolutionary process stops
size_pop_tot = 300  # population size
size_gen = 150  # Gen size
Mu = int(size_pop_tot)
Lambda = int(size_pop_tot * 1.2)
mutpb_start = 0.2  # mutation probability
cxpb_start = 0.7  # crossover probability
mutpb = copy(mutpb_start)
cxpb = copy(cxpb_start)


######################### LOAD DATA ####################à
obj = mods.Spaceplane()

GP_points = np.load(data_path + "/GP_creationSet.npy")
ntot = len(GP_points)

ref_traj = sio.loadmat(models_path + "/reference_trajectory_ascent.mat")
tref = ref_traj['timetot'][0]
total_time_simulation = tref[-1]
tfin = tref[-1]

vref = ref_traj['vtot'][0]
chiref = ref_traj['chitot'][0]
gammaref = ref_traj['gammatot'][0]
tetaref = ref_traj['tetatot'][0]
lamref = ref_traj['lamtot'][0]
href = ref_traj['htot'][0]
mref = ref_traj['mtot'][0]
alfaref = ref_traj['alfatot'][0]
deltaref = ref_traj['deltatot'][0]

indexes = np.where(np.diff(tref) == 0)
vref = np.delete(vref, indexes)
chiref = np.delete(chiref, indexes)
gammaref = np.delete(gammaref, indexes)
tetaref = np.delete(tetaref, indexes)
lamref = np.delete(lamref, indexes)
href = np.delete(href, indexes)
mref = np.delete(mref, indexes)
alfaref = np.delete(alfaref, indexes)
deltaref = np.delete(deltaref, indexes)
tref = np.delete(tref, indexes)

vfun = PchipInterpolator(tref, vref)
chifun = PchipInterpolator(tref, chiref)
gammafun = PchipInterpolator(tref, gammaref)
tetafun = PchipInterpolator(tref, tetaref)
lamfun = PchipInterpolator(tref, lamref)
hfun = PchipInterpolator(tref, href)
mfun = PchipInterpolator(tref, mref)
alfafun = PchipInterpolator(tref, alfaref)
deltafun = PchipInterpolator(tref, deltaref)

with open(models_path + "/impulse.dat") as f:
    impulse = []
    for line in f:
        line = line.split()
        if line:
            line = [float(i) for i in line]
            impulse.append(line)

f.close()

cl = sio.loadmat(models_path + "/crowd_cl.mat")['cl']
cd = sio.loadmat(models_path + "/crowd_cd.mat")['cd']
cm = sio.loadmat(models_path + "/crowd_cm.mat")['cm']
obj.angAttack = sio.loadmat(models_path + "/crowd_alpha.mat")['newAngAttack'][0]
obj.mach = sio.loadmat(models_path + "/crowd_mach.mat")['newMach'][0]
presv = []
spimpv = []

for i in range(len(impulse)):
    presv.append(impulse[i][0])
    spimpv.append(impulse[i][1])

presv = np.asarray(presv)
spimpv = np.asarray(spimpv)

######################### SAVE DATA #############################################

if flag_save:
    os.makedirs("Results/SGP_FestipIC_OfflineControlLaw_{}it_Res_{}_{}".format(str(ntot), os.path.basename(__file__), timestr))
    savedata_file = "Results/SGP_FestipIC_OfflineControlLaw_{}it_Res_{}_{}/".format(str(ntot), os.path.basename(__file__), timestr)


################################# M A I N ###############################################

def main(height_start, v_wind, deltaH):
    nbCPU = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(nbCPU)
    toolbox.register("map", pool.map)

    best_pop = toolbox.population(n=size_pop_tot)

    hof = igpfuns.HallOfFame(10)

    print("INITIAL POP SIZE: %d" % size_pop_tot)
    print("GEN SIZE: %d" % size_gen)
    print("\n")

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", igpfuns.Min)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   ###################################

    pop, log, data, all_lengths = sgpfuns.eaMuPlusLambdaTolSimple(best_pop, toolbox, Mu, Lambda, size_gen, cxpb, mutpb,
                                                                  pset, creator, stats=mstats, halloffame=hof, verbose=True,
                                                                  fit_tol=fit_tol, height_start=height_start,
                                                                  v_wind=v_wind, deltaH=deltaH, cl=cl, cd=cd, cm=cm,
                                                                  spimpv=spimpv, presv=presv, change_time=change_time,
                                                                  obj=obj, vfun=vfun, chifun=chifun, gammafun=gammafun,
                                                                  hfun=hfun, alfafun=alfafun, deltafun=deltafun,
                                                                  tfin=tfin, x_ini_h=x_ini_h, mod_hof=True, check=False)
    ####################################################################################################################

    pool.close()
    pool.join()
    return pop, log, hof


####################################    P R I M I T I V E  -  S E T     ################################################

pset = gp.PrimitiveSet("Main", 4)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(operator.mul, 2, name='Mul')
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
pset.renameArguments(ARG3='errH')

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
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", mods.evaluate)

toolbox.register("select", sgpfuns.xselDoubleTournament, fitness_size=2, parsimony_size=1.6, fitness_first=True)
toolbox.register("mate", rops.xmateSimple)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)
toolbox.register("mutate", rops.xmutSimple, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", igpfuns.staticLimitMod(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", igpfuns.staticLimitMod(key=operator.attrgetter("height"), max_value=limit_height))

toolbox.decorate("mate", igpfuns.staticLimitMod(key=len, max_value=limit_size))
toolbox.decorate("mutate", igpfuns.staticLimitMod(key=len, max_value=limit_size))

########################################################################################################################

if __name__ == "__main__":

    success_range = 0
    nt = 0
    t_evals = []
    while nt < ntot:

        height_start = copy(GP_points[nt][0])
        deltaH = copy(GP_points[nt][2])
        v_wind = copy(GP_points[nt][1])
        height_orig = copy(GP_points[nt][0])
        delta_orig = copy(GP_points[nt][2])
        wind_orig = copy(GP_points[nt][1])

        size_pop = size_pop_tot
        mutpb = copy(mutpb_start)
        cxpb = copy(cxpb_start)
        print(" ---------------Iter={}, V wind={}, Gust Height Start={}, "
              "Size Gust Zone={} -----------------".format(nt, round(v_wind,2), round(height_start,2), round(deltaH,2)))
        x_ini = [vref[0], chiref[0], gammaref[0], tetaref[0], lamref[0], href[0], mref[0]]
        idx = 0
        def find_height(t, x, *args):
            return height_start - x[5]

        find_height.terminal = True

        ####### propagation to find where the gust zone starts ###############
        tev = np.linspace(2, tfin, 200)
        init = solve_ivp(mods.sys_init, [2, tfin], x_ini, events=find_height, method='RK45', t_eval=tev,
                         args=(cl, cd, cm, spimpv, presv, obj, alfafun, deltafun))
        change_time = init.t[-1]

        x_ini_h = [init.y[0, -1], init.y[1, -1], init.y[2, -1], init.y[3, -1], init.y[4, -1], init.y[5, -1], init.y[6, -1]]

        ########### Start of GP ####################

        start = time()
        pop, log, hof = main(height_start, v_wind, deltaH)
        end = time()
        ########### End of GP ####################

        t_offdesign = end - start
        t_evals.append(t_offdesign)
        print("Time elapsed: {}".format(t_offdesign))

        if flag_save:  # save hall of fame
            output = open(savedata_file + "SGP_hof_{}.pkl".format(nt), "wb")
            cPickle.dump(hof, output, -1)
            output.close()
        print(hof[-1][0])
        print(hof[-1][1])
        fAlpha = gp.compile(hof[-1][0], pset=pset)
        fDelta = gp.compile(hof[-1][1], pset=pset)
        #####  Test obtained control law  #####

        Npoints = 500
        t_max_int = 20
        solgp, t_stop, stop_index = mods.RK4(change_time, tfin, mods.sys2GP_uncert, Npoints, x_ini_h, t_max_int,
                                             args=(fAlpha, fDelta, wind_orig, cl, cd, cm, spimpv, presv, height_orig,
                                                   delta_orig, vfun, chifun, gammafun, hfun, alfafun, deltafun,
                                                   obj))

        vout = solgp[:stop_index, 0]
        chiout = solgp[:stop_index, 1]
        gammaout = solgp[:stop_index, 2]
        tetaout = solgp[:stop_index, 3]
        lamout = solgp[:stop_index, 4]
        hout = solgp[:stop_index, 5]
        mout = solgp[:stop_index, 6]
        ttgp = np.linspace(change_time, t_stop, stop_index)
        v_ass, chi_ass = mods.vass(solgp[stop_index - 1, :], obj.omega)

        v_orbit = np.sqrt(obj.GMe / (obj.Re + hout[-1]))

        if np.cos(obj.incl) / np.cos(lamout[-1]) > 1:
            chi_orbit = np.pi
        else:
            if np.cos(obj.incl) / np.cos(lamout[-1]) < - 1:
                chi_orbit = 0.0
            else:
                chi_orbit = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lamout[-1]))

        print("V orbit: {} - {} - {} m/s".format(v_orbit * 0.95, v_ass, v_orbit * 1.05))
        print("Chi orbit: {} - {} - {} deg".format(np.rad2deg(chi_orbit * 0.8), np.rad2deg(chi_ass),
                                                   np.rad2deg(chi_orbit * 1.2)))
        print("V ref: {} - {} - {} m/s".format(vref[-1] * 0.99, vout[-1], vref[-1] * 1.01))
        print("Chi ref: {} - {} - {} deg".format(np.rad2deg(chiref[-1] * 0.99), np.rad2deg(chiout[-1]),
                                                 np.rad2deg(chiref[-1] * 1.01)))
        print("Gamma ref: {} - {} - {} deg".format(-0.5, np.rad2deg(gammaout[-1]), 0.5))
        print("Teta ref: {} - {} - {} deg".format(np.rad2deg(tetaref[-1] * 1.01), np.rad2deg(tetaout[-1]),
                                                  np.rad2deg(tetaref[-1] * 0.99)))
        print("Lambda ref: {} - {} - {} deg".format(np.rad2deg(lamref[-1] * 0.99), np.rad2deg(lamout[-1]),
                                                    np.rad2deg(lamref[-1] * 1.01)))
        print("H ref: {} - {} - {} m/s".format(href[-1] * 0.99, hout[-1], href[-1] * 1.01))

        if (v_orbit * 0.95 <= v_ass <= v_orbit * 1.05 and chi_orbit * 0.8 <= chi_ass <= chi_orbit * 1.2 and
            np.deg2rad(-0.5) <= gammaout[-1] <= np.deg2rad(0.5) and href[-1] * 0.99 <= hout[-1] <= href[-1] * 1.01) or \
                (vref[-1] * 0.99 <= vout[-1] <= vref[-1] * 1.01 and
                 chiref[-1] * 0.99 <= chiout[-1] <= chiref[-1] * 1.01 and
                 np.deg2rad(-0.5) <= gammaout[-1] <= np.deg2rad(0.5) and
                 tetaref[-1] * 1.01 <= tetaout[-1] <= tetaref[-1] * 0.99 and
                 lamref[-1] * 0.99 <= lamout[-1] <= lamref[-1] * 1.01 and
                 href[-1] * 0.99 <= hout[-1] <= href[-1] * 1.01):  # tolerance of 1%
            success_range += 1
            print("Success")

        if flag_save:
            np.save(savedata_file + "{}_v_out".format(nt), vout)
            np.save(savedata_file + "{}_chi_out".format(nt), chiout)
            np.save(savedata_file + "{}_gamma_out".format(nt), gammaout)
            np.save(savedata_file + "{}_teta_out".format(nt), tetaout)
            np.save(savedata_file + "{}_lam_out".format(nt), lamout)
            np.save(savedata_file + "{}_h_out".format(nt), hout)
            np.save(savedata_file + "{}_m_out".format(nt), mout)
            np.save(savedata_file + "{}_t_out".format(nt), ttgp)

        nt += 1

    print("Success Rate {}/{}".format(success_range, ntot))

    if flag_save:
        np.save(savedata_file + "Success_range", success_range)
        np.save(savedata_file + "Evaluation_time", t_evals)










