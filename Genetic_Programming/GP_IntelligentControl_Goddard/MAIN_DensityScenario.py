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
This script simulates the Intelligent Control of a Goddard Rocket with 2 controls where the optimal trajectory is
defined with a simplified density model and during the flight a more complex density model is used. When the tracking
error becomes greater then a certain threshold, the new Genetic Programming control law is evaluated.
"""

from scipy.integrate import solve_ivp, simps
import numpy as np
import operator
import random
from deap import gp, base, creator, tools
import matplotlib.pyplot as plt
import multiprocessing
from scipy.interpolate import PchipInterpolator
from time import time, strftime
from functools import partial
import GP_PrimitiveSet as gpprim
import os
import DEAP_Mods as mods
import DensityScenario_utils as utils
import _pickle as cPickle
import statistics
from copy import copy

##########################  TUNABLE PARAMETERS #####################################
flag_save = True  # set to True to save output data from the evolutionary process and allow to save plots
learning = True  # set to True to use learning approach
if learning:
    str_learning = "Learning"
else:
    str_learning = "NoLearning"
plot = False  # set to True to save the plots of the final trajectory
plot_comparison = False  # set to True to save the plot of the comparison of the evaluation time of the different simulations

ntot = 1000
if ntot > 1 and plot is True:
    plot = False
    plot_tree = False
    print("It is advised to plot only one iteration since to plot multiple iterations on the same plots will be very messy and unclear")

nEph = 2  # number of ephemeral constants to use
limit_height = 8  # Max height (complexity) of the controller law
limit_size = 30  # Max size (complexity) of the controller law
fit_tol = 1.2  # below this value of fitness function the evolutions stops
delta_eval = 10  # time slots used for GP law evaluation

size_pop_tot = 250  # total population size
if learning:
    old_hof = mods.HallOfFame(int(size_pop_tot/1.5))

size_gen = 50  # Maximum generation number

Mu = int(size_pop_tot)

mutpb_start = 0.7  # mutation rate
cxpb_start = 0.2  # Crossover rate
mutpb = copy(mutpb_start)
cxpb = copy(cxpb_start)

nbCPU = multiprocessing.cpu_count()  # select the available CPU's. Set to 1 to not use multiprocessing

######################### SAVE DATA #############################################
timestr = strftime("%Y%m%d-%H%M%S")

if flag_save:
    os.makedirs("Results_2C_Density_{}_{}it_Res_{}_{}".format(str_learning, str(ntot), os.path.basename(__file__), timestr))
    savefig_file = "Results_2C_Density_{}_{}it_Res_{}_{}/Plot_".format(str_learning, str(ntot), os.path.basename(__file__), timestr)
    savedata_file = "Results_2C_Density_{}_{}it_Res_{}_{}/".format(str_learning, str(ntot), os.path.basename(__file__), timestr)

###############################  S Y S T E M - P A R A M E T E R S  ####################################################

class Rocket:

    def __init__(self):
        self.GMe = 3.986004418 * 10 ** 14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371.0 * 1000  # Earth Radius [m]
        self.Vr = np.sqrt(self.GMe / self.Re)  # m/s
        self.H0 = 10.0  # m
        self.V0 = 0.0
        self.M0 = 100000.0  # kg
        self.Mp = self.M0 * 0.99
        self.Cd = 0.6
        self.A = 4.0  # m2
        self.Isp = 300.0  # s
        self.g0 = 9.80665  # m/s2
        self.Tmax = self.M0 * self.g0 * 1.5
        self.MaxQ = 14000.0  # Pa
        self.MaxG = 8.0  # G
        self.Htarget = 400.0 * 1000  # m
        self.Rtarget = self.Re + self.Htarget  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s


tref = np.load("Files/time.npy")
total_time_simulation = tref[-1]
tfin = tref[-1]

Rref = np.load("Files/R.npy")
Thetaref = np.load("Files/Theta.npy")
Vrref = np.load("Files/Vr.npy")
Vtref = np.load("Files/Vt.npy")
mref = np.load("Files/m.npy")
Ttref = np.load("Files/Tt.npy")
Trref = np.load("Files/Tr.npy")

Rfun = PchipInterpolator(tref, Rref)
Thetafun = PchipInterpolator(tref, Thetaref)
Vrfun = PchipInterpolator(tref, Vrref)
Vtfun = PchipInterpolator(tref, Vtref)
mfun = PchipInterpolator(tref, mref)
Ttfun = PchipInterpolator(tref, Ttref)
Trfun = PchipInterpolator(tref, Trref)

Nstates = 5
Ncontrols = 2
obj = Rocket()


################################ M A I N ###############################################


def initPOP1():
    global old_hof
    # this function outputs the first n individuals of the hall of fame of the first GP run
    res = old_hof.shuffle()
    # for i in range(10):
    #   res.append(hof[0])
    return res


def main(size_pop, size_gen, Mu, cxpb, mutpb):
    global tfin, flag, n, old_hof
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun, Ttfun

    flag = False

    pool = multiprocessing.Pool(nbCPU)

    toolbox.register("map", pool.map)
    old_entropy = 0
    for i in range(10):
        pop = mods.POP(toolbox.population(n=size_pop))
        best_pop = pop.items
        if pop.entropy > old_entropy and len(pop.indexes) == len(pop.categories) - 1:
            best_pop = pop.items
            old_entropy = pop.entropy

    if learning:
        if nt > 0:
            pop2 = toolbox.popx()
            for ind in pop2:
                del ind.fitness.values
            best_pop = pop2 + best_pop

    hof = mods.HallOfFame(10)

    print("INITIAL POP SIZE: %d" % size_pop)
    print("GEN SIZE: %d" % size_gen)
    print("\n")

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", mods.Min)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   ###################################

    pop, log = mods.eaMuPlusLambdaTol(best_pop, toolbox, Mu, cxpb, mutpb, size_gen, fit_tol, limit_size, stats=mstats, halloffame=hof, verbose=True)
    ####################################################################################################################

    pool.close()
    pool.join()
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ##############################################


def evaluate(individual, Rfun, Thetafun, Vrfun, Vtfun, Trfun, Ttfun):
    global tfin, t_eval2, penalty, fit_old, mutpb, cxpb, Cd_new, flag

    penalty = []

    flag = False

    fTr = toolbox.compileR(expr=individual[0])
    fTt = toolbox.compileT(expr=individual[1])

    def sys(t, x, Rfun, Thetafun, Vrfun, Vtfun, Trfun, Ttfun):
        global penalty, flag, flag_offdesign, v_wind, flag_thrust, change_time, height_start, delta
        # State Variables
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        if R < obj.Re - 0.5 or np.isnan(R):
            penalty.append((R - obj.Re) / obj.Htarget)
            R = obj.Re
            flag = True

        if m < obj.M0 - obj.Mp or np.isnan(m):
            penalty.append((m - (obj.M0 - obj.Mp)) / obj.M0)
            m = obj.M0 - obj.Mp
            flag = True

        r = Rfun(t)
        th = Thetafun(t)
        vr = Vrfun(t)
        vt = Vtfun(t)

        er = r - R
        et = th - theta
        evr = vr - Vr
        evt = vt - Vt

        rho = rho_newmodel(R - obj.Re)
        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
        g0 = obj.g0
        Isp = obj.Isp

        Tr = Trfun(t) + fTr(er, evr)
        Tt = Ttfun(t) + fTt(et, evt)

        if np.iscomplex(Tr):
            flag = True
            Tr = 0
        elif Tr < 0.0 or np.isnan(Tr):
            penalty.append((Tr) / obj.Tmax)
            Tr = 0.0
            flag = True
        elif Tr > obj.Tmax or np.isinf(Tr):
            penalty.append((Tr - obj.Tmax) / obj.Tmax)
            Tr = obj.Tmax
            flag = True
        if np.iscomplex(Tt):
            flag = True
            Tt = 0
        elif Tt < 0.0 or np.isnan(Tt):
            penalty.append((Tt) / obj.Tmax)
            Tt = 0.0
            flag = True
        elif Tt > obj.Tmax or np.isinf(Tt):
            penalty.append((Tt - obj.Tmax) / obj.Tmax)
            Tt = obj.Tmax
            flag = True

        dxdt = np.array((Vr,
                         Vt / R,
                         Tr / m - Dr / m - g + Vt ** 2 / R,
                         Tt / m - Dt / m - (Vr * Vt) / R,
                         -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))
        return dxdt

    def find_close(t,x):
        """To stop propagation when the difference in height get smaller than 20 meters"""
        return abs(x[0] - Rfun(t)) - 20
    find_close.terminal = True

    sol = solve_ivp(partial(sys, Rfun=Rfun, Thetafun=Thetafun, Vrfun=Vrfun, Vtfun=Vtfun, Trfun=Trfun, Ttfun=Ttfun), [t_init, tfin], init_cond)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]

    tt = sol.t

    r = Rfun(tt)
    theta = Thetafun(tt)

    err1 = (r - y1) / obj.Htarget
    err2 = (theta - y2) / 0.9

    fitness1 = abs(simps(abs(err1), tt))
    fitness2 = abs(simps(abs(err2), tt))
    if fitness1 > fitness2:
        use = fitness1
    else:
        use = fitness2
    if penalty != []:
        pen = np.sqrt(sum(np.array(penalty) ** 2))

    if flag is True:
        x = [use, pen]
        return x
    else:
        return [use, 0.0]


####################################    P R I M I T I V E  -  S E T     ################################################

psetR = gp.PrimitiveSet("Radial", 2)
psetR.addPrimitive(operator.add, 2, name="Add")
psetR.addPrimitive(operator.sub, 2, name="Sub")
psetR.addPrimitive(operator.mul, 2, name='Mul')
psetR.addPrimitive(gpprim.TriAdd, 3)
psetR.addPrimitive(np.tanh, 1, name="Tanh")
psetR.addPrimitive(gpprim.Sqrt, 1)
psetR.addPrimitive(gpprim.Log, 1)
psetR.addPrimitive(gpprim.ModExp, 1)
psetR.addPrimitive(gpprim.Sin, 1)
psetR.addPrimitive(gpprim.Cos, 1)

for i in range(nEph):
    psetR.addEphemeralConstant("randR{}".format(i), lambda: round(random.uniform(-10, 10), 6))

psetR.renameArguments(ARG0='errR')
psetR.renameArguments(ARG1='errVr')


psetT = gp.PrimitiveSet("Tangential", 2)
psetT.addPrimitive(operator.add, 2, name="Add")
psetT.addPrimitive(operator.sub, 2, name="Sub")
psetT.addPrimitive(operator.mul, 2, name='Mul')
psetT.addPrimitive(gpprim.TriAdd, 3)
psetT.addPrimitive(np.tanh, 1, name="Tanh")
psetT.addPrimitive(gpprim.Sqrt, 1)
psetT.addPrimitive(gpprim.Log, 1)
psetT.addPrimitive(gpprim.ModExp, 1)
psetT.addPrimitive(gpprim.Sin, 1)
psetT.addPrimitive(gpprim.Cos, 1)

for i in range(nEph):
    psetT.addEphemeralConstant("randT{}".format(i), lambda: round(random.uniform(-10, 10), 3))

psetT.renameArguments(ARG0='errTheta')
psetT.renameArguments(ARG1='errVt')

################################################## TOOLBOX #############################################################

creator.create("Fitness", mods.FitnessMulti, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)
creator.create("SubIndividual", gp.PrimitiveTree)

toolbox = base.Toolbox()

toolbox.register("exprR", gp.genFull, pset=psetR, type_=psetR.ret, min_=1, max_=4)
toolbox.register("exprT", gp.genFull, pset=psetT, type_=psetT.ret, min_=1, max_=4)
toolbox.register("legR", tools.initIterate, creator.SubIndividual, toolbox.exprR)
toolbox.register("legT", tools.initIterate, creator.SubIndividual, toolbox.exprT)
toolbox.register("legs", tools.initCycle, list, [toolbox.legR, toolbox.legT], n=1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.legs)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("popx", tools.initIterate, list, initPOP1)
toolbox.register("compileR", gp.compile, pset=psetR)
toolbox.register("compileT", gp.compile, pset=psetT)
toolbox.register("evaluate", evaluate, Rfun=Rfun, Thetafun=Thetafun, Vrfun=Vrfun, Vtfun=Vtfun, Trfun=Trfun, Ttfun=Ttfun)
toolbox.register("select", mods.InclusiveTournament)
toolbox.register("mate", mods.xmate)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)
toolbox.register("mutate", mods.xmut, expr=toolbox.expr_mut, unipb=0.5, shrpb=0.25, inspb=0.15, psetR=psetR, psetT=psetT)

toolbox.decorate("mate", mods.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", mods.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

toolbox.decorate("mate", mods.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", mods.staticLimit(key=len, max_value=limit_size))

##################################### SIMULATION ##################################################


nt = 0  # simulation iteration counter

success_time = 0
success_range = 0
success_range_time = 0
pos_final_cond = []
theta_final_cond = []
n_eval = []
t_offdesign = []

xini = [obj.Re, 0, 0, 0, obj.M0]
tt = np.linspace(0, tfin, 1000)
if plot:
    #############  Propagate equations to obtain the trajectory which would be performed using the reference control values and the ISA density model
    res_RefContrIsa = solve_ivp(partial(utils.sys2GP_ISA, obj=obj, Trfun=Trfun, Ttfun=Ttfun), [0, tfin], xini, t_eval=tt)  # integrate with reference control and ISA atm model
    R_true = PchipInterpolator(res_RefContrIsa.t, res_RefContrIsa.y[0, :])
    Th_true = PchipInterpolator(res_RefContrIsa.t, res_RefContrIsa.y[1, :])
    Vr_true = PchipInterpolator(res_RefContrIsa.t, res_RefContrIsa.y[2, :])
    Vt_true = PchipInterpolator(res_RefContrIsa.t, res_RefContrIsa.y[3, :])
    m_true = PchipInterpolator(res_RefContrIsa.t, res_RefContrIsa.y[4, :])
    alt = np.linspace(0, Rref[-1], 1000)
    alt_50k = np.linspace(0, 50000, 1000)
    plt.figure(0, dpi=300)
    plt.plot(alt_50k / 1e3, utils.isa(alt_50k, 0)[1], marker='.', color='k', linewidth=3, label="ISA")
    plt.plot(alt_50k / 1e3, utils.air_density(alt_50k), 'r--', linewidth=3, label="Density simple model")
    plt.xlabel("Altitude [km]")
    plt.ylabel("Density [kg/m3]")

    plt.figure(1, dpi=300)
    plt.semilogy(alt_50k / 1e3, utils.isa(alt_50k, 0)[1], marker='.', color='k', linewidth=3, label="ISA")
    plt.semilogy(alt_50k / 1e3, utils.air_density(alt_50k), 'r--', linewidth=3, label="Density simple model")
    plt.xlabel("Altitude [km]")
    plt.ylabel("Density [kg/m3]")

    plt.figure(2, dpi=300)
    plt.plot(tref, (Rref - obj.Re)/1e3, 'r--', linewidth=3, label='Reference')
    plt.plot(res_RefContrIsa.t, (res_RefContrIsa.y[0, :] - obj.Re)/1e3, linewidth=3, color='k', label='With ISA')
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [km]")

    plt.figure(3, dpi=300)
    plt.plot(tref, np.rad2deg(Thetaref), "r--", linewidth=3, label="Reference")
    plt.plot(res_RefContrIsa.t, np.rad2deg(res_RefContrIsa.y[1, :]), linewidth=3, color='k', label="With ISA")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")


while nt < ntot:
    print("---------------------------------------------------------------------------------------------------------------- Simulation Iteration = {} ".format(nt))
    rho_newmodel = []

    def find_diff(t, x):
        '''detects when the position error is greater than 100 meters and time is greater than 10 second'''
        if t >= 10:
            v = 0
        else:
            v = 1
        return 100 - abs(x[0] - Rfun(t)) + v

    find_diff.terminal = True

    ########## start mission #####################
    res_start = solve_ivp(partial(utils.sys2GP_ISA, obj=obj, Trfun=Trfun, Ttfun=Ttfun), [0, tfin], xini, events=find_diff, t_eval=tt)  # integrate with standard thrust and ISA atm model until condition is broke

    r = copy(res_start.y[0, :])  # values up to the difference in height
    t = copy(res_start.t)

    ########### create new density model #########################

    new_alt = np.linspace(r[-1], 400000 + obj.Re, 1000)
    rho_new = utils.isa(r - obj.Re, 0)[1]  # new measured values
    rho_old = utils.air_density(new_alt - obj.Re)
    rho_newdata = np.hstack((rho_new, rho_old))
    r_comp = np.hstack((r, new_alt))
    i = 1
    to_remove = []
    while i < len(r_comp):
        if r_comp[i] <= r_comp[i - 1]:
            to_remove.append(i)
        i += 1
    r_comp_i = np.delete(r_comp, to_remove)
    rho_newdata_i = np.delete(rho_newdata, to_remove)
    rho_newmodel = PchipInterpolator(r_comp_i - obj.Re, rho_newdata_i)  # this atm model uses atm data (ISA) until the decision point and then uses standard model

    n = 1

    if plot:
        plt.figure(0)
        plt.plot(alt_50k / 1e3, rho_newmodel(alt_50k), label="Model {}".format(n))

        plt.figure(1)
        plt.semilogy(alt_50k/1e3, rho_newmodel(alt_50k), label="Model {}".format(n))



    init_cond = [res_start.y[0,:][-1], res_start.y[1,:][-1], res_start.y[2,:][-1], res_start.y[3,:][-1], res_start.y[4,:][-1]]
    t_init = res_start.t[-1]

    while t[-1] < tfin and n < 10:
        ########################### evaluates new controller ###########################
        if learning:
            size_pop = size_pop_tot - len(old_hof)  # Pop size
        else:
            size_pop = size_pop_tot
        x_ini_evaltime = init_cond
        res_evaltime = solve_ivp(partial(utils.sys2GP_ISA, obj=obj, Trfun=Trfun, Ttfun=Ttfun), [t_init, t_init + delta_eval], x_ini_evaltime)  # integrate with standard thrust and ISA to find initial condition for GP after evaluation time
        init_cond = copy([res_evaltime.y[0, :][-1], res_evaltime.y[1, :][-1], res_evaltime.y[2, :][-1], res_evaltime.y[3, :][-1], res_evaltime.y[4, :][-1]])  # intial conditions for GP
        t_init = copy(res_evaltime.t[-1])

        mutpb = copy(mutpb_start)
        cxpb = copy(cxpb_start)
        start = time()
        pop, log, hof = main(size_pop, size_gen, Mu, cxpb, mutpb)  #GP
        end = time()
        if learning:
            old_hof.update(hof[-5:], for_feasible=True)

        t_offdesign.append(end - start)
        print("Time elapsed: {}".format(t_offdesign))

        if flag_save:
            output = open("Results_2C_Density_{}_{}it_Res_{}_{}/hof_Offline.pkl".format(str_learning, str(ntot), os.path.basename(__file__), timestr), "wb")  # save of hall of fame after first GP run
            cPickle.dump(hof, output, -1)
            output.close()

        ######### here new controller is created based on new model ###############


        tev = np.linspace(t_init, tfin, 1000)
        find_diff.terminal = True

        res_newMod = solve_ivp(partial(utils.sys_rho, expr1=hof[-1][0], expr2=hof[-1][1], obj=obj, toolbox=toolbox, Rfun=Rfun,
                                Thetafun=Thetafun, Vrfun=Vrfun, Vtfun=Vtfun, Trfun=Trfun, Ttfun=Ttfun, rho_newmodel=rho_newmodel),
                        [t_init, tfin], init_cond, events=find_diff, t_eval=tev)  # check where the condition is broken with the new controller

        r = copy(res_newMod.y[0, :])
        theta = copy(res_newMod.y[1, :])
        t = copy(res_newMod.t)
        init_cond = copy([res_newMod.y[0, :][-1], res_newMod.y[1, :][-1], res_newMod.y[2, :][-1], res_newMod.y[3, :][-1], res_newMod.y[4, :][-1]])
        t_init = copy(res_newMod.t[-1])

        ############ creation of new density model ###################
        new_alt = np.linspace(r[-1], 400000 + obj.Re, 1000)
        r_isa = np.linspace(obj.Re, r[-1], 1000)
        rho_new = utils.isa(r_isa - obj.Re, 0)[1]  # new measured values
        rho_old = rho_newmodel(new_alt - obj.Re)
        rho_newdata = np.hstack((rho_new, rho_old))
        r_comp = np.hstack((r_isa, new_alt))
        i = 1
        to_remove = []
        while i < len(r_comp):
            if r_comp[i] <= r_comp[i - 1]:
                to_remove.append(i)
            i += 1
        r_comp_i = np.delete(r_comp, to_remove)
        rho_newdata_i = np.delete(rho_newdata, to_remove)
        rho_newmodel = PchipInterpolator(r_comp_i - obj.Re, rho_newdata_i)

        if plot:
            plt.figure(0)
            plt.plot(alt_50k / 1e3, rho_newmodel(alt_50k), label="Model {}".format(n+1))

            plt.figure(1)
            plt.semilogy(alt_50k/1e3, rho_newmodel(alt_50k), label="Model {}".format(n+1))

            plt.figure(2)
            plt.plot(res_newMod.t, (res_newMod.y[0, :]-obj.Re)/1e3, label="Model {}".format(n))

            plt.figure(3)
            plt.plot(res_newMod.t, np.rad2deg(res_newMod.y[1, :]), label="Model {}".format(n))


        if t_offdesign[-1] <= delta_eval:
            success_time += 1
        n += 1
    pos_final_cond.append(r[-1])
    theta_final_cond.append(theta[-1])

    if (Rref[-1] - obj.Re)*0.99 < (r[-1] - obj.Re) < (Rref[-1] - obj.Re)*1.01 and Thetaref[-1]*0.99 < theta[-1] < Thetaref[-1]*1.01:  # tolerance of 1%
        success_range += 1
    all_good = True
    for i in range(n-1):
        if t_offdesign[-1-i] > delta_eval:
            all_good = False
    if all_good is True and (Rref[-1] - obj.Re)*0.99 < (r[-1] - obj.Re) < (Rref[-1] - obj.Re)*1.01 and Thetaref[-1]*0.99 < theta[-1] < Thetaref[-1]*1.01:
        success_range_time += 1
    n_eval.append(n-1)
    nt += 1

if flag_save:
    np.save(savedata_file + "Final_pos", pos_final_cond)
    np.save(savedata_file + "Final_ang", theta_final_cond)
    np.save(savedata_file + "Success_time", success_time)
    np.save(savedata_file + "Success_range", success_range)
    np.save(savedata_file + "Success_range_time", success_range_time)
    np.save(savedata_file + "N_of_evals", n_eval)
    np.save(savedata_file + "Total_iterations", ntot)
    np.save(savedata_file + "Real_eval_times", t_offdesign)

if plot:
    plt.figure(0)
    plt.grid()
    plt.legend(loc='best')
    if flag_save:
        plt.savefig(savefig_file + "density.eps", format='eps', dpi=300)
    plt.figure(1)
    plt.grid()
    plt.legend(loc='best')
    if flag_save:
        plt.savefig(savefig_file + "density_log.eps", format='eps', dpi=300)
    plt.figure(2)
    plt.grid()
    plt.legend(loc='best')
    if flag_save:
        plt.savefig(savefig_file + "altitude.eps", format='eps', dpi=300)
    plt.figure(3)
    plt.grid()
    plt.legend(loc='best')
    if flag_save:
        plt.savefig(savefig_file + "theta.eps", format='eps', dpi=300)
    plt.show(block=True)

it = len(t_offdesign)
print("Max {}, Min {}, Median {}".format(np.max(t_offdesign), np.min(t_offdesign), statistics.median(t_offdesign)))
print("Total it: {} in {} mission eval".format(it, ntot))

print("Success time {}, Success range {}, Success total {}".format((success_time/it)*100, success_range/ntot*100, success_range_time/ntot*100))

if plot_comparison:
    plt.figure(num=10, figsize=(14,8))
    plt.plot(t_offdesign, '.')
    plt.xlabel("GP Evaluations")
    plt.ylabel("Time [s]")
    plt.ylim(0, 100)
    plt.grid()
    plt.title("3rd Scenario - {}".format(str_learning))
    if flag_save:
        plt.savefig(savefig_file + "comparison.eps", format='eps', dpi=300)

    plt.show(block=True)



