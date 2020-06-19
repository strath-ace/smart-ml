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
This script simulates the Intelligent Control of a Goddard Rocket with 2 controls where a wind gust of a random constant 
intensity acting in a random altitude range is applied.
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
import _pickle as cPickle
import DEAP_Mods as mods
import GustScenario_utils as utils
import pygraphviz as pgv
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
plot_tree = False  # set to True to save the plots of the tree of the found control laws
plot_comparison = False  # set to True to save the plot of the comparison of the evaluation time of the different simulations

ntot = 1000
if ntot > 1 and (plot is True or plot_tree is True):
    plot = False
    plot_tree = False
    print("It is advised to plot only one iteration since to plot multiple iterations on the same plots will be very messy and unclear")

nEph = 2  # number of ephemeral constants to use
limit_height = 8  # Max height (complexity) of the controller law
limit_size = 30  # Max size (complexity) of the controller law

fit_tol = 1.1
delta_eval = 10 # time slots used for GP law evaluation
size_pop_tot = 250 # total population size
if learning:
    old_hof = mods.HallOfFame(int(size_pop_tot/1.5))

size_gen = 150  # Maximum generation number

Mu = int(size_pop_tot)
mutpb_start = 0.7  # mutation rate
cxpb_start = 0.2  # Crossover rate
mutpb = copy(mutpb_start)
cxpb = copy(cxpb_start)

nbCPU = multiprocessing.cpu_count()  # select the available CPU's. Set to 1 to not use multiprocessing


######################### SAVE DATA #############################################
timestr = strftime("%Y%m%d-%H%M%S")

if flag_save:
    os.makedirs("Results_Gust_2C_{}_{}it_Res_{}_{}".format(str_learning, str(ntot), os.path.basename(__file__), timestr))
    savefig_file = "Results_Gust_2C_{}_{}it_Res_{}_{}/Plot_".format(str_learning, str(ntot), os.path.basename(__file__), timestr)
    savedata_file = "Results_Gust_2C_{}_{}it_Res_{}_{}/".format(str_learning, str(ntot), os.path.basename(__file__), timestr)

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
        self.a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
        self.a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
        self.hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
        self.h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
        self.tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
        self.pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
        self.tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
        self.tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s

    @staticmethod
    def air_density(h):
        global flag
        beta = 1 / 8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        try:
            return rho0 * np.exp(-beta * h)
        except RuntimeWarning:
            flag = True
            return rho0 * np.exp(-beta * obj.Rtarget)


Nstates = 5
Ncontrols = 2
obj = Rocket()

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


################################# M A I N ###############################################


def initPOP1():
    global old_hof
    # this function outputs the first n individuals of the hall of fame of the first GP run
    res = old_hof.shuffle()
    # for i in range(10):
    #   res.append(hof[0])
    return res


def main(size_pop, size_gen, Mu, cxpb, mutpb):
    global flag

    flag = False

    pool = multiprocessing.Pool(nbCPU)

    toolbox.register("map", pool.map)
    old_entropy = 0
    best_pop = []
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

    print("INITIAL POP SIZE: %d" % size_pop_tot)
    print("GEN SIZE: %d" % size_gen)
    print("\n")

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", mods.Min)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   ###################################

    pop, log = mods.eaMuPlusLambdaTol(best_pop, toolbox, Mu, cxpb, mutpb, size_gen, fit_tol, limit_size, stats=mstats, halloffame=hof,
                                 verbose=True)
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
        global penalty, flag, v_wind, change_time, height_start, delta
        # State Variables
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        if height_start < R < height_start + delta:
            Vt = Vt - v_wind * np.cos(theta)
            Vr = Vr - v_wind * np.sin(theta)

        if R < obj.Re - 0.5 or np.isnan(R):
            penalty.append((R - obj.Re) / obj.Htarget)
            R = obj.Re
            flag = True

        if m < obj.M0 - obj.Mp or np.isnan(m):
            penalty.append((m - (obj.M0 - obj.Mp)) / obj.M0)
            m = obj.M0 - obj.Mp
            flag = True

        er = Rfun(t) - R
        et = Thetafun(t) - theta
        evr = Vrfun(t) - Vr
        evt = Vtfun(t) - Vt

        rho = obj.air_density(R - obj.Re)
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


    sol = solve_ivp(partial(sys, Rfun=Rfun, Thetafun=Thetafun, Vrfun=Vrfun, Vtfun=Vtfun, Trfun=Trfun, Ttfun=Ttfun), [change_time+delta_eval, tfin], x_ini_h)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]

    tt = sol.t

    if tt[-1] < tfin:
        flag = True
        penalty.append((tt[-1] - tfin) / tfin)

    r = Rfun(tt)
    theta = Thetafun(tt)

    err1 = (r - y1) / obj.Htarget
    err2 = (theta - y2) / 0.9

    fitness1 = simps(abs(err1), tt)
    fitness2 = simps(abs(err2), tt)
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
    psetT.addEphemeralConstant("randT{}".format(i), lambda: round(random.uniform(-10, 10), 6))

psetT.renameArguments(ARG0='errTheta')
psetT.renameArguments(ARG1='errVt')

################################################## TOOLBOX #############################################################

creator.create("Fitness", mods.FitnessMulti,  weights=(-1.0, -1.0))
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


###########################################  SIMULATION   #################################################


success_range_time = 0
success_time = 0
success_range = 0
nt = 0
stats = []
success = 0
r_diff = []
theta_diff = []

stats.append(["Wind speed", "Start height", "Size gust range", "Delta eval", "Real time eval"])
while nt < ntot:
    ####  RANDOM VARIATIONS DEFINITION ####
    height_start = obj.Re + random.uniform(1000, 40000)
    delta = random.uniform(10000, 15000)
    v_wind = random.uniform(0, 80)

    if learning:
        size_pop = size_pop_tot - len(old_hof)  # Pop size
    else:
        size_pop = size_pop_tot

    mutpb = copy(mutpb_start)
    cxpb = copy(cxpb_start)
    print(" ---------------Iter={}, V wind={}, Gust Height Start={}, Size Gust Zone={} -----------------".format(nt, round(v_wind,2), round(height_start-obj.Re,2), round(delta,2)))
    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]
    idx = 0
    def find_height(t, x):
        return height_start - x[0]

    find_height.terminal = True
    ####### integration to find where the gust zone starts ###############
    init = solve_ivp(partial(utils.sys_init, obj=obj, Trfun=Trfun, Ttfun=Ttfun), [0, tfin], x_ini, events=find_height)
    change_time = init.t[-1]
    ######## second integration to find out initial conditions after evaluation, assuming delta_eval as evaluation time #########
    x_ini_1 = [init.y[0, :][-1], init.y[1, :][-1], init.y[2, :][-1], init.y[3, :][-1], init.y[4, :][-1]]
    init2 = solve_ivp(partial(utils.sys_ifnoC, height_start=height_start, delta=delta, v_wind=v_wind, Trfun=Trfun, Ttfun=Ttfun, obj=obj), [change_time, change_time+delta_eval], x_ini_1)

    x_ini_h = [init2.y[0, :][-1], init2.y[1, :][-1], init2.y[2, :][-1], init2.y[3, :][-1], init2.y[4, :][-1]]  #  initial conditions to be used by GP, since it has to find a law that starts after the evaluation time, ideally

    start = time()
    pop, log, hof = main(size_pop, size_gen, Mu, cxpb, mutpb)
    end = time()
    if learning:
        old_hof.update(hof[-5:], for_feasible=True)
    t_offdesign = end - start
    print("Time elapsed: {}".format(t_offdesign))
    stats.append([v_wind, height_start-obj.Re, delta, delta_eval, t_offdesign])
    if flag_save:
        output = open("Results_Gust_2C_{}_{}it_Res_{}_{}/hof_Offline{}.pkl".format(str_learning, str(ntot), os.path.basename(__file__), timestr, nt), "wb")  # save of hall of fame after first GP run
        cPickle.dump(hof, output, -1)
        output.close()
    print(hof[-1][0])
    print(hof[-1][1])
    if plot_tree:
        plt.figure(7)
        expr1 = hof[-1][0]
        nodes1, edges1, labels1 = gp.graph(expr1)
        g1 = pgv.AGraph()
        g1.add_nodes_from(nodes1)
        g1.add_edges_from(edges1)
        g1.layout(prog="dot")
        for i in nodes1:
            n = g1.get_node(i)
            n.attr["label"] = labels1[i]
        g1.draw("tree1.png")
        if flag_save:
            plt.savefig(savefig_file + "Tr_tree.eps", format='eps', dpi=300)
        image1 = plt.imread('tree1.png')
        fig1, ax1 = plt.subplots()
        im1 = ax1.imshow(image1)
        ax1.axis('off')

        plt.figure(8)
        expr2 = hof[-1][1]
        nodes2, edges2, labels2 = gp.graph(expr2)
        g2 = pgv.AGraph()
        g2.add_nodes_from(nodes2)
        g2.add_edges_from(edges2)
        g2.layout(prog="dot")
        for i in nodes2:
            n = g2.get_node(i)
            n.attr["label"] = labels2[i]
        g2.draw("tree2.png")
        image2 = plt.imread('tree2.png')
        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(image2)
        ax2.axis('off')
        if flag_save:
            plt.savefig(savefig_file + "Tt_tree.eps", format='eps', dpi=300)


    #########for plot and to find initial condition for propagation#########
    x_ini = [obj.Re, 0, 0, 0, obj.M0]
    tev = np.linspace(0, tfin, 1000)
    teval_val_p = solve_ivp(partial(utils.sys_ifnoC, height_start=height_start, delta=delta, v_wind=v_wind, Trfun=Trfun, Ttfun=Ttfun, obj=obj), [0, tfin], x_ini, t_eval=tev) # only used for plot

    r_eval_p = teval_val_p.y[0, :]
    th_eval_p = teval_val_p.y[1, :]
    vr_eval_p = teval_val_p.y[2, :]
    vt_eval_p = teval_val_p.y[3, :]
    m_eval_p = teval_val_p.y[4, :]
    t_eval_p = teval_val_p.t
    for i in range(len(t_eval_p)):
        if t_eval_p[i] >= change_time + delta_eval:
            index = i
            break
    #####  propagation  #####

    tev = np.linspace(change_time + delta_eval, tfin, 1000)
    solgp = solve_ivp(partial(utils.sys2GP, expr1=hof[-1][0], expr2=hof[-1][1], v_wind=v_wind, height_start=height_start,
                              delta=delta, toolbox=toolbox, obj=obj, Rfun=Rfun, Thetafun=Thetafun, Vrfun=Vrfun,
                              Vtfun=Vtfun, Trfun=Trfun, Ttfun=Ttfun), [change_time + delta_eval, tfin], x_ini_h, t_eval=tev)  # used for next integration

    rout = solgp.y[0, :]
    thetaout = solgp.y[1, :]
    vrout = solgp.y[2, :]
    vtout = solgp.y[3, :]
    mout = solgp.y[4, :]
    ttgp = solgp.t

    if t_offdesign < delta_eval:
        success_time += 1
    if (Rref[-1]-obj.Re) * 0.99 < (rout[-1]-obj.Re) < (Rref[-1]-obj.Re) * 1.01 and Thetaref[-1] * 0.99 < thetaout[-1] < Thetaref[-1] * 1.01:  # tolerance of 1%
        success_range += 1
    if t_offdesign < delta_eval and (Rref[-1]-obj.Re) * 0.99 < (rout[-1]-obj.Re) < (Rref[-1]-obj.Re) * 1.01 and Thetaref[-1] * 0.99 < thetaout[-1] < Thetaref[-1] * 1.01:
        success_range_time += 1
    r_diff.append(abs(Rfun(tfin) - rout[-1]))
    theta_diff.append(abs(Thetafun(tfin) - thetaout[-1]))

    if flag_save:
        np.save(savedata_file + "{}_r_out".format(nt), rout)
        np.save(savedata_file + "{}_th_out".format(nt), thetaout)
        np.save(savedata_file + "{}_vr_out".format(nt), vrout)
        np.save(savedata_file + "{}_vt_out".format(nt), vtout)
        np.save(savedata_file + "{}_m_out".format(nt), mout)
        np.save(savedata_file + "{}_t_out".format(nt), ttgp)

    if plot:
        plt.ion()
        plt.figure(2)
        plt.xlabel("time [s]")
        plt.ylabel("Altitude [km]")
        plt.plot(t_eval_p, (r_eval_p - obj.Re) / 1e3, color='C0', linewidth=2)
        plt.plot(ttgp, (rout - obj.Re) / 1e3, color='C2', linewidth=2, label="V wind={} m/s, Start gust={} m, End gust={} m".format(round(v_wind, 2), round(height_start-obj.Re,2), round(height_start+delta-obj.Re)))
        plt.axhline((height_start-obj.Re)/1e3, color='k')
        plt.axhline((height_start + delta - obj.Re)/1e3, color='k')
        plt.legend(loc='best')

        plt.figure(3)
        plt.plot(t_eval_p, vt_eval_p, color='C0', linewidth=2)
        plt.plot(ttgp, vtout, color='C2', linewidth=2, label="V wind={} m/s, Start gust={} m, End gust={} m".format(round(v_wind, 2), round(height_start-obj.Re,2), round(height_start+delta-obj.Re)))
        plt.xlabel("time [s]")
        plt.ylabel("Tangential Velocity [m/s]")
        plt.legend(loc='best')

        plt.figure(4)
        plt.axhline(obj.M0 - obj.Mp, 0, ttgp[-1], color='r')
        plt.plot(t_eval_p, m_eval_p, color='C0', linewidth=2)
        plt.plot(ttgp, mout, color='C2', linewidth=2, label="V wind={} m/s, Start gust={} m, End gust={} m".format(round(v_wind, 2), round(height_start-obj.Re,2), round(height_start+delta-obj.Re)))
        plt.xlabel("time [s]")
        plt.ylabel("Mass [kg]")
        plt.legend(loc='best')

        plt.figure(5)
        plt.plot(t_eval_p, vr_eval_p, color='C0', linewidth=2)
        plt.plot(ttgp, vrout, color='C2', linewidth=2, label="V wind={} m/s, Start gust={} m, End gust={} m".format(round(v_wind, 2), round(height_start-obj.Re,2), round(height_start+delta-obj.Re)))
        plt.xlabel("time [s]")
        plt.ylabel("Radial Velocity [m/s]")
        plt.legend(loc='best')

        plt.figure(6)
        plt.plot(t_eval_p, np.rad2deg(th_eval_p), color='C0', linewidth=2)
        plt.plot(ttgp, np.rad2deg(thetaout), color='C2', linewidth=2, label="V wind={} m/s, Start gust={} m, End gust={} m".format(round(v_wind, 2), round(height_start-obj.Re,2), round(height_start+delta-obj.Re)))
        plt.xlabel("time [s]")
        plt.ylabel("Angle [deg]")
        plt.legend(loc='best')

    nt += 1

if flag_save:
    np.save(savedata_file + "Position_diff", r_diff)
    np.save(savedata_file + "Theta_diff", theta_diff)
    np.save(savedata_file + "Success_Time", success_time)
    np.save(savedata_file + "Success_range", success_range)
    np.save(savedata_file + "Success_range_time", success_range_time)
    np.save(savedata_file + "Stats_evals", stats)
    np.save(savedata_file + "Num_iterations", ntot)

print("Success time: {}%".format(round(success_time/ntot*100, 2)))
print("Success range : {}%".format(round(success_range/ntot*100, 2)))
print("Success total: {}%".format(round(success_range_time/ntot*100, 2)))
t = []
for i in range(len(stats)):
    t.append(stats[i][-1])
del t[0]
res = list(map(float, t))
print("Evaluation time - Min: {}, Max: {}, Median: {}".format(np.min(res), np.max(res), statistics.median(res)))

if plot:
    plt.ion()
    plt.figure(2)
    plt.plot(tref, (Rref - obj.Re) / 1e3, 'r--', linewidth=3, label="SET POINT")
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [km]")
    plt.legend(loc='best')
    plt.grid()
    if flag_save:
        plt.savefig(savefig_file + "Altitude.svg", format='svg', dpi=1200)

    plt.figure(3)
    plt.plot(tref, Vtref, 'r--', label="SET POINT")
    plt.xlabel("time [s]")
    plt.ylabel("Tangential Velocity [m/s]")
    plt.legend(loc='best')
    plt.grid()
    if flag_save:
        plt.savefig(savefig_file + "Vt.svg", format='svg', dpi=1200)

    plt.figure(4)
    plt.plot(tref, mref, 'r--', label="SET POINT")
    plt.axhline(obj.M0 - obj.Mp, 0, tfin, color='r')
    plt.xlabel("time [s]")
    plt.ylabel("Mass [kg]")
    plt.legend(loc='best')
    plt.grid()
    if flag_save:
        plt.savefig(savefig_file + "Mass.svg", format='svg', dpi=1200)

    plt.figure(5)
    plt.plot(tref, Vrref, 'r--', label="SET POINT")
    plt.xlabel("time [s]")
    plt.ylabel("Radial Velocity [m/s]")
    plt.legend(loc='best')
    plt.grid()
    if flag_save:
        plt.savefig(savefig_file + "Vr.svg", format='svg', dpi=1200)

    plt.figure(6)
    plt.plot(tref, np.rad2deg(Thetaref), 'r--', linewidth=3, label="SET POINT")
    plt.xlabel("time [s]")
    plt.ylabel("Angle [deg]")
    plt.legend(loc='best')
    plt.grid()
    if flag_save:
        plt.savefig(savefig_file + "Theta.svg", format='svg', dpi=1200)
    plt.show(block=True)


if plot_comparison and ntot > 1:
    plt.figure(num=7, figsize=(14, 8))
    plt.plot(res, '.')
    plt.xlabel("GP Evaluations")
    plt.ylabel("Time [s]")
    plt.ylim(0, 100)
    plt.title("2nd Scenario - {}".format(str_learning))
    plt.grid()
    if flag_save:
        plt.savefig(savefig_file + "comparison.eps", format='eps', dpi=300)
    plt.show(block=True)










