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
This script performs the optimization of the GP control law obtained offline by testing it on 500 different disturbance
scenarios contained in the dataset training_points.npy. The results of the successful optimization are stored in a
dataset to be used to train the NN. The dataset has the following shape:

[[t1, x11', ..., x1s', err_x11', ..., err_x1s', c11, ..., c1n],
 [t2, x21', ..., x2s', err_x21', ..., err_x2s', c21, ..., c2n],
 ...,
 [tm, xm1', ..., xms', err_xm1', ..., err_xms', cm1, ..., cmn]]

where m denotes the different optimizations performed, s is the number of the states parameters, and n refers to the
optimized parameters typical of the considered GP equation. In this case, each component is a column vector containing
500 points of the performed trajectory, e.g. x11' contains the trajectory of the state x1 obtained using the optimized
values of the GP control laws from c11 to c1n. The same explanation is valid for the tracking errors, which are measured
as the differences between the obtained trajectory and the reference one.

"""

import random
from deap import gp, creator, base
import _pickle as cPickle
from scipy.interpolate import PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator
from copy import deepcopy
from scipy.optimize import minimize, basinhopping
import multiprocessing as mp
import scipy.io as sio
import TreeTunerUtils as utils
import sys
import os
models_path = os.path.join(os.path.dirname( __file__ ), '..', 'FESTIP_Models')
gpfun_path = os.path.join(os.path.dirname( __file__ ), '..', 'GP_Functions')
data_path = os.path.join(os.path.dirname( __file__ ), '..', 'Datasets')
gplaw_path = os.path.join(os.path.dirname( __file__ ), '..', '1_OfflineCreationGPLaw/ResultsBIOMA2020')
sys.path.append(models_path)
sys.path.append(gpfun_path)
sys.path.append(data_path)
sys.path.append(gplaw_path)
import models_FESTIP as mods
import GP_PrimitiveSet as gpprim
import GP_Functions as funs
import time
from tqdm import *
from scipy.integrate import solve_ivp


################ tunable parameters #####################
method = 'BFGS'  # optimization method to use
hofnum = 4  # select which control law to optimize


nEph = 2
pset = gp.PrimitiveSet("Main", 4)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(operator.mul, 2, name='Mul')
pset.addPrimitive(gpprim.TriAdd, 3)
pset.addPrimitive(np.tanh, 1, name="Tanh")
pset.addPrimitive(gpprim.Sqrt, 1)
pset.addPrimitive(gpprim.Log, 1)
pset.addPrimitive(gpprim.ModExp, 1)
pset.addPrimitive(gpprim.Sin, 1)
pset.addPrimitive(gpprim.Cos, 1)

for i in range(nEph):
    pset.addEphemeralConstant("rand{}".format(i), lambda: round(random.uniform(-10, 10), 6))

pset.addTerminal('w')
pset.renameArguments(ARG0='errV')
pset.renameArguments(ARG1='errChi')
pset.renameArguments(ARG2='errGamma')
pset.renameArguments(ARG3='errH')

################################################## TOOLBOX #############################################################

creator.create("Fitness", funs.FitnessMulti, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)
creator.create("SubIndividual", gp.PrimitiveTree)
toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)


# Retrieve the GP tree through pickle
objects = []
with (open(gplaw_path + '/IGP_hof_{}.pkl'.format(hofnum), "rb")) as openfile:
    while True:
        try:
            objects.append(cPickle.load(openfile))
        except EOFError:
            break


bestFuns = deepcopy(objects[0][-1])

print("Control law of first control parameter: ")
print(bestFuns[0])
print("\n")
print("Control law of second control parameter: ")
print(bestFuns[1])
print("\n")

fAlpha = gp.compile(bestFuns[0], pset=pset)
fDelta = gp.compile(bestFuns[1], pset=pset)

indexes = []
values = []

pset.terminals[pset.ret][-1].value = 1  # set initial value of weights to 1
weight = deepcopy(pset.terminals[pset.ret][-1])  # store weight terminal
mul_fun = deepcopy(pset.primitives[pset.ret][2])  # store mul function primitive


###################  This part add the multiplication between the gp function inputs and weights #######################

for i in range(len(bestFuns)):
    j = 0
    l = len(bestFuns[i])
    stop = False
    while not stop:
        if type(bestFuns[i][j]) == gp.Terminal and bestFuns[i][j].name[0] == "A":
            bestFuns[i].insert(j, mul_fun)
            bestFuns[i].insert(j+1, weight)
            j = j + 3
            l += 2
        else:
            j += 1
        if j == l:
            stop = True

########################################################################################################################

for i in range(len(bestFuns)):
    for j in range(len(bestFuns[i])):
        if type(bestFuns[i][j]) == gp.Terminal and bestFuns[i][j].name == "w":
            values.append(bestFuns[i][j].value)
for i in range(len(bestFuns)):
    for j in range(len(bestFuns[i])):
        if type(bestFuns[i][j]) == gp.rand0 or type(bestFuns[i][j]) == gp.rand1:
            values.append(bestFuns[i][j].value)

values = np.array(values)

print("Modified control law of first control parameter: ")
print(bestFuns[0])
print("\n")
print("Modified control law of second control parameter: ")
print(bestFuns[1])
print("\n")
print("Parameters to be optimized: ", values)

###################### LOAD DATA #################################à
obj = mods.Spaceplane()

training_points = np.load(data_path + "/training_points.npy")
n_samples = len(training_points)

ref_traj = sio.loadmat(models_path + "/reference_trajectory.mat")
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

fAlpha = gp.compile(bestFuns[0], pset=pset)
fDelta = gp.compile(bestFuns[1], pset=pset)

def opti(n):
    x_start = [vref[0], chiref[0], gammaref[0], tetaref[0], lamref[0], href[0], mref[0]]
    final_cond = [vref[-1], chiref[-1], gammaref[-1], tetaref[-1], lamref[-1], href[-1]]
    height_start = training_points[n][0]
    deltaH = training_points[n][2]
    v_wind = training_points[n][1]

    def find_height(t, x, *args):
        return height_start - x[5]

    find_height.terminal = True
    ####### propagation to find where the gust zone starts ###############
    tev = np.linspace(2, tfin, 200)
    init = solve_ivp(mods.sys_init, [2, tfin], x_start, events=find_height, method='RK45', t_eval=tev,
                     args=(cl, cd, cm, spimpv, presv, obj, alfafun, deltafun))

    change_time = init.t[-1]
    x_ini_h = [init.y[0, -1], init.y[1, -1], init.y[2, -1], init.y[3, -1], init.y[4, -1], init.y[5, -1], init.y[6, -1]]

    if method == 'basinhopping':
        best = basinhopping(utils.tree_eval_opt, values, niter=30, disp=False,
                        minimizer_kwargs={'args': (height_start, deltaH, v_wind, bestFuns, pset, obj, vfun, chifun,
                                                   gammafun, hfun, alfafun, deltafun, x_start, cl, cd, cm, spimpv,
                                                   presv, tfin, final_cond),
                        'method': 'BFGS'})
    else:
        best = minimize(utils.tree_eval_opt, values,
                            args = (height_start, deltaH, v_wind, bestFuns, pset, obj, vfun, chifun, gammafun, hfun,
                                    alfafun, deltafun, cl, cd, cm, spimpv, presv, tfin, final_cond, change_time,
                                    x_ini_h),
                        method=method)

    success, res, t_int = utils.test_param_opt(best.x, height_start, deltaH, v_wind, bestFuns, pset, obj, vfun,
                                        chifun, gammafun, hfun, alfafun, deltafun, cl, cd, cm,
                                        spimpv, presv, tfin, href, final_cond, change_time, x_ini_h)
    if success is True:
        print("{} - Wind Speed = {} m/s, Start altitude = {} m, Gust size = {} m".format(n, round(v_wind, 2), round(height_start, 2), round(deltaH, 2)))
        best_val = list(best.x)
        data = np.column_stack((t_int.reshape(len(t_int),1),
                           res[:,0].reshape(len(t_int),1),
                           res[:,1].reshape(len(t_int),1),
                           res[:,2].reshape(len(t_int),1),
                           res[:,3].reshape(len(t_int),1),
                           res[:,4].reshape(len(t_int), 1),
                           res[:,5].reshape(len(t_int), 1),
                           (vfun(t_int)-res[:,0]).reshape(len(t_int),1),
                           (chifun(t_int)-res[:,1]).reshape(len(t_int),1),
                           (gammafun(t_int)-res[:,2]).reshape(len(t_int),1),
                           (tetafun(t_int)-res[:,3]).reshape(len(t_int),1),
                           (lamfun(t_int)-res[:,4]).reshape(len(t_int),1),
                           (hfun(t_int)-res[:,5]).reshape(len(t_int),1)))
        for vv in best_val:
            data = np.column_stack((data, np.ones((len(t_int), 1)) * vv))
        return [data, n]
    else:
        print("FAIL - Wind Speed = {} m/s, Start altitude = {} m, Gust size = {} m".format(round(v_wind, 2), round(height_start, 2), round(deltaH, 2)))

if __name__ == "__main__":
    succ = 0
    fail = 0
    nbCPU = mp.cpu_count()
    pool = mp.Pool(nbCPU)
    start = time.time()
    opt_res = []
    with tqdm(total=n_samples) as pbar:
        for i, res in enumerate(pool.imap_unordered(opti, range(n_samples))):
            pbar.update()
            opt_res.append(res)

    end = time.time()
    pool.close()
    pool.join()
    print("Optimization on {} points performed in {} s".format(n_samples, end-start))
    dataset = []
    success_indexes = []
    for i in range(len(opt_res)):
        try:
            success_indexes.append(opt_res[i][1])
        except:
            continue
        if hasattr(opt_res[i][0], '__len__') and len(dataset) == 0:
            dataset = opt_res[i][0]
        elif hasattr(opt_res[i][0], '__len__') and len(dataset) != 0:
            dataset = np.vstack((dataset, opt_res[i][0]))


    np.save("dataset_forNN_{}samples_1percent_{}_hof{}.npy".format(n_samples, method, hofnum), dataset)

    print("{} success {}/{}".format(method, len(success_indexes), n_samples))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(training_points)):
        if i in success_indexes:
            c = 'g'
            label = 'Success'
        else:
            c = 'r'
            label = 'Failure'

        ax.scatter(training_points[i,0]/1000, training_points[i,1], training_points[i,2]/1000, c=c, marker='.', label=label)

    ax.set_xlabel('Height start [km]')
    ax.set_ylabel('wind speed [m/s]')
    ax.set_zlabel('Gust range [km]')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()



