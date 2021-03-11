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
This script performs the test of the proposed IC approach. The NN can be trained from scratches or the model can be
loaded. Then it is tested on the control task.
"""

import tensorflow as tf
import random
from deap import gp, base, creator
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import numpy as np
from scipy.integrate import solve_ivp
from tensorflow.keras import regularizers
import operator
from copy import deepcopy
import _pickle as cPickle
from tensorflow.keras import Sequential
import scipy.io as sio
from sklearn import preprocessing
import sys
import os
models_path = os.path.join(os.path.dirname( __file__ ), '../../FESTIP_Models')
gpfun_path = os.path.join(os.path.dirname( __file__ ), '../../../../IGP')
data_path = os.path.join(os.path.dirname( __file__ ), '..', 'Datasets')
gplaw_path = os.path.join(os.path.dirname( __file__ ), '..', '1_OfflineCreationGPLaw')
sys.path.append(models_path)
sys.path.append(gpfun_path)
sys.path.append(data_path)
sys.path.append(gplaw_path)
import GP_PrimitiveSet as gpprim
import IGP_Functions as funs
import models_FESTIP as mods

########################### Tunable parameters  #################################à
hofnum = 4  # select the GP law used to produce the training dataset
method = 'BFGS'  # choose between NM or BFGS
save = False  # set to True if you want to train the model from scratches
config = '30'  # select between '30', '25x25', '30x30'
plot_modelStats = False  # set to True to plot the training statistics
run = 0  # index used to differentiate multiple runs
plot_trajectory = False  #  to plot the performed trajectory when the NN succeeds and the GP fails


#######################################################################################
pset = gp.PrimitiveSet("Main", 4)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(operator.mul, 2, name='Mul')
pset.addPrimitive(gpprim.TriAdd, 3)
pset.addPrimitive(np.tanh, 1, name="Tanh")
pset.addPrimitive(gpprim.ModSqrt, 1, name='Sqrt')
pset.addPrimitive(gpprim.ModLog, 1, name='Log')
pset.addPrimitive(gpprim.ModExp, 1)
pset.addPrimitive(np.sin, 1, name='Sin')
pset.addPrimitive(np.cos, 1, name='Cos')
for i in range(2):
    pset.addEphemeralConstant("rand{}".format(i), lambda: round(random.uniform(-10, 10), 4))
pset.addTerminal('w')
pset.renameArguments(ARG0='errV')
pset.renameArguments(ARG1='errChi')
pset.renameArguments(ARG2='errGamma')
pset.renameArguments(ARG3='errH')
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)
creator.create("SubIndividual", gp.PrimitiveTree)
toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)

########################################### LOAD DATA ###################################################
obj = mods.Spaceplane()

'''Retrieve the GP tree through pickle'''
objects = []
with (open(gplaw_path + '/IGP_hof_{}.pkl'.format(hofnum), "rb")) as openfile:
    while True:
        try:
            objects.append(cPickle.load(openfile))
        except EOFError:
            break

bestFuns = deepcopy(objects[0][-1])

'''indexes = []
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
print("Parameters to be optimized: ", values)'''

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

n_samples = 500
dataset = np.load(data_path + "/dataset_forNN_{}samplesTEST_1percent_{}_hof{}.npy".format(n_samples, method, hofnum), allow_pickle=True)
np.random.shuffle(dataset)

datax = preprocessing.normalize(dataset[:, 0:13])
datay = dataset[:, 13:]


if save is True:
    ###################### BUILD MODEL ##################################
    if config == '30':
        model = Sequential([
            tf.keras.layers.Dense(30, activation='relu', input_shape=(13,), kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(dataset.shape[1]-13, activation='linear')])
    elif config == '25x25':
        model = Sequential([
            tf.keras.layers.Dense(25, activation='relu', input_shape=(13,), kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(dataset.shape[1] - 13, activation='linear')])
    else:
        model = Sequential([
            tf.keras.layers.Dense(30, activation='relu', input_shape=(13,), kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(dataset.shape[1] - 13, activation='linear')])

    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.summary()
    history = model.fit(datax, datay, epochs=50, validation_split=0.15)
    if plot_modelStats:
        plt.figure()
        plt.plot(history.history['loss'], label="Loss")
        plt.plot(history.history['val_loss'], linestyle='-', label='Validation Loss')
        plt.legend(loc='best')
        plt.show(block=True)
    model.save("model_NN_2Controls_{}_{}samples_{}_hof{}.h5".format(config, n_samples, method, hofnum))
else:
    model = tf.keras.models.load_model('model_NN_2Controls_{}_{}samples_{}_hof{}.h5'.format(config, n_samples, method, hofnum))

###################  PREDICTION WITH NEW VALUES #################################


expr1 = bestFuns[0]
expr2 = bestFuns[1]

n = 0
nn_sc = 0
gp_sc = 0
failure = 0
data_fail = []
data_nn = []
data_gp = []
diff = []
data_nn_gp = []

test_points = np.load(data_path + "/TestSetNN.npy")
nn_success = False
while n < len(test_points) and nn_success is False:
    rand_p = random.uniform(-1, 1)
    rand_t = random.uniform(-1, 1)
    rand_l = random.uniform(-1, 1)
    rand_d = random.uniform(-1, 1)
    height_start = test_points[n][0]
    deltaH = test_points[n][2]
    v_wind = test_points[n][1]

    x_ini = [vref[0], chiref[0], gammaref[0], tetaref[0], lamref[0], href[0], mref[0]]
    def find_height(t, x, *args):
        return height_start - x[5]

    find_height.terminal = True
    ####### propagation to find where the gust zone starts ###############
    tev = np.linspace(2, tfin, 200)
    init = solve_ivp(mods.sys_init, [2, tfin], x_ini, events=find_height, method='RK45', t_eval=tev,
                     args=(cl, cd, cm, spimpv, presv, obj, alfafun, deltafun))

    change_time = init.t[-1]

    x_ini_h = [init.y[0, -1], init.y[1, -1], init.y[2, -1], init.y[3, -1], init.y[4, -1], init.y[5, -1], init.y[6, -1]]
    t_eval = np.linspace(change_time, tfin, 100)

    ############################### Control using only GP law non optimized ##########################################
    res_stand = solve_ivp(mods.sys2GP_NN, [change_time, tfin], x_ini_h, method='RK23', t_eval=t_eval,
                          args=(expr1, expr2, height_start, deltaH, v_wind, False, [], obj, vfun, chifun, gammafun,
                                tetafun, lamfun, hfun, alfafun, deltafun, pset, cl, cd, cm, presv, spimpv))

    v_stand = res_stand.y[0, :]
    chi_stand = res_stand.y[1, :]
    gamma_stand = res_stand.y[2, :]
    teta_stand = res_stand.y[3, :]
    lam_stand = res_stand.y[4, :]
    h_stand = res_stand.y[5, :]
    m_stand = res_stand.y[6, :]
    t_stand = res_stand.t

    v_ass_stand, chi_ass_stand = mods.vass(res_stand.y[:, -1], obj.omega)

    if h_stand[-1] < 1.0:
        h_stand[-1] = 1.0

    v_orbit_stand = np.sqrt(obj.GMe / (obj.Re + h_stand[-1]))

    if np.cos(obj.incl) / np.cos(lam_stand[-1]) > 1:
        chi_orbit_stand = np.pi
    else:
        if np.cos(obj.incl) / np.cos(lam_stand[-1]) < - 1:
            chi_orbit_stand = 0.0
        else:
            chi_orbit_stand = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam_stand[-1]))

    ######################## Control using always the NN to optimized the GP law #######################################
    res2 = solve_ivp(mods.sys2GP_NN, [change_time, tfin], x_ini_h, method='RK23', t_eval=t_eval,
                          args=(expr1, expr2, height_start, deltaH, v_wind, True, model, obj, vfun, chifun, gammafun,
                                tetafun, lamfun, hfun, alfafun, deltafun, pset, cl, cd, cm, presv, spimpv))

    v_cont = res2.y[0, :]
    chi_cont = res2.y[1, :]
    gamma_cont = res2.y[2, :]
    teta_cont = res2.y[3, :]
    lam_cont = res2.y[4, :]
    h_cont = res2.y[5, :]
    m_cont = res2.y[6, :]
    t_cont = res2.t

    v_ass_cont, chi_ass_cont = mods.vass(res2.y[:, -1], obj.omega)

    if h_cont[-1] < 1.0:
        h_cont[-1] = 1.0

    v_orbit_cont = np.sqrt(obj.GMe / (obj.Re + h_cont[-1]))

    if np.cos(obj.incl) / np.cos(lam_cont[-1]) > 1:
        chi_orbit_cont = np.pi
    else:
        if np.cos(obj.incl) / np.cos(lam_cont[-1]) < - 1:
            chi_orbit_cont = 0.0
        else:
            chi_orbit_cont = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam_cont[-1]))


    print("---------------- ITER {} -----------------".format(n))
    print("Wind Speed = {} m/s, Start altitude = {} m, Gust size = {} m".format(round(v_wind, 2), round(height_start, 2), round(deltaH, 2)))

    if (v_orbit_cont * 0.95 <= v_ass_cont <= v_orbit_cont * 1.05 and chi_orbit_cont * 0.8 <= chi_ass_cont <= chi_orbit_cont * 1.2
        and np.deg2rad(-0.5) <= gamma_cont[-1] <= np.deg2rad(0.5) and href[-1] * 0.99 <= h_cont[-1] <= href[-1] * 1.01) \
            or (v_orbit_stand * 0.95 <= v_ass_stand <= v_orbit_stand * 1.05 and chi_orbit_stand * 0.8 <= chi_ass_stand <= chi_orbit_stand * 1.2
                and np.deg2rad(-0.5) <= gamma_stand[-1] <= np.deg2rad(0.5) and href[-1] * 0.99 <= h_stand[-1] <= href[-1] * 1.01) \
            or (vref[-1] * 0.99 <= v_cont[-1] <= vref[-1] * 1.01 and chiref[-1] * 0.99 <= chi_cont[-1] <= chiref[-1] * 1.01
                and np.deg2rad(-0.5) <= gamma_cont[-1] <= np.deg2rad(0.5) and href[-1] * 0.99 <= h_cont[-1] <= href[-1] * 1.01
                and tetaref[-1]*1.01 <= teta_cont[-1] <= tetaref[-1]*0.99 and lamref[-1]*0.99 <= lam_cont[-1] <= lamref[-1]*1.01) \
            or (vref[-1] * 0.99 <= v_stand[-1] <= vref[-1] * 1.01 and chiref[-1] * 0.99 <= chi_stand[-1] <= chiref[-1] * 1.01
                and np.deg2rad(-0.5) <= gamma_stand[-1] <= np.deg2rad(0.5) and href[-1] * 0.99 <= h_stand[-1] <= href[-1] * 1.01
                and tetaref[-1]*1.01 <= teta_stand[-1] <= tetaref[-1]*0.99 and lamref[-1]*0.99 <= lam_stand[-1] <= lamref[-1]*1.01):

        if (v_orbit_cont * 0.95 <= v_ass_cont <= v_orbit_cont * 1.05 and chi_orbit_cont * 0.8 <= chi_ass_cont <= chi_orbit_cont * 1.2 and \
                np.deg2rad(-0.5) <= gamma_cont[-1] <= np.deg2rad(0.5) and href[-1] * 0.99 <= h_cont[-1] <= href[-1] * 1.01) or\
            (vref[-1] * 0.99 <= v_cont[-1] <= vref[-1] * 1.01 and chiref[-1] * 0.99 <= chi_cont[-1] <= chiref[-1] * 1.01 and
             np.deg2rad(-0.5) <= gamma_cont[-1] <= np.deg2rad(0.5) and href[-1] * 0.99 <= h_cont[-1] <= href[-1] * 1.01 and \
            tetaref[-1]*1.01 <= teta_cont[-1] <= tetaref[-1]*0.99 and lamref[-1]*0.99 <= lam_cont[-1] <= lamref[-1]*1.01):

            print("NN success")
            nn_sc += 1
            if plot_trajectory:
                nn_success = True
            data_nn.append([v_wind, height_start, deltaH])

        if (v_orbit_stand * 0.95 <= v_ass_stand <= v_orbit_stand * 1.05 and chi_orbit_stand * 0.8 <= chi_ass_stand <= chi_orbit_stand * 1.2 and \
                np.deg2rad(-0.5) <= gamma_stand[-1] <= np.deg2rad(0.5) and href[-1] * 0.99 <= h_stand[-1] <= href[-1] * 1.01) or \
            (vref[-1] * 0.99 <= v_stand[-1] <= vref[-1] * 1.01 and chiref[-1] * 0.99 <= chi_stand[-1] <= chiref[-1] * 1.01 and
             np.deg2rad(-0.5) <= gamma_stand[-1] <= np.deg2rad(0.5) and href[-1] * 0.99 <= h_stand[-1] <= href[-1] * 1.01 and \
            tetaref[-1]*1.01 <= teta_stand[-1] <= tetaref[-1]*0.99 and lamref[-1]*0.99 <= lam_stand[-1] <= lamref[-1]*1.01):

            print("GP success")
            gp_sc += 1
            data_gp.append([v_wind, height_start, deltaH])
            if plot_trajectory:
                nn_success = False

    else:
        diff_v = abs(v_orbit_cont-v_ass_cont)/v_orbit_cont
        diff_chi = abs(chi_orbit_cont - chi_ass_cont)/chi_orbit_cont
        diff_gamma = abs(gammaref[-1] - gamma_cont[-1]) / gammaref[-1]
        diff_h = abs(href[-1] - h_cont[-1]) / href[-1]
        diff.append(np.max([diff_v, diff_chi, diff_gamma, diff_h])*100)
        failure += 1
        data_fail.append([v_wind, height_start, deltaH])

    print("\n")
    if nn_success and plot_trajectory:
        plt.ion()
        plt.figure(2)
        plt.plot(t_stand, v_stand, label="GP control law")
        plt.plot(t_cont, v_cont, label="NN optimization")

        plt.figure(3)
        plt.plot(t_stand, np.rad2deg(chi_stand), label="GP control law")
        plt.plot(t_cont, np.rad2deg(chi_cont), label="NN optimization")

        plt.figure(4)
        plt.plot(t_stand, np.rad2deg(gamma_stand), label="GP control law")
        plt.plot(t_cont, np.rad2deg(gamma_cont), label="NN optimization")

        plt.figure(5)
        plt.plot(t_stand, np.rad2deg(teta_stand), label="GP control law")
        plt.plot(t_cont, np.rad2deg(teta_cont), label="NN optimization")

        plt.figure(6)
        plt.plot(t_stand, np.rad2deg(lam_stand), label="GP control law")
        plt.plot(t_cont, np.rad2deg(lam_cont), label="NN optimization")

        plt.figure(7)
        plt.plot(t_stand, h_stand/1000, label="GP control law")
        plt.plot(t_cont, h_cont/1000, label="NN optimization")

        plt.figure(8)
        plt.plot(t_stand, m_stand, label="GP control law")
        plt.plot(t_cont, m_cont, label="NN optimization")

    n += 1

if nn_success and plot_trajectory:
    plt.ion()
    plt.figure(2)
    plt.plot(tref, vref, 'r--', linewidth=3, label="SET POINT")
    plt.xlabel("time [s]")
    plt.ylabel("Speed [m/s]")
    plt.legend(loc='best', ncol=2)
    plt.grid()

    plt.figure(3)
    plt.plot(tref, np.rad2deg(chiref), 'r--', label="SET POINT")
    plt.xlabel("time [s]")
    plt.ylabel("Chi [deg]")
    plt.legend(loc='best', ncol=2)
    plt.grid()

    plt.figure(4)
    plt.plot(tref, np.rad2deg(gammaref), 'r--', label="SET POINT")
    plt.xlabel("time [s]")
    plt.ylabel("Gamma [deg]")
    plt.legend(loc='best', ncol=2)
    plt.grid()

    plt.figure(5)
    plt.plot(tref, np.rad2deg(tetaref), 'r--', label="SET POINT")
    plt.xlabel("time [s]")
    plt.ylabel("Theta [deg]")
    plt.legend(loc='best', ncol=2)
    plt.grid()

    plt.figure(6)
    plt.plot(tref, np.rad2deg(lamref), 'r--', linewidth=3, label="SET POINT")
    plt.xlabel("time [s]")
    plt.ylabel("Lambda [deg]")
    plt.legend(loc='best', ncol=2)
    plt.grid()

    plt.figure(7)
    plt.plot(tref, href/1000, 'r--', linewidth=3, label="SET POINT")
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [km]")
    plt.legend(loc='best', ncol=2)
    plt.grid()

    plt.figure(8)
    plt.plot(tref, mref, 'r--', linewidth=3, label="SET POINT")
    plt.xlabel("time [s]")
    plt.ylabel("Mass [kg]")
    plt.legend(loc='best', ncol=2)
    plt.grid()

    plt.show(block=True)

np.save('NN_nn_success_points_hof{}_{}_{}_{}.npy'.format(hofnum, method, config, run), data_nn)
np.save('NN_gp_success_points_hof{}_{}_{}_{}.npy'.format(hofnum, method, config, run), data_gp)
np.save('NN_failure_points_hof{}_{}_{}_{}.npy'.format(hofnum, method, config, run), data_fail)

print("NN success {}%".format(round(nn_sc/len(test_points)*100,2)))
print("GP success {}%".format(round(gp_sc/len(test_points)*100,2)))
print("Total successes {}%".format(round((len(test_points)-failure)/len(test_points)*100, 2)))




