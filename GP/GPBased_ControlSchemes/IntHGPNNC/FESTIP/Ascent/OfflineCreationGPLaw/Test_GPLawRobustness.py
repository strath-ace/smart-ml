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

'''
This script tests the robustness capabilities of the produced control laws both with IGP and SGP on a predefined
set of 500 disturbance scenarios
'''

import sys
import os
data_path = os.path.join(os.path.dirname( __file__ ), '..', 'Datasets')
gplaw_path = os.path.join(os.path.dirname( __file__ ), 'ResultsBIOMA2020')
models_path = os.path.join(os.path.dirname( __file__ ), '../../../../Intelligent_nonIntelligent_GPControl/FESTIP/FESTIP_Models')
sys.path.append(data_path)
sys.path.append(gplaw_path)
from scipy.integrate import solve_ivp
import numpy as np
import operator
import _pickle as pickle
import random
from deap import gp, base, creator
import multiprocessing
from scipy.interpolate import PchipInterpolator
from time import strftime
import matplotlib
import GP.GP_Algorithms.IGP.IGP_Functions as funs
import GP.GPBased_ControlSchemes.Intelligent_nonIntelligent_GPControl.FESTIP.FESTIP_Models.models_FESTIP as mods
import scipy.io as sio
import GP.GP_Algorithms.GP_PrimitiveSet as gpprim
from tqdm import *


matplotlib.rcParams.update({'font.size': 22})
timestr = strftime("%Y%m%d-%H%M%S")

nbCPU = multiprocessing.cpu_count()

######################### LOAD DATA ####################à
obj = mods.Spaceplane_Ascent()

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

samples = np.load(data_path + "/training_points.npy")
original_points = np.load(data_path + "/GP_creationSet.npy")

####################################    P R I M I T I V E  -  S E T     ################################################

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

for i in range(2):
    pset.addEphemeralConstant("rand{}".format(i), lambda: round(random.uniform(-10, 10), 4))

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

########################################################################################################################

def run_test(i):

    height_start = samples[i][0]
    deltaH = samples[i][2]
    v_wind = samples[i][1]

    x_ini = [vref[0], chiref[0], gammaref[0], tetaref[0], lamref[0], href[0], mref[0]]

    def find_height(t, x, *args):
        return height_start - x[5]

    find_height.terminal = True
    tev = np.linspace(2, tfin, 200)
    ####### propagation to find where the gust zone starts ###############
    init = solve_ivp(mods.sys_init, [2, tfin], x_ini, events=find_height, method='RK45', t_eval=tev,
                     args=(cl, cd, cm, spimpv, presv, obj, alfafun, deltafun))
    change_time = init.t[-1]

    x_ini_h = [init.y[0, -1], init.y[1, -1], init.y[2, -1], init.y[3, -1], init.y[4, -1], init.y[5, -1], init.y[6, -1]]

    ##### propagation from the start of the gust zone until the end of the trajectory #####

    Npoints = 500
    t_max_int = 30
    fAlpha = gp.compile(hof[-1][0], pset=pset)
    fDelta = gp.compile(hof[-1][1], pset=pset)
    solgp, t_stop, stop_index = mods.RK4(change_time, tfin, mods.sys2GP_uncert, Npoints, x_ini_h, t_max_int,
                                                args=(fAlpha, fDelta, v_wind, cl, cd, cm, spimpv, presv, height_start,
                                                      deltaH, vfun, chifun, gammafun, hfun, alfafun, deltafun, obj))

    vout = solgp[:stop_index, 0]
    chiout = solgp[:stop_index, 1]
    gammaout = solgp[:stop_index, 2]
    tetaout = solgp[:stop_index, 3]
    lamout = solgp[:stop_index, 4]
    hout = solgp[:stop_index, 5]

    v_ass, chi_ass = mods.vass(solgp[stop_index - 1, :], obj.omega)


    if hout[-1]<obj.hmin:
        hout[-1] = obj.hmin

    v_orbit = np.sqrt(obj.GMe / (obj.Re + hout[-1]))

    if np.cos(obj.incl) / np.cos(lamout[-1]) > 1:
        chi_orbit = np.pi
    else:
        if np.cos(obj.incl) / np.cos(lamout[-1]) < - 1:
            chi_orbit = 0.0
        else:
            chi_orbit = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lamout[-1]))


    if (v_orbit * 0.95 <= v_ass <= v_orbit * 1.05 and chi_orbit * 0.8 <= chi_ass <= chi_orbit * 1.2 and \
            np.deg2rad(-0.5) <= gammaout[-1] <= np.deg2rad(0.5) and href[-1] * 0.99 <= hout[-1] <= href[
        -1] * 1.01) or (vref[-1] * 0.99 <= vout[-1] <= vref[-1] * 1.01 and
         chiref[-1] * 0.99 <= chiout[-1] <= chiref[-1] * 1.01 and
         np.deg2rad(-0.5) <= gammaout[-1] <= np.deg2rad(0.5) and
         tetaref[-1] * 1.01 <= tetaout[-1] <= tetaref[-1] * 0.99 and
         lamref[-1] * 0.99 <= lamout[-1] <= lamref[-1] * 1.01 and
         href[-1] * 0.99 <= hout[-1] <= href[-1] * 1.01):  # tolerance of 1%
        return
    else:
        index = i
        return index


if __name__ == '__main__':
    gp_types = ['IGP', 'SGP']
    hof_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for hof_num in hof_nums:
        for gp_type in gp_types:

            with open(gplaw_path + '/' + gp_type + '_hof_{}.pkl'.format(hof_num), 'rb') as f:
                hof = pickle.load(f)

            print(hof[-1].fitness.values)
            print("Ind 1 Length {}, Height {}".format(len(hof[-1][0]), hof[-1][0].height))
            print("Ind 2 Length {}, Height {}".format(len(hof[-1][1]), hof[-1][1].height))
            success_index = np.linspace(0, len(samples)-1, len(samples), dtype=int)
            failure_index = []
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            with tqdm(total=len(samples)) as pbar:
                for i, id in enumerate(pool.imap_unordered(run_test, range(len(samples)))):
                    pbar.update()
                    failure_index.append(id)
            indexx = list(filter(None.__ne__, failure_index))
            success_index = [x for x in success_index if x not in failure_index]
            pool.close()
            pool.join()
            print(gp_type+" hof {} Successes {}/{}".format(hof_num, len(success_index), len(samples)))
            np.save(gp_type+'_success_points_hof_{}.npy'.format(hof_num), success_index)












