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
This script contains the functions used by the OfflineGpLawTuner.py script
"""

from deap import gp
from copy import deepcopy
import numpy as np
import sys
import os
data_path = os.path.join(os.path.dirname( __file__ ), '..', 'Datasets')
sys.path.append(data_path)
import GP.GPBased_ControlSchemes.Intelligent_nonIntelligent_GPControl.FESTIP.FESTIP_Models.models_FESTIP as mods

def update_eqs(exprs, prediction):
    k = 0
    for i in range(len(exprs)):
        for j in range(len(exprs[i])):
            if type(exprs[i][j]) == gp.Terminal and exprs[i][j].name == "w":
                exprs[i][j] = deepcopy(exprs[i][j])
                exprs[i][j].value = deepcopy(prediction[k])
                k += 1
    for i in range(len(exprs)):
        for j in range(len(exprs[i])):
            if type(exprs[i][j]) == gp.rand0 or type(exprs[i][j]) == gp.rand1:
                exprs[i][j] = deepcopy(exprs[i][j])
                exprs[i][j].value = deepcopy(prediction[k])
                k += 1

    return exprs[0], exprs[1]

def test_param_opt(param, height_start, deltaH, v_wind, bestFuns, pset, obj, vfun, chifun, gammafun, hfun, alfafun,
                   deltafun, cl, cd, cm, spimpv, presv, tfin, href, final_cond, change_time, x_ini_h):


    bestFuns[0], bestFuns[1] = update_eqs(bestFuns, param)

    fAlpha = gp.compile(bestFuns[0], pset=pset)
    fDelta = gp.compile(bestFuns[1], pset=pset)

    flag = False

    Npoints = 500
    t_max_int = 30

    solgp, t_stop, stop_index = mods.RK4(change_time, tfin, mods.sys2GP_uncert, Npoints, x_ini_h, t_max_int,
                                         args=(fAlpha, fDelta, v_wind, cl, cd, cm, spimpv, presv, height_start,
                                               deltaH, vfun, chifun, gammafun, hfun, alfafun, deltafun, obj))
    vout = solgp[:stop_index, 0]
    chiout = solgp[:stop_index, 1]
    gammaout = solgp[:stop_index, 2]
    tetaout = solgp[:stop_index, 3]
    lamout = solgp[:stop_index, 4]
    hout = solgp[:stop_index, 5]
    tt = np.linspace(change_time, t_stop, stop_index)
    v_ass, chi_ass = mods.vass(solgp[stop_index - 1, :], obj.omega)

    if hout[-1] < 1.0:
        hout[-1] = 1.0

    v_orbit = np.sqrt(obj.GMe / (obj.Re + hout[-1]))

    if np.cos(obj.incl) / np.cos(lamout[-1]) > 1:
        chi_orbit = np.pi
    else:
        if np.cos(obj.incl) / np.cos(lamout[-1]) < - 1:
            chi_orbit = 0.0
        else:
            chi_orbit = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lamout[-1]))

    if tt[-1] < tfin:
        flag = True

    if flag is True:
        return False, solgp, tt
    if (v_orbit * 0.95 <= v_ass <= v_orbit * 1.05 and
        chi_orbit * 0.8 <= chi_ass <= chi_orbit * 1.2 and
        np.deg2rad(-0.5) <= gammaout[-1] <= np.deg2rad(0.5) and
        href[-1] * 0.99 <= hout[-1] <= href[-1] * 1.01) or \
        (final_cond[0] * 0.99 <= vout[-1] <= final_cond[0] * 1.01 and
         final_cond[1] * 0.99 <= chiout[-1] <= final_cond[1] * 1.01 and
         np.deg2rad(-0.5) <= gammaout[-1] <= np.deg2rad(0.5) and
         final_cond[3] * 1.01 <= tetaout[-1] <= final_cond[3] * 0.99 and
         final_cond[4] * 0.99 <= lamout[-1] <= final_cond[4] * 1.01 and
         final_cond[5] * 0.99 <= hout[-1] <= final_cond[5] * 1.01):

        return True, solgp, tt
    else:
        return False, solgp, tt


def tree_eval_opt(param, height_start, deltaH, v_wind, bestFuns, pset, obj, vfun, chifun, gammafun, hfun, alfafun,
                  deltafun, cl, cd, cm, spimpv, presv, tfin, final_cond, change_time, x_ini_h):


    bestFuns[0], bestFuns[1] = update_eqs(bestFuns, param)

    fAlpha = gp.compile(bestFuns[0], pset=pset)
    fDelta = gp.compile(bestFuns[1], pset=pset)


    Npoints = 500
    t_max_int = 30
    solgp, t_stop, stop_index = mods.RK4(change_time, tfin, mods.sys2GP_uncert, Npoints, x_ini_h, t_max_int,
                                                args=(fAlpha, fDelta, v_wind, cl, cd, cm, spimpv, presv, height_start,
                                                      deltaH, vfun, chifun, gammafun, hfun, alfafun, deltafun, obj))
    vout = solgp[:stop_index, 0]
    chiout = solgp[:stop_index, 1]
    gammaout = solgp[:stop_index, 2]
    hout = solgp[:stop_index, 5]

    if hout[-1] < 1.0:
        hout[-1] = 1.0

    diffV = (final_cond[0] - vout[-1]) / obj.vmax
    diffChi = (final_cond[1] - chiout[-1]) / obj.chimax
    diffGamma = gammaout[-1] / obj.gammamax
    diffH = (final_cond[5] - hout[-1]) / obj.hmax

    res = np.array([diffV, diffChi, diffGamma, diffH])

    return np.sqrt(1/len(res)*sum(res**2))
