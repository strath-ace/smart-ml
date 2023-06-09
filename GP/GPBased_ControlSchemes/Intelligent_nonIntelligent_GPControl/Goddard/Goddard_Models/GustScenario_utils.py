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

'''Functions used by the propagators in the MAIN_GustScenario.py file'''

import numpy as np
from deap import gp
from scipy.integrate import solve_ivp, simps

def evaluate(individual, pset, **k):
    """
    Function used to evaluate the fitness of the individual

    Attributes:
        individual: list
            GP individual
        pset: class
            primitive set
        **k: kargs

    Return:
        use: float
            fitness
        pen: float
            constraints violation
    """
    penalty = []

    fTr = gp.compile(expr=individual[0], pset=pset[0])
    fTt = gp.compile(expr=individual[1], pset=pset[1])

    Rfun = k['kwargs']['Rfun']
    Thetafun = k['kwargs']['Thetafun']
    Vrfun = k['kwargs']['Vrfun']
    Vtfun = k['kwargs']['Vtfun']
    Trfun = k['kwargs']['Trfun']
    Ttfun = k['kwargs']['Ttfun']
    change_time = k['kwargs']['change_time']
    tfin = k['kwargs']['tfin']
    x_ini_h = k['kwargs']['x_ini_h']
    obj = k['kwargs']['obj']
    delta_eval = k['kwargs']['delta_eval']
    height_start = k['kwargs']['height_start']
    delta = k['kwargs']['deltaH']
    v_wind = k['kwargs']['v_wind']

    def sys(t, x):
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

        if m < obj.M0 - obj.Mp or np.isnan(m):
            penalty.append((m - (obj.M0 - obj.Mp)) / obj.M0)
            m = obj.M0 - obj.Mp

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
            Tr = 0
        elif Tr < 0.0 or np.isnan(Tr):
            penalty.append((Tr) / obj.Tmax)
            Tr = 0.0
        elif Tr > obj.Tmax or np.isinf(Tr):
            penalty.append((Tr - obj.Tmax) / obj.Tmax)
            Tr = obj.Tmax
        if np.iscomplex(Tt):
            Tt = 0
        elif Tt < 0.0 or np.isnan(Tt):
            penalty.append((Tt) / obj.Tmax)
            Tt = 0.0
        elif Tt > obj.Tmax or np.isinf(Tt):
            penalty.append((Tt - obj.Tmax) / obj.Tmax)
            Tt = obj.Tmax

        dxdt = np.array((Vr,
                         Vt / R,
                         Tr / m - Dr / m - g + Vt ** 2 / R,
                         Tt / m - Dt / m - (Vr * Vt) / R,
                         -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))
        return dxdt


    sol = solve_ivp(sys, [change_time+delta_eval, tfin], x_ini_h)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]

    tt = sol.t

    if tt[-1] < tfin:
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
        return [use, pen]
    else:
        return [use, 0.0]


def sys2GP(t, x, expr1, expr2, v_wind, height_start, delta, toolbox, obj, Rfun, Thetafun, Vrfun, Vtfun, Trfun, Ttfun):
    """
    Function used to propagate dynamics

    Attributes:
        t: float
            time
        x: array
            state variables
        expr1: list
            first GP individual
        expr2: list
            second GP individual
        v_wind: float
            wind gust speed
        height_start: float
            altitude at which the gust is inserted
        delta: float
            altitude range in which the gust acts
        toolbox: class
            GP building blocks
        obj: class
            plant model
        Rfun: interpolating function
            function interpolating R with respect of time
        Thetafun: interpolating function
            function interpolating Theta with respect of time
        Vrfun: interpolating function
            function interpolating Vr with respect of time
        Vtfun: interpolating function
            function interpolating Vt with respect of time
        Trfun: interpolating function
            function interpolating Tr with respect of time
        Ttfun: interpolating function
            function interpolating Tt with respect of time
    Return:
        dxdt: array
            derivatives of state variables
    """
    fTr = toolbox.compileR(expr1)
    fTt = toolbox.compileT(expr2)

    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    if height_start < R < height_start + delta:
        Vt = Vt - v_wind*np.cos(theta)
        Vr = Vr - v_wind * np.sin(theta)

    r = Rfun(t)
    th = Thetafun(t)
    vr = Vrfun(t)
    vt = Vtfun(t)

    er = r - R
    et = th - theta
    evr = vr - Vr
    evt = vt - Vt

    Tr = Trfun(t) + fTr(er, evr)
    Tt = Ttfun(t) + fTt(et, evt)
    if m <= obj.M0 - obj.Mp:
        Tr = 0.0
        Tt = 0.0
        m = obj.M0 - obj.Mp
    if Tr > obj.Tmax:
        Tr = obj.Tmax
    elif Tr < 0:
        Tr = 0.0
    if Tt > obj.Tmax:
        Tt = obj.Tmax
    elif Tt < 0:
        Tt = 0.0
    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt = np.array((Vr,
                     Vt / R,
                     Tr / m - Dr / m - g + Vt ** 2 / R,
                     Tt / m - Dt / m - (Vr * Vt) / R,
                     -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

    return dxdt

def sys_init(t, x, obj, Trfun, Ttfun):
    """
    Function used to propagate dynamics without gust variation

    Attributes:
        t: float
            time
        x: array
            state variables
        obj: class
            plant model
        Trfun: interpolating function
            function interpolating Tr with respect of time
        Ttfun: interpolating function
            function interpolating Tt with respect of time
        rho_newmodel: interpolating function
    Return:
        dxdt: array
            derivatives of state variables
    """
    R = x[0]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    Tr = Trfun(t)
    Tt = Ttfun(t)
    if m <= obj.M0 - obj.Mp:
        Tr = 0.0
        Tt = 0.0
        m = obj.M0 - obj.Mp
    if Tr > obj.Tmax:
        Tr = obj.Tmax
    elif Tr < 0:
        Tr = 0.0
    if Tt > obj.Tmax:
        Tt = obj.Tmax
    elif Tt < 0:
        Tt = 0.0
    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt = np.array((Vr,
                     Vt / R,
                     Tr / m - Dr / m - g + Vt ** 2 / R,
                     Tt / m - Dt / m - (Vr * Vt) / R,
                     -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

    return dxdt

def sys_ifnoC(t, x, height_start, delta, v_wind, Trfun, Ttfun, obj):
    """
    Function used to propagate dynamics with gust and no control

    Attributes:
        t: float
            time
        x: array
            state variables
        v_wind: float
            wind gust speed
        height_start: float
            altitude at which the gust is inserted
        delta: float
            altitude range in which the gust acts
        obj: class
            plant model
        Trfun: interpolating function
            function interpolating Tr with respect of time
        Ttfun: interpolating function
            function interpolating Tt with respect of time
    Return:
        dxdt: array
            derivatives of state variables
    """
    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    if height_start < R < height_start + delta:
        Vt = Vt - v_wind * np.cos(theta)
        Vr = Vr - v_wind * np.sin(theta)

    Tr = Trfun(t)
    Tt = Ttfun(t)
    if m <= obj.M0 - obj.Mp:
        Tr = 0.0
        Tt = 0.0
        m = obj.M0 - obj.Mp
    if Tr > obj.Tmax:
        Tr = obj.Tmax
    elif Tr < 0:
        Tr = 0.0
    if Tt > obj.Tmax:
        Tt = obj.Tmax
    elif Tt < 0:
        Tt = 0.0
    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt = np.array((Vr,
                     Vt / R,
                     Tr / m - Dr / m - g + Vt ** 2 / R,
                     Tt / m - Dt / m - (Vr * Vt) / R,
                     -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

    return dxdt