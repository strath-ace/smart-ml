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

'''Functions used by the propagators in the MAIN_DensityScenario.py file'''

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
    t_init = k['kwargs']['t_init']
    tfin = k['kwargs']['tfin']
    init_cond = k['kwargs']['init_cond']
    obj = k['kwargs']['obj']

    def sys(t, x):
        # State Variables
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        if R < obj.Re - 0.5 or np.isnan(R):
            penalty.append((R - obj.Re) / obj.Htarget)
            R = obj.Re

        if m < obj.M0 - obj.Mp or np.isnan(m):
            penalty.append((m - (obj.M0 - obj.Mp)) / obj.M0)
            m = obj.M0 - obj.Mp

        r = Rfun(t)
        th = Thetafun(t)
        vr = Vrfun(t)
        vt = Vtfun(t)

        er = r - R
        et = th - theta
        evr = vr - Vr
        evt = vt - Vt

        rho = k['kwargs']['rho_newmodel'](R - obj.Re)
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

    def find_close(t, x):
        """To stop propagation when the difference in height get smaller than 20 meters"""
        return abs(x[0] - Rfun(t)) - 20
    find_close.terminal = True

    sol = solve_ivp(sys, [t_init, tfin], init_cond)
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
        x = [use, pen]
        return x
    else:
        return [use, 0.0]


def air_density(h):
    """
    Function used to evaluate the air density

    Attributes:
        h: float
            altitude
    Returns:
        rho: float
            air density
    """
    beta = 1 / 8500.0  # scale factor [1/m]
    rho0 = 1.225  # kg/m3
    rho = rho0 * np.exp(-beta * h)
    return rho


def isa(altitude, singl_val):
    """
    USSA 1962 atmospheric model

    Attributes:
        altitude: float
            altitude
        singl_val: int
            tells if the altitude is an array or a scalar
    Returns:
        pressure: float or array
            pressure
        density: float or array
            air density
        csound: float or array
            speed of sound
    """
    a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
    a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
    hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
    h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
    tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
    pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
    tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
    tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]
    t0 = 288.15
    p0 = 101325
    prevh = 0.0
    R = 287.00
    m0 = 28.9644
    Rs = 8314.32
    m0 = 28.9644
    g0 = 9.80665
    r = 6371000
    if singl_val == 1:
        altitude = np.array([altitude])
    temperature = np.zeros(len(altitude))
    pressure = np.zeros(len(altitude))
    tempm = np.zeros(len(altitude))
    density = np.zeros(len(altitude))
    csound = np.zeros(len(altitude))
    k = 0

    def cal(ps, ts, av, h0, h1):
        if av != 0:
            t1 = ts + av * (h1 - h0)
            p1 = ps * (t1 / ts) ** (-g0 / av / R)
        else:
            t1 = ts
            p1 = ps * np.exp(-g0 / R / ts * (h1 - h0))
        return t1, p1

    def atm90(a90v, z, hi, tc1, pc, tc2, tmc):
        for num in hi:
            if z <= num:
                ind = hi.index(num)
                if ind == 0:
                    zb = hi[0]
                    b = zb - tc1[0] / a90v[0]
                    t = tc1[0] + tc2[0] * (z - zb) / 1000
                    tm = tmc[0] + a90v[0] * (z - zb) / 1000
                    add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
                    add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
                    p = pc[0] * np.exp(-m0 / (a90v[0] * Rs) * g0 * r ** 2 * (add1 - add2))
                else:
                    zb = hi[ind - 1]
                    b = zb - tc1[ind - 1] / a90v[ind - 1]
                    t = tc1[ind - 1] + (tc2[ind - 1] * (z - zb)) / 1000
                    tm = tmc[ind - 1] + a90v[ind - 1] * (z - zb) / 1000
                    add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
                    add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
                    p = pc[ind - 1] * np.exp(-m0 / (a90v[ind - 1] * Rs) * g0 * r ** 2 * (add1 - add2))
                break
        return t, p, tm

    for alt in altitude:
        if alt < 0:
            t = t0
            p = p0
            d = p / (R * t)
            c = np.sqrt(1.4 * R * t)
            density[k] = d
            csound[k] = c
            pressure[k] = p
        elif 0 <= alt < 90000:

            for i in range(0, 8):

                if alt <= hv[i]:
                    t, p = cal(p0, t0, a[i], prevh, alt)
                    d = p / (R * t)
                    c = np.sqrt(1.4 * R * t)
                    density[k] = d
                    csound[k] = c
                    temperature[k] = t
                    pressure[k] = p
                    tempm[k] = t
                    t0 = 288.15
                    p0 = 101325
                    prevh = 0
                    break
                else:
                    t0, p0 = cal(p0, t0, a[i], prevh, hv[i])
                    prevh = hv[i]

        elif 90000 <= alt <= 190000:
            t, p, tpm = atm90(a90, alt, h90, tcoeff1, pcoeff, tcoeff2, tmcoeff)
            d = p / (R * tpm)
            c = np.sqrt(1.4 * R * tpm)
            density[k] = d
            csound[k] = c
            pressure[k] = p
        elif alt > 190000:
            zb = h90[6]
            z = h90[-1]
            b = zb - tcoeff1[6] / a90[6]
            t = tcoeff1[6] + (tcoeff2[6] * (z - zb)) / 1000
            tm = tmcoeff[6] + a90[6] * (z - zb) / 1000
            add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
            add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
            p = pcoeff[6] * np.exp(-m0 / (a90[6] * Rs) * g0 * r ** 2 * (add1 - add2))
            d = p / (R * t)
            c = np.sqrt(1.4 * R * tm)
            density[k] = d
            csound[k] = c
            pressure[k] = p
        k += 1
    return pressure, density, csound


def sys2GP_ISA(t, x, obj, Trfun, Ttfun):
    """
    Function used to propagate dynamics with ISA atmospheric model

    Attributes:
        t: float
            time
        x: array
            state variables
        obj: class
            plant model
        Trfun: interpolating function
            function that interpolates Tr with respect of time
        Ttfun: interpolating function
            function that interpolates Tt with respect of time
    Return:
        dxdt: array
            derivatives of state variables
    """
    Cd = obj.Cd

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

    rho = float(isa(R - obj.Re, 1)[1])

    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt = np.array((Vr,
                     Vt / R,
                     Tr / m - Dr / m - g + Vt ** 2 / R,
                     Tt / m - Dt / m - (Vr * Vt) / R,
                     -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

    return dxdt


def sys_rho(t, x, expr1, expr2, obj, toolbox, Rfun, Thetafun, Vrfun, Vtfun, Trfun, Ttfun, rho_newmodel):
    """
    Function used to propagate dynamics with simplified atmospheric model

    Attributes:
        t: float
            time
        x: array
            state variables
        expr1: list
            first GP individual
        expr2: list
            second GP individual
        obj: class
            plant model
        toolbox: class
            GP building blocks
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
        rho_newmodel: interpolating function
            interpolates rho with respect to the altitude
    Return:
        dxdt: array
            derivatives of state variables
    """
    Cd = obj.Cd
    fTr = toolbox.compileR(expr1)
    fTt = toolbox.compileT(expr2)
    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    er = Rfun(t) - R
    et = Thetafun(t) - theta
    evr = Vrfun(t) - Vr
    evt = Vtfun(t) - Vt

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

    rho = rho_newmodel(R - obj.Re)

    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt = np.array((Vr,
                     Vt / R,
                     Tr / m - Dr / m - g + Vt ** 2 / R,
                     Tt / m - Dt / m - (Vr * Vt) / R,
                     -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

    return dxdt


def sys_rho_init(t, x, obj, Trfun, Ttfun, rho_newmodel):
    """
     Function used to propagate dynamics with simplified atmospheric model without Gp control law

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
             interpolates rho with respect to the altitude
     Return:
         dxdt: array
             derivatives of state variables
     """
    Cd = obj.Cd

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

    rho = rho_newmodel(R - obj.Re)

    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt = np.array((Vr,
                     Vt / R,
                     Tr / m - Dr / m - g + Vt ** 2 / R,
                     Tt / m - Dt / m - (Vr * Vt) / R,
                     -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

    return dxdt