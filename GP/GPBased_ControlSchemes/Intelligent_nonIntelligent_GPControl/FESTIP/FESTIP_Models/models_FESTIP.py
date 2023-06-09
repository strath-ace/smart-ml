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
This script contains all the models related to the FESTIP FSS5 vehicle. All these files should be modified in order
to use the proposed control approach on another vehicle
"""

import sys
import os
gpfun_path = os.path.join(os.path.dirname( __file__ ), '../../../NAGEP/FESTIP/Ascent', 'GP_Functions')
sys.path.append(gpfun_path)
nnfun_path = os.path.join(os.path.dirname( __file__ ), '../../../NAGEP/FESTIP/Ascent', '2_Optimization_GPLaw')
sys.path.append(nnfun_path)
import numpy as np
import random
from deap import gp
from scipy.integrate import simps
import time
from sklearn import preprocessing
import GP.GPBased_ControlSchemes.IntHGPNNC.FESTIP.Ascent.Optimization_GPLaw.TreeTunerUtils as optiutils
from copy import copy
from scipy.integrate import solve_ivp


class Spaceplane_Ascent:
    """
    FESTIP FSS5 model configured for ascent mission
    """
    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = np.deg2rad(5.2)  # deg latitude
        self.longstart = np.deg2rad(-52.775)  # deg longitude
        self.chistart = np.deg2rad(125)  # deg flight direction
        self.incl = np.deg2rad(51.6)  # deg orbit inclination
        self.gammastart = np.deg2rad(89)  # deg
        self.M0 = 450400  # kg  starting mass
        self.g0 = 9.80665  # m/s2
        self.gIsp = self.g0 * 455  # g0 * Isp max
        self.omega = 7.2921159e-5
        self.MaxQ = 40000  # Pa
        self.MaxAx = 30  # m/s2
        self.MaxAz = 15  # m/s2
        self.Htarget = 400000  # m target height after hohmann transfer
        self.wingSurf = 500.0  # m2
        self.lRef = 34.0  # m
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 100000
        self.tvert = 2  # [s]
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(self.incl) / np.cos(self.latstart))
        self.gammamax = np.deg2rad(89)
        self.gammamin = np.deg2rad(-40)
        self.chimax = np.deg2rad(170)
        self.chimin = np.deg2rad(100)
        self.lammax = np.deg2rad(30)
        self.lammin = np.deg2rad(2)
        self.tetamax = np.deg2rad(-10)
        self.tetamin = np.deg2rad(-70)
        self.hmax = 2e5
        self.hmin = 1.0
        self.vmax = 1e4
        self.vmin = 1.0
        self.alfamax = np.deg2rad(40)
        self.alfamin = np.deg2rad(-2)
        self.deltamax = 1.0
        self.deltamin = 0.0
        self.vstart = 0.1
        self.hstart = 0.1
        self.mach = []
        self.angAttack = []
        self.bodyFlap = [-20, -10, 0, 10, 20, 30]
        self.a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
        self.a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
        self.hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
        self.h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
        self.tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
        self.pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
        self.tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
        self.tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]


class Spaceplane_Reentry:
    """
    FESTIP FSS5 model configured for reentry mission
    """
    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.g0 = 9.80665  # m/s2
        self.omega = 7.2921159e-5
        self.m = 45040  # kg  starting mass
        self.vend = 91.44  # m / s
        self.chiend = np.deg2rad(-60)
        self.gammaend = np.deg2rad(-6)
        self.tetaend = np.deg2rad(-80.7112)
        self.tetaend_tol = np.deg2rad(0.0014)
        self.lamend = np.deg2rad(28.6439)
        self.lamend_tol = np.deg2rad(0.0014)
        self.hend = 609.60  # m
        self.hend_tol = 121.92  # m
        self.MaxAx = 30
        self.MaxAz = 25
        self.Maxq = 40000  # Pa
        self.MaxQ = 4e6  # W/m^2
        self.Maxn = 2.5  # g's
        self.Cfin = 9.12e-4  # (kg ^ 0.5 * m ^ 1.5 * s ^ -3)
        self.Lam = np.deg2rad(45)  # deg
        self.k = 4.47228e-9
        self.wingSurf = 500.0  # m2
        self.lRef = 34.0  # m
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.vmax = 5200
        self.chimax = np.deg2rad(180)
        self.gammamax = np.deg2rad(89)
        self.tetamax = np.deg2rad(90)
        self.lammax = np.deg2rad(89)
        self.hmax = 121920
        self.vmin = 1.0
        self.chimin = np.deg2rad(-180)
        self.gammamin = np.deg2rad(-89)
        self.tetamin = np.deg2rad(-90)
        self.lammin = np.deg2rad(-89)
        self.hmin = 1.0
        self.alfamax = np.deg2rad(40)
        self.alfamin = np.deg2rad(-2)
        self.sigmamax = np.deg2rad(90)
        self.sigmamin = np.deg2rad(-90)
        self.mach = []
        self.angAttack = []
        self.bodyFlap = [-20, -10, 0, 10, 20, 30]
        self.a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
        self.a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
        self.hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
        self.h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
        self.tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
        self.pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
        self.tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
        self.tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]
        self.varOld = 0
        self.UB = np.array([self.vmax, self.chimax, self.gammamax, self.tetamax, self.lammax, self.hmax])
        self.LB = np.array([self.vmin, self.chimin, self.gammamin, self.tetamin, self.lammin, self.hmin])


##### ATMOSPHERIC MODELS - USSA1962 MODEL

def cal(ps, ts, av, h0, h1, g0, R):
    if av != 0:
        t1 = ts + av * (h1 - h0)
        p1 = ps * (t1 / ts) ** (-g0 / av / R)
    else:
        t1 = ts
        p1 = ps * np.exp(-g0 / R / ts * (h1 - h0))
    return t1, p1


def atm90(a90v, z, hi, tc1, pc, tc2, tmc, r, m0, g0, Rs):
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


def atm_uncertainty(h, hc, lbt, ubt, lbp, ubp):
    if h > hc:
        h = hc
    if h < 1.0:
        h = 1.0
    eps_p = lbp * (1 - h / hc) + ubp * (h / hc)
    eps_t = lbt * (1 - h / hc) + ubt * (h / hc)
    return eps_p, eps_t


def isa_uncertain(alt, obj, hc, lbt, ubt, lbp, ubp, lhs_sample=None, extend=False):
    t0 = 288.15
    p0 = obj.psl
    prevh = 0.0
    R = 287.00
    Rs = 8314.32
    m0 = 28.9644
    eps_p, eps_t = atm_uncertainty(alt, hc, lbt, ubt, lbp, ubp)
    if lhs_sample is not None and extend:
        r_p = random.uniform(lhs_sample[0]*0.99, lhs_sample[0]*1.01)
        r_t = random.uniform(lhs_sample[1]*0.99, lhs_sample[1]*1.01)
    elif lhs_sample is not None and not extend:
        r_p = lhs_sample[0]
        r_t = lhs_sample[1]
    if alt < 0 or np.isnan(alt):
        t = t0
        p = p0
        if lhs_sample is not None:
            t_out = t * (r_t * (1 + eps_t - (1 - eps_t)) + 1 - eps_t)
            p_out = p * (r_p * (1+eps_p - (1-eps_p)) + 1-eps_p)
            #t_out = t * (lhs_sample[0] * (1+eps_t - (1-eps_t)) + 1-eps_t)
            #p_out = p * (lhs_sample[1] * (1+eps_p - (1-eps_p)) + 1-eps_p)
        else:
            t_out = random.uniform(t * (1-eps_t), t * (1 + eps_t))
            p_out = random.uniform(p * (1-eps_p), p * (1 + eps_p))
        dens = p_out / (R * t_out)
        c = np.sqrt(1.4 * R * t_out)
    elif 0 <= alt < 90000:
        for i in range(0, 8):
            if alt <= obj.hv[i]:
                t, p = cal(p0, t0, obj.a[i], prevh, alt, obj.g0, R)
                if lhs_sample is not None:
                    t_out = t * (r_t * (1 + eps_t - (1 - eps_t)) + 1 - eps_t)
                    p_out = p * (r_p * (1 + eps_p - (1 - eps_p)) + 1 - eps_p)
                    #t_out = t * (lhs_sample[0] * (1 + eps_t - (1 - eps_t)) + 1 - eps_t)
                    #p_out = p * (lhs_sample[1] * (1 + eps_p - (1 - eps_p)) + 1 - eps_p)
                else:
                    t_out = random.uniform(t * (1 - eps_t), t * (1 + eps_t))
                    p_out = random.uniform(p * (1 - eps_p), p * (1 + eps_p))
                dens = p_out / (R * t_out)
                c = np.sqrt(1.4 * R * t_out)
                break
            else:
                t0, p0 = cal(p0, t0, obj.a[i], prevh, obj.hv[i], obj.g0, R)
                prevh = obj.hv[i]

    elif 90000 <= alt <= 190000:
        t, p, tpm = atm90(obj.a90, alt, obj.h90, obj.tcoeff1, obj.pcoeff, obj.tcoeff2, obj.tmcoeff, obj.Re, m0, obj.g0, Rs)
        if lhs_sample is not None:
            t_out = tpm * (r_t * (1 + eps_t - (1 - eps_t)) + 1 - eps_t)
            p_out = p * (r_p * (1 + eps_p - (1 - eps_p)) + 1 - eps_p)
            #t_out = tpm * (lhs_sample[0] * (1+eps_t - (1-eps_t)) + 1-eps_t)
            #p_out = p * (lhs_sample[1] * (1+eps_p - (1-eps_p)) + 1-eps_p)
        else:
            t_out = random.uniform(t * (1-eps_t), t * (1 + eps_t))
            p_out = random.uniform(p * (1-eps_p), p * (1 + eps_p))
        dens = p_out / (R * t_out)
        c = np.sqrt(1.4 * R * t_out)
    elif alt > 190000 or np.isinf(alt):
        zb = obj.h90[6]
        z = obj.h90[-1]
        b = zb - obj.tcoeff1[6] / obj.a90[6]
        t = obj.tcoeff1[6] + (obj.tcoeff2[6] * (z - zb)) / 1000
        tm = obj.tmcoeff[6] + obj.a90[6] * (z - zb) / 1000
        add1 = 1 / ((obj.Re + b) * (obj.Re + z)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((z - b) / (z + obj.Re)))
        add2 = 1 / ((obj.Re + b) * (obj.Re + zb)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((zb - b) / (zb + obj.Re)))
        p = obj.pcoeff[6] * np.exp(-m0 / (obj.a90[6] * Rs) * obj.g0 * obj.Re ** 2 * (add1 - add2))
        if lhs_sample is not None:
            t_out = tm * (r_t * (1 + eps_t - (1 - eps_t)) + 1 - eps_t)
            p_out = p * (r_p * (1 + eps_p - (1 - eps_p)) + 1 - eps_p)
            #t_out = tm * (lhs_sample[0] * (1 + eps_t - (1 - eps_t)) + 1 - eps_t)
            #p_out = p * (lhs_sample[1] * (1 + eps_p - (1 - eps_p)) + 1 - eps_p)
        else:
            t_out = random.uniform(tm * (1 - eps_t), tm * (1 + eps_t))
            p_out = random.uniform(p * (1 - eps_p), p * (1 + eps_p))
        dens = p_out / (R * t_out)
        c = np.sqrt(1.4 * R * t_out)
    #if np.isnan(dens) or np.isnan(c):
    #    print('nan')
    return p_out, dens, c


def isa_noUncertain(alt, obj):
    t0 = 288.15
    p0 = obj.psl
    prevh = 0.0
    R = 287.00
    Rs = 8314.32
    m0 = 28.9644

    if alt < 0 or np.isnan(alt):
        t = t0
        p = p0
        d = p / (R * t)
        c = np.sqrt(1.4 * R * t)
    elif 0 <= alt < 90000:
        for i in range(0, 8):
            if alt <= obj.hv[i]:
                t, p = cal(p0, t0, obj.a[i], prevh, alt, obj.g0, R)
                d = p / (R * t)
                c = np.sqrt(1.4 * R * t)
                break
            else:
                t0, p0 = cal(p0, t0, obj.a[i], prevh, obj.hv[i], obj.g0, R)
                p = p0
                prevh = obj.hv[i]

    elif 90000 <= alt <= 190000:
        t, p, tpm = atm90(obj.a90, alt, obj.h90, obj.tcoeff1, obj.pcoeff, obj.tcoeff2, obj.tmcoeff, obj.Re, m0, obj.g0, Rs)
        d = p / (R * tpm)
        c = np.sqrt(1.4 * R * tpm)
    elif alt > 190000 or np.isinf(alt):
        zb = obj.h90[6]
        z = obj.h90[-1]
        b = zb - obj.tcoeff1[6] / obj.a90[6]
        tm = obj.tmcoeff[6] + obj.a90[6] * (z - zb) / 1000
        add1 = 1 / ((obj.Re + b) * (obj.Re + z)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((z - b) / (z + obj.Re)))
        add2 = 1 / ((obj.Re + b) * (obj.Re + zb)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((zb - b) / (zb + obj.Re)))
        p = obj.pcoeff[6] * np.exp(-m0 / (obj.a90[6] * Rs) * obj.g0 * obj.Re ** 2 * (add1 - add2))
        d = p / (R * tm)
        c = np.sqrt(1.4 * R * tm)

    return p, d, c


##### AERODYNAMIC MODELS

def limCalc(array, value):
    j = 0
    lim = array.__len__()
    for num in array:
        if j == lim-1:
            sup = num
            inf = array[j - 1]
        if value < num:
            sup = num
            if j == 0:
                inf = num
            else:
                inf = array[j - 1]
            break
        j += 1
    s = np.where(array==sup)
    i = np.where(array==inf)
    return i, s


def coefCalc(coeff, m, alfa, mach, angAttack):
    if m > mach[-1] or np.isinf(m):
        m = mach[-1]
    elif m < mach[0] or np.isnan(m):
        m = mach[0]
    if alfa > angAttack[-1] or np.isinf(alfa):
        alfa = angAttack[-1]
    elif alfa < angAttack[0] or np.isnan(alfa):
        alfa = angAttack[0]

    im, sm = limCalc(mach, m)  # moments boundaries and determination of the 2 needed tables

    ia, sa = limCalc(angAttack, alfa)  # angle of attack boundaries

    rowinf1 = coeff[ia][0]
    rowsup1 = coeff[sa][0]
    coeffinf = [rowinf1[im], rowsup1[im]]
    coeffsup = [rowinf1[sm], rowsup1[sm]]
    c1 = coeffinf[0] + (alfa - angAttack[ia]) * ((coeffinf[1] - coeffinf[0]) / (angAttack[sa] - angAttack[ia]))
    c2 = coeffsup[0] + (alfa - angAttack[ia]) * ((coeffsup[1] - coeffsup[0]) / (angAttack[sa] - angAttack[ia]))

    coeffFinal = c1 + (m - mach[im]) * ((c2 - c1) / (mach[sm] - mach[im]))
    return coeffFinal


def aero_uncertainty(h, M, alfa, hc, lb, ub, Mc):
    alfac = np.deg2rad(40)
    if h > hc:
        h = hc
    if h < 1.0:
        h = 1.0
    if M > Mc:
        M = Mc
    if alfa > alfac:
        alfa = alfac
    eps_aero = lb*(1-h/hc) + ub*(h/hc) + lb*(1-M/Mc) + ub*(M/Mc) + lb*(1-alfa/alfac) + ub*(alfa/alfac)
    return eps_aero


def aero_uncertain(M, alfa, cd, cl, v, rho, obj, h, hc, lb, ub, Mc, lhs_sample=None, extend=False):
    if np.isnan(v):
        v = 0
    elif np.isinf(v) or v > 1e6:
        v = 1e6
    elif v < -1e6:
        v = -1e6
    alfag = np.rad2deg(alfa)

    cL = coefCalc(cl, M, alfag, obj.mach, obj.angAttack)
    cD = coefCalc(cd, M, alfag, obj.mach, obj.angAttack)

    eps_aero = aero_uncertainty(h, M, alfa, hc, lb, ub, Mc)
    l = 0.5 * (v ** 2) * obj.wingSurf * rho * cL
    d = 0.5 * (v ** 2) * obj.wingSurf * rho * cD
    if lhs_sample is not None and extend:
        r = random.uniform(lhs_sample * 0.99, lhs_sample * 1.01)
    elif lhs_sample is not None and not extend:
        r = lhs_sample
    if lhs_sample is not None:

        l = l * (r * (1 + eps_aero - (1 - eps_aero)) + 1 - eps_aero)
        d = d * (r * (1 + eps_aero - (1 - eps_aero)) + 1 - eps_aero)
        #l = l * (lhs_sample * (1+eps_aero - (1-eps_aero)) + 1-eps_aero)
        #d = d * (lhs_sample * (1 + eps_aero - (1 - eps_aero)) + 1 - eps_aero)
    else:
        l = l * random.uniform(1-eps_aero, 1 + eps_aero)
        d = d * random.uniform(1 - eps_aero, 1 + eps_aero)

    #if np.isnan(l) or np.isnan(d):
    #    print('nan')
    return l, d

def aero_noUncertain(M, alfa, cd, cl, v, rho, obj):
    if np.isnan(v):
        v = 0
    elif np.isinf(v) or v > 1e6:
        v = 1e6
    elif v < -1e6:
        v = -1e6
    alfag = np.rad2deg(alfa)

    cL = coefCalc(cl, M, alfag, obj.mach, obj.angAttack)
    cD = coefCalc(cd, M, alfag, obj.mach, obj.angAttack)

    l = 0.5 * (v ** 2) * obj.wingSurf * rho * cL
    d = 0.5 * (v ** 2) * obj.wingSurf * rho * cD

    return l, d

##### PROPULSION MODELS

def thrust(presamb, mass, presv, spimpv, delta, tau, obj):
    nimp = 17
    nmot = 1
    thrx = nmot * (5.8e6 + 14.89 * obj.psl - 11.16 * presamb) * delta
    if presamb > obj.psl:
        presamb = obj.psl
        spimp = spimpv[-1]
    elif presamb <= obj.psl:
        for i in range(nimp):
            if presv[i] >= presamb:
                spimp = np.interp(presamb, [presv[i - 1], presv[i]], [spimpv[i - 1], spimpv[i]])
                break
    xcg = ((obj.xcgf - obj.xcg0) / (obj.m10 - obj.M0) * (mass - obj.M0) + obj.xcg0) * obj.lRef
    dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * obj.psl - presamb)
    if tau == 0:
        mommot = 0.0
        thrz = 0.0
        thrust = thrx
        deps = 0.0
    else:
        mommot = tau * dthr
        thrz = -tau * (2.5E+6 - 22 * obj.psl + 9.92 * presamb)
        thrust = np.sqrt(thrx ** 2 + thrz ** 2)
        deps = np.arctan(thrz / thrx)
    return thrust, deps, spimp, mommot


def vass(states, omega):
    """Function to evaluate the absolute velocity considering the earth rotation"""
    Re = 6371000
    v = states[0]
    chi = states[1]
    gamma = states[2]
    lam = states[4]
    h = states[5]

    vv = np.array((-v * np.cos(gamma) * np.cos(chi),
                   v * np.cos(gamma) * np.sin(chi),
                   -v * np.sin(gamma)))
    vv[0] = vv[0] + omega * np.cos(lam) * (Re + h)
    vela = np.sqrt(vv[0] ** 2 + vv[1] ** 2 + vv[2] ** 2)
    chiass = chi
    if vv[0] <= 0.0 or np.isnan(vv[0]):
        if abs(vv[0]) >= abs(vv[1]):
            chiass = np.arctan(abs(vv[1] / vv[0]))
            if vv[1] < 0.0 or np.isnan(vv[1]):
                chiass = -chiass
        elif abs(vv[0]) < abs(vv[1]):
            chiass = np.pi*0.5 - np.arctan(abs(vv[0] / vv[1]))
            if vv[1] < 0.0 or np.isnan(vv[1]):
                chiass = -chiass
    elif vv[0] > 0.0 or np.isinf(vv[0]):
        if abs(vv[0]) >= abs(vv[1]):
            chiass = np.pi - np.arctan((abs(vv[1]/vv[0])))
            if vv[1] < 0.0:
                chiass = - chiass
        elif abs(vv[0]) < abs(vv[1]):
            chiass = np.pi * 0.5 + np.arctan(abs(vv[0] / vv[1]))
            if vv[1] < 0.0:
                chiass = -chiass

    return vela, chiass


def dynamicsONLY_GPOnly(v, chi, gamma, lam, h, D, L, g, obj, sigma):
    dx = np.array((- D / obj.m - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) *
                   (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   (L * np.sin(sigma)) / (obj.m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) * np.tan(lam) *
                   (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) -
                   (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(gamma) * np.cos(chi),
                   (L * np.cos(sigma)) / (obj.m * v) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega *
                   np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) *
                   (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                   np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma)))
    return dx


def evaluate_ascent(individual, pset, **k):

    """This function is used to evalute the fitness function of the GP individuals for an ascent mission"""

    penalty = []
    falfa = gp.compile(individual[0], pset=pset)
    fdelta = gp.compile(individual[1], pset=pset)

    cl = k['kwargs']['cl']
    cd = k['kwargs']['cd']
    cm = k['kwargs']['cm']
    spimpv = k['kwargs']['spimpv']
    presv = k['kwargs']['presv']
    height_start = k['kwargs']['height_start']
    v_wind = k['kwargs']['v_wind']
    deltaH = k['kwargs']['deltaH']
    change_time = k['kwargs']['change_time']
    obj = k['kwargs']['obj']
    vfun = k['kwargs']['vfun']
    chifun = k['kwargs']['chifun']
    gammafun = k['kwargs']['gammafun']
    hfun = k['kwargs']['hfun']
    alfafun = k['kwargs']['alfafun']
    deltafun = k['kwargs']['deltafun']
    tfin = k['kwargs']['tfin']
    x_ini_h = k['kwargs']['x_ini_h']


    def sys(t, x, cl, cd, cm, spimpv, presv, height_start, v_wind, deltaH):
        v = x[0]
        chi = x[1]
        gamma = x[2]
        teta = x[3]
        lam = x[4]
        h = x[5]
        m = x[6]
        empty = False
        if height_start < h < height_start + deltaH:
            v = v - v_wind

        if h < 0.0 or np.isnan(h):
            penalty.append(h / obj.hmax)
            h = obj.hmin

        if m < obj.m10:
            penalty.append((m - obj.m10) / obj.M0)
            m = obj.m10
            empty = True

        ev = vfun(t) - v
        echi = chifun(t) - chi
        egamma = gammafun(t) - gamma
        eh = hfun(t) - h

        alfa = alfafun(t) + falfa(ev, echi, egamma, eh)
        delta = deltafun(t) + fdelta(ev, echi, egamma, eh)

        if delta > 1.0:
            penalty.append(delta - 1.0)
            delta = 1.0
        elif delta < 0 or empty is True:
            penalty.append(delta)
            delta = 0.0
        if alfa > obj.alfamax:
            penalty.append((alfa - obj.alfamax) / obj.alfamax)
            alfa = obj.alfamax
        elif alfa < obj.alfamin:
            penalty.append((alfa - obj.alfamin) / obj.alfamax)
            alfa = obj.alfamin

        Press, rho, c = isa_uncertain(h, obj)

        if np.isnan(v):
            M = 0
        elif np.isinf(v):
            M = 1e6 / c
        else:
            M = v / c

        L, D, MomA = aero_uncertain(M, alfa, cd, cl, cm, v, rho, m, obj, h)

        T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, 0.0, obj)

        eps = Deps + alfa
        g0 = obj.g0

        if h <= 0 or np.isnan(h):
            g = g0
        else:
            g = g0 * (obj.Re / (obj.Re + h)) ** 2

        q = 0.5 * rho * (v ** 2)
        ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
        az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

        if q > obj.MaxQ:
            penalty.append((np.nan_to_num(q) - obj.MaxQ) / obj.MaxQ)
        if ax > obj.MaxAx:
            penalty.append((np.nan_to_num(ax) - obj.MaxAx) / obj.MaxAx)
        if az > obj.MaxAz:
            penalty.append((np.nan_to_num(az) - obj.MaxAz) / obj.MaxAz)
        if az < -obj.MaxAz:
            penalty.append((obj.MaxAz - np.nan_to_num(az)) / obj.MaxAz)

        dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) *
                       (obj.Re + h) * np.cos(lam) * (
                                   np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                       - np.cos(gamma) * np.cos(chi) * np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
                                   np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                       - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(gamma) * np.cos(
                           chi),
                       (T * np.sin(eps) + L) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega
                       * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) *
                       (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                       -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                       np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                       v * np.sin(gamma),
                       -T / (obj.g0 * isp)))
        return dx

    Npoints = 500  # number of integration points
    t_max_int = 20  #
    sol, t_stop, stop_index = RK4(change_time, tfin, sys, Npoints, x_ini_h, t_max_int,
                                  args=(cl, cd, cm, spimpv, presv, height_start, v_wind, deltaH))

    v = sol[:stop_index, 0]
    chi = sol[:stop_index, 1]
    gamma = sol[:stop_index, 2]
    lam = sol[:stop_index, 4]
    h = sol[:stop_index, 5]
    tt = np.linspace(change_time, t_stop, Npoints)

    if h[-1] < 1.0:
        h[-1] = 1.0

    v_ass, chi_ass = vass(sol[stop_index - 1, :], obj.omega)

    v_orbit = np.sqrt(obj.GMe / (obj.Re + h[-1]))

    if np.cos(obj.incl) / np.cos(lam[-1]) > 1:
        chi_orbit = np.pi
    else:
        if np.cos(obj.incl) / np.cos(lam[-1]) < - 1:
            chi_orbit = 0.0
        else:
            chi_orbit = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam[-1]))

    if tt[-1] < tfin:
        penalty.append((tt[-1] - tfin) / tfin)

    vR = vfun(tt)
    chiR = chifun(tt)
    gammaR = gammafun(tt)
    hR = hfun(tt)

    err1 = (vR - v) / obj.vmax
    err2 = (chiR - chi) / obj.chimax
    err3 = (gammaR - gamma) / obj.gammamax
    err6 = (hR - h) / obj.hmax

    fit1 = simps(abs(err1), tt)
    fit2 = simps(abs(err2), tt)
    fit3 = simps(abs(err3), tt)
    fit6 = simps(abs(err6), tt)

    orbital_req = abs(hR[-1] - h[-1]) / obj.hmax + abs(v_ass - v_orbit) / obj.vmax + abs(
        chi_ass - chi_orbit) / obj.chimax + abs(gammaR[-1] - gamma[-1]) / obj.gammamax
    globalFit = np.array([fit1, fit2, fit3, fit6, orbital_req])
    FIT = np.sqrt(1 / len(globalFit) * sum(globalFit ** 2))
    if penalty != []:
        pen = np.sqrt(sum(np.array(penalty) ** 2))
        return [FIT, pen]
    else:
        return [FIT, 0.0]

def sys_init(t, x, cl, cd, spimpv, presv, obj, alfafun, deltafun, uncertain_aero, uncertain_atmo):
    """Function propagated to find initial conditions for GP evaluation."""
    v = x[0]
    chi = x[1]
    gamma = x[2]
    lam = x[4]
    h = x[5]
    m = x[6]

    alfa = float(alfafun(t))
    delta = float(deltafun(t))

    if m < obj.m10:
        m = obj.m10
        delta = 0.0
    if delta > 1.0:
        delta = 1.0
    elif delta < 0:
        delta = 0.0

    Press, rho, c, temp = isa_uncertain(h, obj, [], [], [], [], [], uncertain_atmo)

    if np.isnan(v):
        M = 0
    elif np.isinf(v):
        M = 1e6 / c
    else:
        M = v / c

    L, D, MomA = aero_uncertain(M, alfa, cd, cl, v, rho, obj, h, [], [], [], [], uncertain_aero)

    L = float(L)
    D = float(D)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, 0.0, obj)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = g0 * (obj.Re / (obj.Re + h)) ** 2

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) *
                   (obj.Re + h) * np.cos(lam) * (
                               np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   - np.cos(gamma) * np.cos(chi) * np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
                               np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                   - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(gamma) * np.cos(
                       chi),
                   (T * np.sin(eps) + L) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega
                   * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) *
                   (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                   -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma),
                   -T / (obj.g0 * isp)))
    return dx


def sys2GP_uncert(t, x, falfa, fsigma, cl, cd, vfun, chifun, gammafun, tetafun, lamfun, hfun, alfafun, sigmafun, obj,
                  lbt, ubt, lbp, ubp, lb, ub, hc, Mc, uncertain, uncertain_atmo, uncertain_aero, extend=False, uncertFuns=None):
    """Function propagated to test the GP control law with uncertainties in the physical models"""

    dx = dynamics(t, x, obj, cd, cl, lbt, ubt, lbp, ubp, lb, ub, hc, Mc, uncertain, uncertain_atmo,
             uncertain_aero, vfun, chifun, gammafun, tetafun, lamfun, hfun, alfafun, sigmafun, falfa, fsigma,
                  extend, uncertFuns)
    return dx


def sys2GP_NN(t, x, expr1, expr2, height_start, deltaH, v_wind, Control, model, obj, vfun, chifun, gammafun, thetafun,
           lamfun, hfun, alphafun, deltafun, pset, cl, cd, cm, presv, spimpv):

    """Function propagated to test the NN or GP control law with uncertainties in the physical models"""

    v = x[0]
    chi = x[1]
    gamma = x[2]
    theta = x[3]
    lam = x[4]
    h = x[5]
    m = x[6]

    if height_start <= h <= height_start + deltaH:
        v = v - v_wind

    if h < 0.0 or np.isnan(h):
        h = obj.hmin

    ev = vfun(t) - v
    echi = chifun(t) - chi
    egamma = gammafun(t) - gamma
    etheta = thetafun(t) - theta
    elam = lamfun(t) - lam
    eh = hfun(t) - h

    if Control is True:
        input = preprocessing.normalize(np.array([[t, v, chi, gamma, theta, lam, h, ev, echi, egamma, etheta, elam, eh]]))
        prediction = model.predict(input)[0]
        ex1, ex2 = optiutils.update_eqs([expr1, expr2], prediction)
        fAlpha = gp.compile(ex1, pset=pset)
        fDelta = gp.compile(ex2, pset=pset)
    else:
        fAlpha = gp.compile(expr1, pset=pset)
        fDelta = gp.compile(expr2, pset=pset)

    alfa = alphafun(t) + fAlpha(ev, echi, egamma, eh)
    delta = deltafun(t) + fDelta(ev, echi, egamma, eh)

    if m < obj.m10 or np.isnan(m):
        m = obj.m10
        delta = 0.0
    if delta > 1.0 or np.isinf(delta):
        delta = 1.0
    elif delta < 0 or np.isnan(delta):
        delta = 0.0
    if alfa > obj.alfamax or np.isinf(alfa):
        alfa = obj.alfamax
    elif alfa < obj.alfamin or np.isnan(alfa):
        alfa = obj.alfamin

    Press, rho, c = isa_uncertain(h, obj)

    if np.isnan(v):
        M = 0
    elif np.isinf(v):
        M = 1e6 / c
    else:
        M = v / c

    L, D, MomA = aero_uncertain(M, alfa, cd, cl, cm, v, rho, m, obj, h)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, 0.0, obj)

    eps = Deps + alfa
    g0 = obj.g0

    if h <= 0 or np.isnan(h):
        g = g0
    else:
        g = g0 * (obj.Re / (obj.Re + h)) ** 2

    dxdt = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) *
                   (obj.Re + h) * np.cos(lam) * (
                           np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   - np.cos(gamma) * np.cos(chi) * np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
                           np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                   - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(gamma) * np.cos(
                       chi),
                   (T * np.sin(eps) + L) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega
                   * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) *
                   (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                   -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma),
                   -T / (obj.g0 * isp)))
    return dxdt


def RK4(t_start, t_end, fun, Npoints, init_cond, t_max_int, args):
    """Runge-Kutta 4 with time integration limit. If t_elapsed > t_max_int then the integration stops"""
    int_start= time.time()
    t = np.linspace(t_start, t_end, Npoints)
    dt = t[1] - t[0]
    x = np.zeros((Npoints, len(init_cond)))
    x[0,:] = init_cond
    for i in range(Npoints - 1):
        int_end = time.time()
        if int_end - int_start > t_max_int:
            t_stop = t[i]
            stop_index = i
            break
        k1 = fun(t[i], x[i, :], *args)
        k2 = fun(t[i] + dt / 2, x[i, :] + 0.5 * dt * k1, *args)
        k3 = fun(t[i] + dt / 2, x[i, :] + 0.5 * dt * k2, *args)
        k4 = fun(t[i] + dt, x[i, :] + dt * k3, *args)
        x[i + 1, :] = x[i, :] + (1 / 6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        t_stop = t[i + 1]
        stop_index = i + 2
    return x, t_stop, stop_index


def prop_untilCviol(tfin, Npoints, x_ini, vfun, chifun, gammafun, tetafun, lamfun, hfun,th, cl, cd, alfafun, sigmafun,
                    obj, hc, lbt, ubt, lbp, ubp, lb, ub, Mc, uncertain_atmo, uncertain_aero, t_change, t_start, extend=False,
                    lhs_sample=None):
    t = t_start
    time = np.linspace(t, tfin, Npoints)
    dt = time[1] - time[0]
    c_viol = False
    traj = np.zeros((6, Npoints + 1))
    j = 0
    traj[:, j] = x_ini
    uncertain = False
    while t < tfin and not c_viol:
        if t > t_change:
            uncertain = True
            if traj[0, -1] > vfun(t) * (1 + th) or traj[0, -1] < vfun(t) * (1 - th) or traj[4, -1] > lamfun(t) * (
                    1 + th) \
                    or traj[4, -1] < lamfun(t) * (1 - th) or traj[5, -1] > hfun(t) * (1 + th) or traj[5, -1] < hfun(
                t) * (1 - th):
                c_viol = True
            if chifun(t) > 0:
                if traj[1, -1] > chifun(t) * (1 + th) or traj[1, -1] < chifun(t) * (1 - th):
                    c_viol = True
            else:
                if traj[1, -1] < chifun(t) * (1 + th) or traj[1, -1] > chifun(t) * (1 - th):
                    c_viol = True
            if gammafun(t) > 0:
                if traj[2, -1] > gammafun(t) * (1 + th) or traj[2, -1] < gammafun(t) * (1 - th):
                    c_viol = True
            else:
                if traj[2, -1] < gammafun(t) * (1 + th) or traj[2, -1] > gammafun(t) * (1 - th):
                    c_viol = True
            if tetafun(t) > 0:
                if traj[3, -1] > tetafun(t) * (1 + th) or traj[3, -1] < tetafun(t) * (1 - th):
                    c_viol = True
            else:
                if traj[3, -1] < tetafun(t) * (1 + th) or traj[3, -1] > tetafun(t) * (1 - th):
                    c_viol = True
        res = solve_ivp(propagate_OptwithUncert, [t, t + dt], x_ini, method='RK45',
                        args=(cl, cd, alfafun, sigmafun, obj, hc, lbt, ubt, lbp, ubp, lb, ub, Mc,
                              uncertain, uncertain_atmo, uncertain_aero, vfun, chifun, gammafun, tetafun, lamfun, hfun,
                              [], [], extend, lhs_sample))
        j += 1
        t += dt
        traj[:, j] = res.y[:, -1]
        x_ini = copy(res.y[:, -1])

    return traj[:, j-1], t-dt


def propagate_OptwithUncert(t, x, cl, cd, alfafun, sigmafun, obj, hc, lbt, ubt, lbp, ubp, lb, ub, Mc, uncertain,
                            uncertain_atmo, uncertain_aero, vfun, chifun, gammafun, tetafun, lamfun, hfun, falfa,
                            fsigma, extend=False, uncertFuns=None):

    dx = dynamics(t, x, obj, cd, cl, lbt, ubt, lbp, ubp, lb, ub, hc, Mc, False, uncertain_atmo,
             uncertain_aero, vfun, chifun, gammafun, tetafun, lamfun, hfun, alfafun, sigmafun, falfa, fsigma, extend, uncertFuns)
    return dx


def dynamics(t, states, obj, cd, cl, lbt, ubt, lbp, ubp, lb, ub, hc, Mc, uncertain, uncertain_atmo,
             uncertain_aero, vfun, chifun, gammafun, tetafun, lamfun, hfun, alfafun, sigmafun, falfa, fsigma,
             extend=False, uncertFuns=None):
    v = states[0]
    chi = states[1]
    gamma = states[2]
    teta = states[3]
    lam = states[4]
    h = states[5]

    if h < 0.0 or np.isnan(h):
        h = obj.hmin

    ev = (vfun(t) - v) / obj.vmax
    echi = (chifun(t) - chi) / obj.chimax
    egamma = (gammafun(t) - gamma) / obj.gammamax
    eteta = (tetafun(t) - teta) / obj.tetamax
    elam = (lamfun(t) - lam) / obj.lammax
    eh = (hfun(t) - h) / obj.hmax

    if falfa == []:
        alfa = alfafun(t)
        sigma = sigmafun(t)
    else:
        alfa = alfafun(t) + falfa(ev, echi, egamma, eteta, elam, eh)
        sigma = sigmafun(t) + fsigma(ev, echi, egamma, eteta, elam, eh)

    if sigma > obj.sigmamax:
        sigma = obj.sigmamax
    elif sigma < obj.sigmamin:
        sigma = obj.sigmamin
    if alfa > obj.alfamax:
        alfa = obj.alfamax
    elif alfa < obj.alfamin:
        alfa = obj.alfamin
    if v < obj.vmin:
        v = obj.vmin

    if uncertain_atmo is True:
        r_p = float(uncertFuns[0](t))
        r_t = float(uncertFuns[1](t))
        Press, rho, c = isa_uncertain(h, obj, hc, lbt, ubt, lbp, ubp, [r_p, r_t], uncertain_atmo)
    else:
        Press, rho, c = isa_uncertain(h, obj, hc, lbt, ubt, lbp, ubp, uncertain_atmo)

    rho = np.nan_to_num(rho)
    c = np.nan_to_num(c)

    if np.isnan(v):
        M = 0
    elif np.isinf(v):
        M = 1e6 / c
    else:
        M = v / c

    if uncertain_aero is True:
        r_a = float(uncertFuns[2](t))
        L, D = aero_uncertain(M, alfa, cd, cl, v, rho, obj, h, hc, lb, ub, Mc, r_a, uncertain_aero)
    else:
        L, D = aero_uncertain(M, alfa, cd, cl, v, rho, obj, h, hc, lb, ub, Mc, uncertain_aero)

    L = float(L)
    D = float(D)

    g0 = obj.g0

    if h <= 0 or np.isnan(h):
        g = g0
    else:
        g = g0 * (obj.Re / (obj.Re + h)) ** 2

    q = 0.5 * rho * (v ** 2)

    # Heat transfer
    Q = obj.Cfin * np.sqrt(rho) * (v ** 3) * (1 - 0.18 * (np.sin(obj.Lam) ** 2)) * np.cos(obj.Lam)

    # Load factor
    az = (D * np.sin(alfa) + L * np.cos(alfa)) / obj.m


    dx = dynamicsONLY_GPOnly(v, chi, gamma, lam, h, D, L, g, obj, sigma)
    return dx


def dynamicsONLY(v, chi, gamma, lam, h, D, L, g, obj, sigma):
    dx = np.array((- D / obj.m - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) *
                   (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   (L * np.sin(sigma)) / (obj.m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) * np.tan(lam) *
                   (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) -
                   (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(gamma) * np.cos(chi),
                   (L * np.cos(sigma)) / (obj.m * v) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega *
                   np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) *
                   (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                   np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma)))
    return dx


def evaluate_reentry(individual, pset, **k):
    """This function is used to evalute the fitness function of the GP individuals"""
    penalty = []

    falfa = gp.compile(individual[0], pset=pset)
    fsigma = gp.compile(individual[1], pset=pset)

    obj = k['kwargs']['obj']
    vfun = k['kwargs']['vfun']
    chifun = k['kwargs']['chifun']
    gammafun = k['kwargs']['gammafun']
    tetafun = k['kwargs']['tetafun']
    lamfun = k['kwargs']['lamfun']
    hfun = k['kwargs']['hfun']
    t_startGP = k['kwargs']['t_startGP']
    tfin = k['kwargs']['tfin']
    Npoints = k['kwargs']['Npoints']


    def sys(t, x, cl, cd, lbt, ubt, lbp, ubp, lb, ub, hc, Mc, uncertain, uncertain_atmo, uncertain_aero, obj, vfun,
            chifun, gammafun, tetafun, lamfun, hfun, alfafun, sigmafun, uncert_Funs=None, extend=False):
        v = x[0]
        chi = x[1]
        gamma = x[2]
        teta = x[3]
        lam = x[4]
        h = x[5]

        if h < 0.0 or np.isnan(h):
            penalty.append(h / obj.hmax)
            h = obj.hmin

        ev = (vfun(t) - v) / obj.vmax
        echi = (chifun(t) - chi) / obj.chimax
        egamma = (gammafun(t) - gamma) / obj.gammamax
        eteta = (tetafun(t) - teta) / obj.tetamax
        elam = (lamfun(t) - lam) / obj.lammax
        eh = (hfun(t) - h)/obj.hmax

        alfa = alfafun(t) + falfa(ev, echi, egamma, eteta, elam, eh)
        sigma = sigmafun(t) + fsigma(ev, echi, egamma, eteta, elam, eh)

        if sigma > obj.sigmamax:
            sigma = obj.sigmamax
        elif sigma < obj.sigmamin:
            sigma = obj.sigmamin
        if alfa > obj.alfamax:
            alfa = obj.alfamax
        elif alfa < obj.alfamin:
            alfa = obj.alfamin
        if v < obj.vmin:
            v = obj.vmin

        if uncertain and uncertain_atmo:
            if uncert_Funs is not None:
                r_p = float(uncert_Funs[0](t))
                r_t = float(uncert_Funs[1](t))
                Press, rho, c = isa_uncertain(h, obj, hc, lbt, ubt, lbp, ubp, [r_p, r_t], extend)
            else:
                Press, rho, c = isa_uncertain(h, obj, hc, lbt, ubt, lbp, ubp)
        else:
            Press, rho, c = isa_noUncertain(h, obj)

        rho = np.nan_to_num(rho)
        c = np.nan_to_num(c)

        if np.isnan(v):
            M = 0
        elif np.isinf(v):
            M = 1e6 / c
        else:
            M = v / c
        if uncertain and uncertain_aero:
            if uncert_Funs is not None:
                r_a = float(uncert_Funs[2](t))
                L, D = aero_uncertain(M, alfa, cd, cl, v, rho, obj, h, hc, lb, ub, Mc, r_a, extend)
            else:
                L, D = aero_uncertain(M, alfa, cd, cl, v, rho, obj, h, hc, lb, ub, Mc)
        else:
            L, D = aero_noUncertain(M, alfa, cd, cl, v, rho, obj)

        L = float(L)
        D = float(D)

        g0 = obj.g0

        if h <= 0 or np.isnan(h):
            g = g0
        else:
            g = g0 * (obj.Re / (obj.Re + h)) ** 2

        q = 0.5 * rho * (v ** 2)

        # Heat transfer
        Q = obj.Cfin * np.sqrt(rho) * (v**3) * (1-0.18*(np.sin(obj.Lam)**2)) * np.cos(obj.Lam)

        # Load factor
        az = (D * np.sin(alfa) + L * np.cos(alfa)) / obj.m

        if q > obj.Maxq:
            penalty.append((np.nan_to_num(q) - obj.Maxq) / obj.Maxq)
        if Q > obj.MaxQ:
            penalty.append((np.nan_to_num(Q) - obj.MaxQ) / obj.MaxQ)
        if az > obj.MaxAz:
            penalty.append((np.nan_to_num(az) - obj.MaxAz))
        if az < -obj.MaxAz:
            penalty.append((obj.MaxAz - np.nan_to_num(az)))

        dx = dynamicsONLY(v, chi, gamma, lam, h, D, L, g, obj, sigma)
        return dx


    sol, t_stop, stop_index = RK4(t_startGP, tfin, sys, Npoints, k['kwargs']['x_ini_h'], k['kwargs']['t_max_int'],
                                  args=(k['kwargs']['cl'], k['kwargs']['cd'], k['kwargs']['lbt'], k['kwargs']['ubt'], k['kwargs']['lbp'],
                                        k['kwargs']['ubp'], k['kwargs']['lb'], k['kwargs']['ub'], k['kwargs']['hc'], k['kwargs']['Mc'],
                                        k['kwargs']['uncertain'], k['kwargs']['uncertain_atmo'], k['kwargs']['uncertain_aero'],
                                        obj, vfun, chifun, gammafun, tetafun, lamfun, hfun, k['kwargs']['alfafun'],
                                        k['kwargs']['sigmafun'], k['kwargs']['uncert_Funs'], k['kwargs']['extend']))

    v = sol[:, 0]
    chi = sol[:, 1]
    gamma = sol[:, 2]
    teta = sol[:, 3]
    lam = sol[:, 4]
    h = sol[:, 5]
    tt = np.linspace(t_startGP, tfin, Npoints)

    if tt[-1] < tfin:
        penalty.append((tt[-1] - tfin) / tfin)

    vR = vfun(tt)
    chiR = chifun(tt)
    gammaR = gammafun(tt)
    tetaR = tetafun(tt)
    lamR = lamfun(tt)
    hR = hfun(tt)

    err1 = (vR - v) / obj.vmax
    err2 = (chiR - chi) / obj.chimax
    err3 = (gammaR - gamma) / obj.gammamax
    err4 = (tetaR - teta) / obj.tetamax
    err5 = (lamR - lam) / obj.lammax
    err6 = (hR - h) / obj.hmax
    err4Fin = (tetaR - teta)
    err5Fin = (lamR - lam)
    err6Fin = (hR - h) / obj.hmax

    fit1 = simps(abs(err1[-50:]), tt[-50:])
    fit2 = simps(abs(err2[-50:]), tt[-50:])
    fit3 = simps(abs(err3[-50:]), tt[-50:])
    fit4 = simps(abs(err4[-50:]), tt[-50:])
    fit5 = simps(abs(err5[-50:]), tt[-50:])
    fit6 = simps(abs(err6[-50:]), tt[-50:])

    fac_req = np.array([err6Fin[-1]/10, err5Fin[-1], err4Fin[-1]])
    globalFit = np.array([fit1, fit2, fit3, fit4, fit5, fit6])
    globalFit = np.hstack((globalFit/1e2, fac_req*1e3))
    FIT = np.sqrt((1 / len(globalFit)) * sum(globalFit ** 2))
    if penalty != []:
        pen = np.sqrt(1/len(penalty) * sum(np.array(penalty) ** 2))
        return [FIT, pen]
    else:
        return [FIT, 0.0]


def check_success(toolbox, hof, **kwargs):
    obj = kwargs['obj']
    success = False
    falfa = toolbox.compile(hof[-1][0])
    fsigma = toolbox.compile(hof[-1][1])

    solgp, tend, end_index = RK4(kwargs['t_startGP'], kwargs['tfin'], sys2GP_uncert, kwargs['Npoints'],
                                 kwargs['x_ini_h'], 200,
                                 args=(falfa, fsigma, kwargs['cl'], kwargs['cd'], kwargs['vfun'], kwargs['chifun'],
                                       kwargs['gammafun'], kwargs['tetafun'], kwargs['lamfun'], kwargs['hfun'],
                                       kwargs['alfafun'], kwargs['sigmafun'], kwargs['obj'], kwargs['lbt'], kwargs['ubt'],
                                       kwargs['lbp'], kwargs['ubp'], kwargs['lb'], kwargs['ub'], kwargs['hc'],
                                       kwargs['Mc'], kwargs['uncertain'], kwargs['uncertain_atmo'],
                                       kwargs['uncertain_aero'], kwargs['extend'], kwargs['uncert_Funs']))

    tetaout = solgp[:, 3]
    lamout = solgp[:, 4]
    hout = solgp[:, 5]

    if (obj.hend - obj.hend_tol <= hout[-1] and hout[-1] <= obj.hend + obj.hend_tol) and \
            (obj.lamend - obj.lamend_tol <= lamout[-1] and lamout[-1] <= obj.lamend + obj.lamend_tol) and \
            (obj.tetaend - obj.tetaend_tol <= tetaout[-1] and tetaout[-1] <= obj.tetaend + obj.tetaend_tol):
        success = True
        print('Success')
    else:
        if hout[-1] < obj.hend - obj.hend_tol:
            diff_h = obj.hend - obj.hend_tol - hout[-1]
            print("Height below FAC box of : {} m".format(diff_h))
        elif hout[-1] > obj.hend + obj.hend_tol:
            diff_h = hout[-1] - (obj.hend + obj.hend_tol)
            print("Height above FAC box of : {} m".format(diff_h))
        if lamout[-1] < obj.lamend - obj.lamend_tol:
            diff_lam = obj.lamend - obj.lamend_tol - lamout[-1]
            print("Lambda below FAC box of : {} deg".format(np.rad2deg(diff_lam)))
        elif lamout[-1] > obj.lamend + obj.lamend_tol:
            diff_lam = lamout[-1] - (obj.lamend + obj.lamend_tol)
            print("Lambda above FAC box of : {} deg".format(np.rad2deg(diff_lam)))
        if tetaout[-1] < obj.tetaend - obj.tetaend_tol:
            diff_teta = obj.tetaend - obj.tetaend_tol - tetaout[-1]
            print("Teta below FAC box of : {} deg".format(np.rad2deg(diff_teta)))
        elif tetaout[-1] > obj.tetaend + obj.tetaend_tol:
            diff_teta = tetaout[-1] - (obj.tetaend + obj.tetaend_tol)
            print("Teta above FAC box of : {} deg".format(np.rad2deg(diff_teta)))

    return success