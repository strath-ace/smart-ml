# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2023 University of Strathclyde and Author ------
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

"""This script contains the functions with the surrogate models"""

import numpy as np
from numpy import sin, cos, exp, tanh


def atm_uncertainty(h, hc, lbp, ubp):
    """
    Function used to evaluate the epsilon for the atmospheric uncertainty

    Attribute:
        h: float
            altitude
        hc: float
            max altitude
        lbp: float
            lower bound uncertainty
        ubp: float
            upper bound uncertainty
    Return:
        eps: float
            bounding parameter
    """
    if hasattr(h, '__len__'):
        h[h > hc] = hc
        h[h < 0] = 0
    else:
        if h > hc:
            h = hc
        if h < 0:
            h = 0
    eps = lbp*(1-h/hc) + ubp*(h/hc)
    return eps


def aero_uncertainty(h, M, alfa, hc, Mc, lb, ub):
    """
    Function used to evaluate the epsilon for the aerodynamic uncertainty

    Attribute:
        h: float
            altitude
        M: float
            mach
        alfa: float
            angle of attack
        hc: float
            max altitude
        Mc: float
            max Mach
        lb: float
            lower bound uncertainty
        ub: float
            upper bound uncertainty
    Return:
        eps_aero: float
            bounding parameter
    """
    alfac = np.deg2rad(40)
    if hasattr(h, '__len__'):
        h[h > hc] = hc
        h[h < 0] = 0
        M[M > Mc] = Mc
        alfa[alfa > alfac] = alfac
    else:
        if h > hc:
            h = hc
        if h < 0:
            h = 0
        if M > Mc:
            M = Mc
        if alfa > alfac:
            alfa = alfac
    eps_aero = lb*(1-h/hc) + ub*(h/hc) + lb*(1-M/Mc) + ub*(M/Mc) + lb*(1-alfa/alfac) + ub*(alfa/alfac)
    return eps_aero


def control_variation(errs, lb, ub):
    """
    Function used to evaluate the epsilon for the control output

    Attribute:
        errs: array
            errors
        lb: float
            lower bound
        ub: float
            upper bound
    Return:
        eps_contr: float
            bounding parameter
    """
    eps_contr = sum(lb * (1 - errs) + ub * errs)
    return eps_contr



def rho_surrogate(h):
    return 0.11952631818219178*tanh(0.000000017226276837875536599168*h**2)-1.3436621687042358*tanh(84.76115648*tanh(tanh(tanh(0.000001*h))))+1.2244531895809327


def rho_surrogate_uncertain(h, r_dens, eps_dens):
    return (0.11952631818219178*tanh(0.000000017226276837875536599168*h**2)-1.3436621687042358*tanh(84.76115648*tanh(tanh(tanh(0.000001*h))))+1.2244531895809327)*(2*r_dens*eps_dens+1-eps_dens)


def c_surrogate(h):
    return 109.90582761074495*tanh(exp(1.0017342136916225458999986130223*tanh(0.00001*h) - 0.000088437103321551204842235477552057*h)) - 16.599597569453845*cos(tanh(sin(0.000088284*h))) - 6.1194528190911*exp(0.000063532*h) - 54.22689549690425*cos(tanh(sin(sin(sin(7.554770747980552489136383842648*cos(sin(tanh(0.000066398*h)))))))) + 731.28009692197*exp(0.0000000001*h**2) - 407.83209815481246

def c_surrogate_uncertain(h, r_c, eps_c):
    return (109.90582761074495*tanh(exp(1.0017342136916225458999986130223*tanh(0.00001*h) - 0.000088437103321551204842235477552057*h)) - 16.599597569453845*cos(tanh(sin(0.000088284*h))) - 6.1194528190911*exp(0.000063532*h) - 54.22689549690425*cos(tanh(sin(sin(sin(7.554770747980552489136383842648*cos(sin(tanh(0.000066398*h)))))))) + 731.28009692197*exp(0.0000000001*h**2) - 407.83209815481246)*(2*r_c*eps_c+1-eps_c)


def cl_surrogate(M, alfa):
    return 7.894646515205631*alfa-0.23277133648367332236995058699788*sin(alfa)-2.438368416648305*alfa*tanh(M-tanh(alfa))+5.78334756014982*alfa*tanh(M-0.9224)-3.763210105438917*alfa*exp(tanh(M-0.9224))-0.011243987799501937

def cl_surrogate_uncertain(M, alfa, r_a, eps_a):
    return (7.894646515205631*alfa-0.23277133648367332236995058699788*sin(alfa)-2.438368416648305*alfa*tanh(M-tanh(alfa))+5.78334756014982*alfa*tanh(M-0.9224)-3.763210105438917*alfa*exp(tanh(M-0.9224))-0.011243987799501937)*(2*r_a*eps_a+1-eps_a)


def cd_surrogate(M, alfa):
    return 14.929618050502855*cos(alfa*tanh(M-alfa))-16.425697655429648*cos(alfa)-2.450234201101943*exp(alfa-M)+2.3836550136005212*exp(2.4183*alfa**2-M)+1.5726997378619498


def cd_surrogate_uncertain(M, alfa, r_a, eps_a):
    return (14.929618050502855*cos(alfa*tanh(M-alfa))-16.425697655429648*cos(alfa)-2.450234201101943*exp(alfa-M)+2.3836550136005212*exp(2.4183*alfa**2-M)+1.5726997378619498)*(2*r_a*eps_a+1-eps_a)

