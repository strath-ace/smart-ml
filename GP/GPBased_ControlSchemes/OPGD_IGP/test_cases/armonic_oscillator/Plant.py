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

import numpy as np

class Oscillator:
    """
    Class containing the parameters for the harmonic oscillator test case
    """
    def __init__(self):
        self.m = 1  # mass
        self.k = 2  # spring stiffness
        self.a = 1
        self.b = -1
        self.c = 0.3
        self.h = 0.5
        self.x0 = np.array(([4, 0]))  # initial conditions
        self.xf = np.array(([0, 0]))
        self.Qt = np.diag([1, 1])
        self.Qz = np.diag([5, 5])
        self.Qu = np.array(([1]))
        self.t0 = 0
        self.tf = 10
        self.dt = 0.05
        self.eq_f = ['x2',
                     '-(k/m)*x1-(c/m)*(a*x1**2+b)*x2+u1/m']
        self.eq_g = ['h*Qz1*(x1-x1r)**2 + h*Qz2*(x2-x2r)**2 + h*Qu*u1**2']
        self.states_dict = {'x1': 'X', 'x2': 'V'}
        self.cont_dict = {'u1': 'u1'}
        self.vars_order = ['eX', 'eV']
        self.GPinput_dict = {'eX':'sub(X, x1r)', 'eV':'sub(V, x2r)'}
        self.str_symbols = ['a', 'k', 'c', 'b', 'm', 'Qu', 'h', 'Qz1', 'Qz2', 'x1r', 'x2r']
        self.val_symbols = np.array(([self.a, self.k, self.c, self.b, self.m, self.Qu[0], 0.5, self.Qz[0,0], self.Qz[1,1], self.xf[0], self.xf[1]]))
        self.n_states = 2
        self.Npoints = int((self.tf - self.t0) / self.dt) + 1  # number of propagation points
        self.t_points = np.linspace(self.t0, self.tf, self.Npoints)  # time points

        # Adam parameters

        self.alfa = 0.01
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8
