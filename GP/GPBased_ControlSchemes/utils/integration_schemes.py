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

def RK4(t_start, t_end, fun, Npoints, init_cond, args):
    """
    Runge-Kutta 4 integration scheme
    """
    t = np.linspace(t_start, t_end, Npoints)
    dt = t[1] - t[0]
    x = np.zeros((Npoints, len(init_cond)))
    x[0,:] = init_cond
    for i in range(Npoints-1):
        k1 = fun(t[i], x[i, :], *args)
        k2 = fun(t[i] + dt / 2, x[i, :] + 0.5 * dt * k1, *args)
        k3 = fun(t[i] + dt / 2, x[i, :] + 0.5 * dt * k2, *args)
        k4 = fun(t[i] + dt, x[i, :] + dt * k3, *args)
        x[i + 1, :] = x[i, :] + (1 / 6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return x