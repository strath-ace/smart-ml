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
from scipy.integrate import simps
import GP.GPBased_ControlSchemes.utils.integration_schemes as int_schemes
from GP.GPBased_ControlSchemes.OPGD_IGP.test_cases.inverted_pendulum.dynamics import dynamics
from GP.GPBased_ControlSchemes.OPGD_IGP.test_cases.inverted_pendulum.Plant import Pendulum

"""This script perform the trajectory propagation with the reference control law"""

if __name__=='__main__':

    obj = Pendulum()

    x = np.zeros((obj.Npoints, obj.n_states))
    u = np.zeros(obj.Npoints)

    x[0, :] = obj.x0
    vv = x[0, :] - obj.xf
    u[0] = -obj.K @ (x[0,:] - obj.xf)
    failure = False
    for i in range(obj.Npoints - 1):
        sol_forward = int_schemes.RK4(obj.t_points[i], obj.t_points[i + 1], dynamics, 2, x[i, :],
                                      args=(obj, u[i]))
        if np.isnan(sol_forward[-1, :]).any() or np.isinf(sol_forward[-1, :]).any():
            failure = True
            break
        else:
            x[i + 1, :] = sol_forward[-1, :]
            u[i + 1] = -obj.K @ (x[i+1,:] - obj.xf)
    g = np.zeros(obj.Npoints)
    for i in range(obj.Npoints):
        g[i] = 0.5 * ((x[i, :] - obj.xf).T @ obj.Qz @ (x[i, :] - obj.xf) + np.array(([[u[i]]])).T @ obj.Qu @ np.array(([[u[i]]])))
    int_g = simps(g, obj.t_points)
    h = 0.5 * ((x[-1, :] - obj.xf).T @ obj.Qt @ (x[-1, :] - obj.xf))
    FIT = int_g + h
    print('Fitness reference ', FIT)
    np.save('x_ref.npy', x)
    np.save('u_ref.npy', u)
    np.save('time_points.npy', obj.t_points)