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

def dynamics(t, x, obj, u):
    """
        Author(s): Francesco Marchetti
        email: francesco.marchetti@strath.ac.uk/framarc93@gmail.com

        Function describing the equation of motion

        Attributes:
            t: float
                time
            x: array
                state variables
            obj: class
                a class containing the plant's parameters
            u: float
                control variable
        Return:
            x_out: array
                the array of updated state variables
        """
    m = obj.m
    l = obj.l
    M = obj.M
    g = obj.g
    x1dot = x[1]
    x2dot = (-(m**2)*(l**2)*g*np.cos(x[2])*np.sin(x[2])+m*(l**2)*(m*l*(x[3]**2)*np.sin(x[2]))+m*(l**2)*u)/(m*(l**2)*(M+m*(1-np.cos(x[2])**2)))
    x3dot = x[3]
    x4dot = ((m+M)*m*g*l*np.sin(x[2])-m*l*np.cos(x[2])*(m*l*(x[3]**2)*np.sin(x[2]))-m*l*np.cos(x[2])*u)/(m*(l**2)*(M+m*(1-np.cos(x[2])**2)))
    x_out = np.array((x1dot, x2dot, x3dot, x4dot))
    return x_out



