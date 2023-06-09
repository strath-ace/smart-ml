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
    k = obj.k
    m = obj.m
    a = obj.a
    b  = obj.b
    c = obj.c
    xdot = x[1]
    vdot = - (k / m) * x[0] - (c / m) * (a * x[0] ** 2 + b) * x[1] + u / m
    x_out = np.array((xdot, vdot))
    return x_out


