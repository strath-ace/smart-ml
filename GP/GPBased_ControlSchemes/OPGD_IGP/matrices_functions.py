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
import sympy
import GP.GP_Algorithms.GP_PrimitiveSet as gpprim
import operator


def ETA_function(x, Qt, xf):
    """
    Function used to create the ETA sensitivity matrix

    Attributes:
        x: array
            state variables
        Qt: array
            Qt matrix
        xf: array
            reference final states
    """
    ETA = (x-xf)*np.diag(Qt)
    return ETA


def fill_sensitivity_matrix(M, z, gamma, symbols):
    """
    Function used to fill the sensitivity matrices

    Attributes:
        M: list
            sensitivity matrix
        z: array
            states
        gamma: array
            optimization variables
        symbols: list
            symbols used in the matrix
    Return:
        M_filled: array
            filled sensitivity matrix
    """
    if len(np.shape(M)) == 2:
        M_filled = np.zeros((len(M), len(M[0])))
        for i in range(len(M)):
            for j in range(len(M[0])):
                M_filled[i, j] = M[i][j](*z, *gamma, *symbols)
    else:
        M_filled = np.zeros((len(M)))
        for i in range(len(M)):
            M_filled[i] = M[i](*z, *gamma, *symbols)
    return M_filled

def define_matrices(eq_f, eq_g, n_weights, n_states, symbols):
    """
    Function used to define the sensitivity matrices

    Attributes:
        eq_f: list
            list containing the strings of the symbolic equation of motion
        eq_g: string
            string with the symbolic equation of the g fitness function
        n_weights: integer
            number of optimization variables
        n_states: integer
            number of state variables
        symbols: list
            list of symbols used in the equations

    Returns:
        A: list
            A matrix
        B: list
            B matrix
        PSI: list
            PSI matrix
        PHI: list
            PHI matrix
    """
    w = [sympy.symbols('w%d' % i) for i in range(1, n_weights + 1)]
    x = [sympy.symbols('x%d' % i) for i in range(1, n_states + 1)]


    A = []
    B = []
    PSI = []
    PHI = []
    for i in range(len(eq_f)):
        parsed_expr = sympy.sympify(eq_f[i], locals={'TriAdd':gpprim.TriAdd, 'add':operator.add, 'sub':operator.sub,
                                                     'mul':operator.mul, 'TriMul':gpprim.TriMul, 'cos':sympy.cos,
                                                     'sin':sympy.sin})
        eqsA = []
        eqsB = []
        for j in range(n_states):
            eqsA.append(sympy.lambdify([*x, *w, *symbols], sympy.diff(parsed_expr, x[j])))
        A.append(eqsA)
        for j in range(len(w)):
            eqsB.append(sympy.lambdify([*x, *w, *symbols], sympy.diff(parsed_expr, w[j])))
        B.append(eqsB)
    parsed_expr = sympy.sympify(eq_g[0], locals={'TriAdd': gpprim.TriAdd, 'add': operator.add, 'sub': operator.sub,
                                                 'mul': operator.mul, 'TriMul': gpprim.TriMul, 'cos':sympy.cos,
                                                 'sin':sympy.sin})
    for i in range(len(w)):
        PSI.append(sympy.lambdify([*x, *w, *symbols], sympy.diff(parsed_expr, w[i])))
    for i in range(len(x)):
        PHI.append(sympy.lambdify([*x, *w, *symbols], sympy.diff(parsed_expr, x[i])))
    return A, B, PSI, PHI