# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2021 University of Strathclyde and Author ------
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
File containing the recombination operators used by the IGP and SGP

References:
[1] https://pastebin.com/QKMhafRq
"""

import random
from deap import gp

########################### GENETIC OPERATORS FOR MULTIPLE TREE OUTPUT   #####################################
def xmateSimple(ind1, ind2):
    """From [1] """
    i1 = random.randrange(len(ind1))
    i2 = random.randrange(len(ind2))
    ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
    return ind1, ind2


def xmutSimple(ind, expr, pset):
    """From [1] """
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr, pset)
    ind[i1] = indx[0]
    return ind,


def xmate(ind1, ind2):
    """From [1] and modified"""
    ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2


def xmut(ind, expr, unipb, shrpb, inspb, pset, creator):
    """
    Author(s): Francesco Marchetti
    email: francesco.marchetti@strath.ac.uk

    From [1] and modified. Added several mutations possibilities.
    """
    choice = random.random()
    try:
        if type(ind[0]) == creator.SubIndividual:
            if hasattr(pset, '__len__'):
                if choice < unipb:
                    indx1 = gp.mutUniform(ind[0], expr, pset=pset[0])
                    ind[0] = indx1[0]
                    indx2 = gp.mutUniform(ind[1], expr, pset=pset[1])
                    ind[1] = indx2[0]
                elif unipb <= choice < unipb + shrpb:
                    indx1 = gp.mutShrink(ind[0])
                    ind[0] = indx1[0]
                    indx2 = gp.mutShrink(ind[1])
                    ind[1] = indx2[0]
                elif unipb + shrpb <= choice < unipb + shrpb + inspb:
                    indx1 = gp.mutInsert(ind[0], pset=pset[0])
                    ind[0] = indx1[0]
                    indx2 = gp.mutInsert(ind[1], pset=pset[1])
                    ind[1] = indx2[0]
                else:
                    choice2 = random.random()
                    if choice2 < 0.5:
                        indx1 = gp.mutEphemeral(ind[0], "all")
                        ind[0] = indx1[0]
                        indx2 = gp.mutEphemeral(ind[1], "all")
                        ind[1] = indx2[0]
                    else:
                        indx1 = gp.mutEphemeral(ind[0], "one")
                        ind[0] = indx1[0]
                        indx2 = gp.mutEphemeral(ind[1], "one")
                        ind[1] = indx2[0]
            else:
                if choice < unipb:
                    indx1 = gp.mutUniform(ind[0], expr, pset=pset)
                    ind[0] = indx1[0]
                    indx2 = gp.mutUniform(ind[1], expr, pset=pset)
                    ind[1] = indx2[0]
                elif unipb <= choice < unipb + shrpb:
                    indx1 = gp.mutShrink(ind[0])
                    ind[0] = indx1[0]
                    indx2 = gp.mutShrink(ind[1])
                    ind[1] = indx2[0]
                elif unipb + shrpb <= choice < unipb + shrpb + inspb:
                    indx1 = gp.mutInsert(ind[0], pset=pset)
                    ind[0] = indx1[0]
                    indx2 = gp.mutInsert(ind[1], pset=pset)
                    ind[1] = indx2[0]
                else:
                    choice2 = random.random()
                    if choice2 < 0.5:
                        indx1 = gp.mutEphemeral(ind[0], "all")
                        ind[0] = indx1[0]
                        indx2 = gp.mutEphemeral(ind[1], "all")
                        ind[1] = indx2[0]
                    else:
                        indx1 = gp.mutEphemeral(ind[0], "one")
                        ind[0] = indx1[0]
                        indx2 = gp.mutEphemeral(ind[1], "one")
                        ind[1] = indx2[0]
    except AttributeError:
        if choice < unipb:
            indx1 = gp.mutUniform(ind, expr, pset=pset)
            ind = indx1[0]
        elif unipb <= choice < unipb + shrpb:
            indx1 = gp.mutShrink(ind)
            ind = indx1[0]
        elif unipb + shrpb <= choice < unipb + shrpb + inspb:
            indx1 = gp.mutInsert(ind, pset=pset)
            ind = indx1[0]
        else:
            choice2 = random.random()
            if choice2 < 0.5:
                indx1 = gp.mutEphemeral(ind, "all")
                ind = indx1[0]
            else:
                indx1 = gp.mutEphemeral(ind, "one")
                ind = indx1[0]
    return ind,