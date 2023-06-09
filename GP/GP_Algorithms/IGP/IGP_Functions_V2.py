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

"""
File containing the modification introduced into the DEAP library and new functions and classes to enhance the
performances.

This is the latest version of the functions used by the IGP.

References:
[1] Entropy-Driven Adaptive Representation. J. P. Rosca. Proceedings of the Workshop on Genetic Programming: From Theory
to Real-World Applications, 23-32. 1995
[2] https://pastebin.com/QKMhafRq
"""
import multiprocess
import numpy as np
from copy import deepcopy, copy
import random
from functools import partial, wraps
from deap import tools, gp
from operator import eq, attrgetter
import _pickle as cPickle

#######################################################################################################################
"""                                         NEW FUNCTIONS AND CLASSES                                               """
#######################################################################################################################

#######################################################################################################################
"""                                   MODIFIED FUNCTIONS FROM DEAP LIBRARY                                          """
#######################################################################################################################



############## MODIFIED BLOAT CONTROL #########################################


######################## MODIFIED SELECTION MECHANISMS ###########################
def xselDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first):
    """
    From [2]
    """
    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            prob = parsimony_size / 2.
            ind1, ind2 = select(individuals, k=2)

            lind1 = sum([len(gpt) for gpt in ind1])
            lind2 = sum([len(gpt) for gpt in ind2])
            if lind1 > lind2:
                ind1, ind2 = ind2, ind1
            elif lind1 == lind2:
                # random selection in case of a tie
                prob = 0.5

            # Since size1 <= size2 then ind1 is selected
            # with a probability prob
            chosen.append(ind1 if random.random() < prob else ind2)

        return chosen

    def _fitTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            aspirants = select(individuals, k=fitness_size)
            chosen.append(max(aspirants, key=attrgetter("fitness")))
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament, select=tools.selRandom)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, k, tsize)



####################### MODIFIED HALL OF FAME #############################

##############################  MODIFIED EVOLUTIONARY STRATEGIES  ##############################

########################### GENETIC OPERATORS FOR MULTIPLE TREE OUTPUT   #####################################










