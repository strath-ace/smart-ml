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

"""File containing the modification introduced into the DEAP library and new functions and classes to enhance the
performances

References:
[1] Entropy-Driven Adaptive Representation. J. P. Rosca. Proceedings of the Workshop on Genetic Programming: From Theory
to Real-World Applications, 23-32. 1995
[2] https://pastebin.com/QKMhafRq
"""

from operator import eq, mul, truediv
from functools import wraps, partial
from collections import Sequence
import numpy as np
from copy import deepcopy
from deap import tools, gp
import random
import sys

#######################################################################################################################
"""                                         NEW FUNCTIONS AND CLASSES                                               """
#######################################################################################################################

def InclusiveTournament(individuals, mu, to_mate):
    """
    Author(s): Francesco Marchetti
    email: francesco.marchetti@strath.ac.uk

    Rationale behind InclusiveTournament: a double tournament selection is performed in each category, so to maintain
     diversity. Double Tournamet is used so to avoid bloat. An exploited measure is used to point out when a category is
     completely exploited. For example, if in a category are present only 4 individuals, the tournament will be
     performed at maximum 4 times in that category. This to avoid a spreading of clones of the same individuals."""

    organized_pop, good_indexes = subset_diversity(individuals)
    chosen = []
    exploited = np.zeros((len(good_indexes)))
    enough = False
    if to_mate is True:
        enough = True
    j = 0
    count = 0
    while len(chosen) < mu:
        if j > len(good_indexes) - 1:
            j = 0
        i = good_indexes[j]

        if exploited[j] < len(organized_pop["cat{}".format(i)]):
            selected = selDoubleTournament(organized_pop["cat{}".format(i)], 1, 1.6, enough, fitness_first=True)
            chosen.append(selected)
            if selected.fitness.values[-1] == 0:
                count += 1
            exploited[j] += 1
        j += 1
        choice = random.random()
        if choice > 0.8:
            enough = True
        elif choice <= 0.8 and to_mate is False:
            enough = False
        if count >= 2 * mu / 3:
            enough = True

    if enough is True and to_mate is False:
        print("Greed prevention")
    elif enough is True and to_mate is True:
        print("Mating constraints on feasibles")

    return chosen


################### POPULATION CLASS  ###############################

class POP(object):
    '''This class is used to collect data about a population. Used at the beginning for the selection of the initial
        population. Entropy measure comes from [1]. This class is used to evaluate different parameters regarding
        the population

    Attributes:
        items : list
            population to which apply the POP class
        lens : float
            array containing the lengths of each individual in the population
        max : int
            maximum length of the individuals in the population
        min : int
            minimum length of the individuals in the population
        maxDiff : int
            difference in length between the biggest and the smallest individual
        categories : dict
            dictionary containing the different categories in which the population is divided
        indexes : int
            array contatining the indexes of the filled categories
        entropy : float
            entropy measure of the population calculated according to [1]

    Methods:
        output_stats():
            print the statistics of the considered population
    '''

    def __init__(self, population):
        '''
        Parameters:
            population : list
                list of individuals
        '''
        self.items = list()
        self.lens = np.zeros(len(population))
        for i in range(len(population)):
            item = deepcopy(population[i])
            self.items.append(item)
            self.lens[i] = len(population[i][0]) + len(population[i][1])
        self.min = int(min(self.lens))
        self.max = int(max(self.lens))
        self.maxDiff = self.max - self.min
        self.categories, self.indexes = subset_diversity(population)
        self.entropy = -sum(self.categories["distribution"]["percentage"] * np.log(self.categories["distribution"]["percentage"]))
        #self.output_stats()   # remove comment to print stats

    def output_stats(self):

        print("\n")
        print("--------------------------- Population STATISTICS -----------------------------------")
        print("------------------ Min len: {}, Max len: {}, Max Diff: {} ----------------------".format(self.min, self.max, self.maxDiff))
        print("------------------------------- Entropy: {0:.3f} --------------------------------------".format(self.entropy))
        print("-- Distribution (%): {} ------------".format(self.categories["distribution"]["percentage"] * 100))
        print("-------------------------------------------------------------------------------------")


def subset_feasible(population):
    """Function used to create a subset of feasible individuals from the input population"""
    sub_pop = []
    for ind in population:
        if ind.fitness.values[-1] == 0:
            sub_pop.append(ind)
    return sub_pop


def subset_unfesible(population):
    """Function used to create a subset of unfeasible individuals from the input population"""
    sub_pop = []
    for ind in population:
        if ind.fitness.values[0] < 10 and ind.fitness.values[1] < 30 and ind.fitness.values[-1] < 10 and ind.fitness.values[-1] != 0:
            sub_pop.append(ind)
    return sub_pop


def subset_diversity(population):
    """Function used to divide the individuals in the input population into cat_number categories. The division into
    categories is done according to the length of the individuals in the population"""

    cat_number = 10  # here the number of categories is selected
    lens = []
    categories = {}
    distribution = []
    distr_stats = {}
    invalid_ind = []

    for ind in population:
        lens.append((len(ind[0]) + len(ind[1])))
    cat = np.linspace(min(lens), max(lens), cat_number + 1)
    useful_ind = np.linspace(0, len(cat) - 2, len(cat) - 1)

    for i in range(len(cat) - 1):
        categories["cat{}".format(i)] = []
    for ind in population:
        for i in range(len(cat) - 1):
            if (len(ind[0]) + len(ind[1])) >= cat[i] and (len(ind[0]) + len(ind[1])) <= cat[i + 1]:
                categories["cat{}".format(i)].append(ind)

    for i in range(len(cat) - 1):
        if categories["cat{}".format(i)] == []:
            invalid_ind.append(i)
        distribution.append(len(categories["cat{}".format(i)]))
    distr_stats["individuals"] = distribution
    distr_stats["percentage"] = np.array(distribution) / len(population)
    categories["distribution"] = distr_stats
    if invalid_ind != []:
        useful_ind = np.delete(useful_ind, invalid_ind, 0)
    return categories, np.asarray(useful_ind, dtype=int)


#######################################################################################################################
"""                                         MODIFIED FUNCTIONS FROM DEAP LIBRARY                                    """
#######################################################################################################################

############## MODIFIED BLOAT CONTROL #########################################
def staticLimit(key, max_value):
    """This is a modification of the staticLimit function implemented in the DEAP library in order to
    deal with individual composed by multiple trees


    Original description:
        Implement a static limit on some measurement on a GP tree, as defined
        by Koza in [Koza1989]. It may be used to decorate both crossover and
        mutation operators. When an invalid (over the limit) child is generated,
        it is simply replaced by one of its parents, randomly selected.

        This operator can be used to avoid memory errors occuring when the tree
        gets higher than 90 levels (as Python puts a limit on the call stack
        depth), because it can ensure that no tree higher than this limit will ever
        be accepted in the population, except if it was generated at initialization
        time.

        :param key: The function to use in order the get the wanted value. For
                    instance, on a GP tree, ``operator.attrgetter('height')`` may
                    be used to set a depth limit, and ``len`` to set a size limit.
        :param max_value: The maximum value allowed for the given measurement.
        :returns: A decorator that can be applied to a GP operator using \
        :func:`~deap.base.Toolbox.decorate`

        .. note::
           If you want to reproduce the exact behavior intended by Koza, set
           *key* to ``operator.attrgetter('height')`` and *max_value* to 17.

        .. [Koza1989] J.R. Koza, Genetic Programming - On the Programming of
            Computers by Means of Natural Selection (MIT Press,
            Cambridge, MA, 1992)

        """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                check = max(ind[0], ind[1], key=key)
                if key(ind[0]) == key(check):
                    j = 0
                else:
                    j = 1
                if key(check) > max_value:
                    new_inds[i][j] = random.choice(keep_inds)[j]
            return new_inds

        return wrapper

    return decorator


####################  MODIFIED SELECTION ALGORITHM ##########################

def selDoubleTournament(individuals, k, parsimony_size, enough, fitness_first):
    """This is a modification of the xselDoubleTournament function from [2] which itself is a modification of
     the selDoubleTournament function implemented in the DEAP library. The modification is done in order to deal with
     individual composed by multiple trees and with the 2 fitness functions used in this work fitness functions. This

    Original description:

    Tournament selection which use the size of the individuals in order to discriminate good solutions.
    This kind of tournament is obviously useless with fixed-length representation, but has been shown to significantly
    reduce excessive growth of individuals, especially in GP, where it can be used as a bloat control technique (see
    [Luke2002fighting]_). This selection operator implements the double tournament technique presented in this paper.
    The core principle is to use a normal tournament selection, but using a special sample function to select aspirants,
    which is another tournament based on the size of the individuals. To ensure that the selection pressure is not too
    high, the size of the size tournament (the number of candidates evaluated) can be a real number between 1 and 2.
    In this case, the smaller individual among two will be selected with a probability*size_tourn_size*/2. For instance,
    if *size_tourn_size* is set to 1.4, then the smaller individual will have a 0.7 probability to be selected.
    .. note::
        In GP, it has been shown that this operator produces better results
        when it is combined with some kind of a depth limit.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param parsimony_size: The number of individuals participating in each size tournament. This value has to be a real
                            number in the range [1,2], see above for details.
    :param fitness_first: Set this to True if the first tournament done should be the fitness one (i.e. the fitness
                          tournament producing aspirants for the size tournament). Setting it to False will behaves as
                          the opposite (size tournament feeding fitness tournaments with candidates). It has been shown
                          that this parameter does not have a significant effect in most cases (see [Luke2002fighting]_).
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.
    .. [Luke2002fighting] Luke and Panait, 2002, Fighting bloat with
        nonparametric parsimony pressure
    """

    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, enough, k, select):
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            if len(individuals) == 1:
                return random.choice(individuals)
            else:
                prob = parsimony_size / 2.
                ind1, ind2 = select(individuals, enough, k=2)
                lind1 = sum([len(gpt) for gpt in ind1])       # Modified part
                lind2 = sum([len(gpt) for gpt in ind2])       # Modified part
                if lind1 > lind2:
                    ind1, ind2 = ind2, ind1
                elif lind1 == lind2:
                    # random selection in case of a tie
                    prob = 0.5

                # Since size1 <= size2 then ind1 is selected
                # with a probability prob
                chosen.append(ind1 if random.random() < prob else ind2)

            return chosen[0]

    def _fitTournament(individuals, enough, k):
        chosen = []
        for _ in range(k):
            a1, a2 = random.sample(individuals, 2)  # Modified part. Always select 2 individuals for the tournament
            if enough is False:
                if a1.fitness.values[-1] == 0 and a2.fitness.values[-1] == 0:      # Modified part
                    if sum(a1.fitness.wvalues) > sum(a2.fitness.wvalues):          #
                        chosen.append(a1)                                          #
                    else:                                                          #
                        chosen.append(a2)                                          #
                elif a1.fitness.values[-1] == 0 and a2.fitness.values[-1] != 0:    #
                    chosen.append(a1)                                              #
                elif a1.fitness.values[-1] != 0 and a2.fitness.values[-1] == 0:    #
                    chosen.append(a2)                                              #
                elif a1.fitness.values[-1] != 0 and a2.fitness.values[-1] != 0:    #
                    if a1.fitness.values[-1] < a2.fitness.values[-1]:              #
                        chosen.append(a1)                                          #
                    else:                                                          #
                        chosen.append(a2)                                          #
            else:                                                                  #
                if sum(a1.fitness.wvalues) > sum(a2.fitness.wvalues):              #
                    chosen.append(a1)                                              #
                else:                                                              #
                    chosen.append(a2)                                              # Modified part
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament)
        return _sizeTournament(individuals, enough, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, enough, k)


def selBest(individuals):
    """
    This is a modification of the selBest function implemented in the DEAP library in order to deal with individual
    composed by multiple trees and with the 2 fitness functions used in this work fitness functions

    Original description:

    Select the *k* best individuals among the input *individuals*. The list returned contains references to the
    input *individuals*.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list containing the k best individuals.

    """
    best = individuals[0]
    choice = random.random()
    for ind in individuals:                                                  # Modified Part
        if ind.fitness.values[-1] == 0 and best.fitness.values[-1] == 0:     #
            if ind.fitness.values[0] < best.fitness.values[0]:               #
                best = ind                                                   #
        elif ind.fitness.values[-1] == 0 and best.fitness.values[-1] != 0:   #
            best = ind                                                       #
        elif ind.fitness.values[-1] != 0 and best.fitness.values[-1] != 0:   #
            if choice > 0.9:                                                 #
                if ind.fitness.values[-1] < best.fitness.values[-1]:         #
                    best = ind                                               #
            else:                                                            #
                if sum(ind.fitness.values) < sum(best.fitness.values):       #
                    best = ind                                               # Modified Part
    return best

######################## MODIFIED FITNESS CLASS ###########################

class FitnessMulti(object):
    """Fitness class modified from DEAP library. Only modification is the sum inserted in the comparison functions.

    Original Description:
    The fitness is a measure of quality of a solution. If *values* are provided as a tuple, the fitness is initialized
    using those values, otherwise it is empty (or invalid).
    :param values: The initial values of the fitness as a tuple, optional.
    Fitnesses may be compared using the ``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``. The comparison of those operators
    is made lexicographically. Maximization and minimization are taken care off by a multiplication between the
    :attr:`weights` and the fitness :attr:`values`. The comparison can be made between fitnesses of different size,
    if the fitnesses are equal until the extra elements, the longer fitness will be superior to the shorter.
    Different types of fitnesses are created in the :ref:`creating-types` tutorial.
    .. note::
       When comparing fitness values that are **minimized**, ``a > b`` will return :data:`True` if *a* is **smaller**
       than *b*.
    """

    weights = None
    """The weights are used in the fitness comparison. They are shared among
    all fitnesses of the same type. When subclassing :class:`Fitness`, the
    weights must be defined as a tuple where each element is associated to an
    objective. A negative weight element corresponds to the minimization of
    the associated objective and positive weight to the maximization.

    .. note::
        If weights is not defined during subclassing, the following error will
        occur at instantiation of a subclass fitness object:

        ``TypeError: Can't instantiate abstract <class Fitness[...]> with
        abstract attribute weights.``
    """

    wvalues = ()
    """Contains the weighted values of the fitness, the multiplication with the
    weights is made when the values are set via the property :attr:`values`.
    Multiplication is made on setting of the values for efficiency.

    Generally it is unnecessary to manipulate wvalues as it is an internal
    attribute of the fitness used in the comparison operators.
    """

    def __init__(self, values=()):
        if self.weights is None:
            raise TypeError("Can't instantiate abstract %r with abstract "
                            "attribute weights." % (self.__class__))

        if not isinstance(self.weights, Sequence):
            raise TypeError("Attribute weights of %r must be a sequence."
                            % self.__class__)

        if len(values) > 0:
            self.values = values

    def getValues(self):
        return tuple(map(truediv, self.wvalues, self.weights))

    def setValues(self, values):
        try:
            self.wvalues = tuple(map(mul, values, self.weights))
        except TypeError:
            _, _, traceback = sys.exc_info()
            raise TypeError("Both weights and assigned values must be a "
                            "sequence of numbers when assigning to values of "
                            "%r. Currently assigning value(s) %r of %r to a "
                            "fitness with weights %s."
                            % (self.__class__, values, type(values),
                               self.weights)).with_traceback(traceback)

    def delValues(self):
        self.wvalues = ()

    values = property(getValues, setValues, delValues,
                      ("Fitness values. Use directly ``individual.fitness.values = values`` "
                       "in order to set the fitness and ``del individual.fitness.values`` "
                       "in order to clear (invalidate) the fitness. The (unweighted) fitness "
                       "can be directly accessed via ``individual.fitness.values``."))

    def dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.

        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self.wvalues) != 0

    def __hash__(self):
        return hash(self.wvalues)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        return sum(self.wvalues) <= sum(other.wvalues)  # Modified Part

    def __lt__(self, other):
        return sum(self.wvalues) < sum(other.wvalues)  # Modified Part

    def __eq__(self, other):
        return sum(self.wvalues) == sum(other.wvalues)  # Modified Part

    def __ne__(self, other):
        return not self.__eq__(other)

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.

        It assumes that the elements in the :attr:`values` tuple are
        immutable and the fitness does not contain any other object
        than :attr:`values` and :attr:`weights`.
        """
        copy_ = self.__class__()
        copy_.wvalues = self.wvalues
        return copy_

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
        return "%s.%s(%r)" % (self.__module__, self.__class__.__name__,
                              self.values if self.valid else tuple())

############################# MODIFIED HALL OF FAME ###################################################################

class HallOfFame(object):
    """
    Modified HallOfFame class taken from the DEAP library. The introduced modifications allow for:
        - a inverted classification (the last individual is the best) in comparison with the original
        - individuals are inserted in the hall of fame according to the following scheme:
            1 - for_feasible is checked, if is True, the best individuals are those with a penalty = 0 (second
                fitness function) and the lowest first fitness function
                1.1 - the individuals with a penalty = 0 are prioritized and then are compared on the value of the first
                      fitness function; the one with the lowest value are inserted.
                1.2 - the individuals with a penalty !=0 are compared on the sum of both the fitness. The ones with the lowest
                      sum are inserted
            2 - if for_feasible is False, the sum of both fitness is considered and the ones with the lowest sum
                are inserted

    Original description:
    The hall of fame contains the best individual that ever lived in the
    population during the evolution. It is lexicographically sorted at all
    time so that the first element of the hall of fame is the individual that
    has the best first fitness value ever seen, according to the weights
    provided to the fitness at creation time.
    The insertion is made so that old individuals have priority on new
    individuals. A single copy of each individual is kept at all time, the
    equivalence between two individuals is made by the operator passed to the
    *similar* argument.
    :param maxsize: The maximum number of individual to keep in the hall of
                    fame.
    :param similar: An equivalence operator between two individuals, optional.
                    It defaults to operator :func:`operator.eq`.
    The class :class:`HallOfFame` provides an interface similar to a list
    (without being one completely). It is possible to retrieve its length, to
    iterate on it forward and backward and to get an item or a slice from it.

    """

    def __init__(self, maxsize, similar=eq):
        self.maxsize = maxsize
        self.keys = list()
        self.items = list()
        self.similar = similar

    def shuffle(self):
        """
        Author: Francesco Marchetti
        Method used to shuffle the order of the individuals in the hall of fame
        """
        arr_start = deepcopy(self.items)
        arr_end = []
        while len(arr_start) > 0:
            ind = random.randint(0, len(arr_start) - 1)
            arr_end.append(arr_start[ind])
            arr_start.pop(ind)
        return arr_end

    def update(self, population, for_feasible):
        """Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        if len(self) == 0 and self.maxsize != 0 and len(population) > 0:
            # Working on an empty hall of fame is problematic for the
            # "for else"
            self.insert(population[0], for_feasible)

        if for_feasible is True:                                                                                        # Modified Part
            for ind in population:
                if ind.fitness.values[-1] == 0.0:                                                                       # Modified Part
                    if self[0].fitness.values[-1] == 0.0:                                                               # Modified Part
                        if sum(ind.fitness.values) < sum(self[0].fitness.values) or len(self) < self.maxsize:           # Modified Part
                            for hofer in self:
                                # Loop through the hall of fame to check for any
                                # similar individual
                                if self.similar(ind, hofer):
                                    break
                            else:
                                # The individual is unique and strictly better than
                                # the worst
                                if len(self) >= self.maxsize:
                                    self.remove(0)                                                                      # Modified Part
                                self.insert(ind, for_feasible)
                    else:
                        for hofer in self:
                            # Loop through the hall of fame to check for any
                            # similar individual
                            if self.similar(ind, hofer):
                                break
                        else:
                            # The individual is unique and strictly better than
                            # the worst
                            if len(self) >= self.maxsize:
                                self.remove(0)                                                                          # Modified Part
                            self.insert(ind, for_feasible)
                elif (sum(ind.fitness.values) < sum(self[0].fitness.values)) or len(self) < self.maxsize:               # Modified Part
                    for hofer in self:
                        # Loop through the hall of fame to check for any
                        # similar individual
                        if self.similar(ind, hofer):
                            break
                    else:
                        # The individual is unique and strictly better than
                        # the worst
                        if len(self) >= self.maxsize:
                            self.remove(0)                                                                              # Modified Part
                        self.insert(ind, for_feasible)
        else:                                                                                                           # Modified Part
            for ind in population:                                                                                      # Modified Part
                if ind.fitness.values[0] < 10 and ind.fitness.values[1] < 40 and sum(ind.fitness.values) < sum(self[0].fitness.values) or len(self) < self.maxsize:   # Modified Part
                    for hofer in self:
                        # Loop through the hall of fame to check for any
                        # similar individual
                        if self.similar(ind, hofer):
                            break
                    else:
                        # The individual is unique and strictly better than
                        # the worst
                        if len(self) >= self.maxsize:
                            self.remove(0)                                                                              # Modified Part
                        self.insert(ind, for_feasible)

    def insert(self, item, for_feasible):
        """Insert a new individual in the hall of fame using the
        :func:`~bisect.bisect_right` function. The inserted individual is
        inserted on the right side of an equal individual. Inserting a new
        individual in the hall of fame also preserve the hall of fame's order.
        This method **does not** check for the size of the hall of fame, in a
        way that inserting a new individual in a full hall of fame will not
        remove the worst individual to maintain a constant size.

        :param item: The individual with a fitness attribute to insert in the
                     hall of fame.
        """

        def bisect_right(a, x, lo=0, hi=None):
            """
            Modified bisect_right function from the python bisect library.

            Original description:

            Return the index where to insert item x in list a, assuming a is sorted.
            The return value i is such that all e in a[:i] have e <= x, and all e in
            a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
            insert just after the rightmost x already there.
            Optional args lo (default 0) and hi (default len(a)) bound the
            slice of a to be searched.
            """

            if lo < 0:
                raise ValueError('lo must be non-negative')
            if hi is None:
                hi = len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                # must indentify 4 cases: if both are feasible, if the new is feasible and the one in the list is not,
                # viceversa and if both are infeasible
                if for_feasible is True:                                                                                # Modified Part
                    # 1st case: both are feasible
                    if x.values[-1] == 0 and a[mid].values[-1] == 0:                                                    # Modified Part
                        if sum(x.values) < sum(a[mid].values):                                                          # Modified Part
                            hi = mid
                        else:
                            lo = mid + 1
                    # 2nd case: value to insert is feasible, the one in the list is not
                    elif x.values[-1] == 0 and a[mid].values[-1] != 0:                                                  # Modified Part
                        hi = mid
                    # 3rd case: value to insert is not feasible, the one in the list is feasible
                    elif x.values[-1] != 0 and a[mid].values[-1] == 0:                                                  # Modified Part
                        lo = mid + 1
                    # 4th case: both are infeasible
                    elif x.values[-1] != 0 and a[mid].values[-1] != 0:                                                  # Modified Part
                        if x.values[-1] < a[mid].values[-1]:                                                            # Modified Part
                            hi = mid
                        else:
                            lo = mid + 1
                else:                                                                                                   # Modified Part
                    if sum(x.values) < sum(a[mid].values):                                                              # Modified Part
                        hi = mid
                    else:
                        lo = mid + 1
            return lo

        item = deepcopy(item)
        i = bisect_right(self.keys, item.fitness)
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, item.fitness)

    def remove(self, index):
        """Remove the specified *index* from the hall of fame.

        :param index: An integer giving which item to remove.
        """
        del self.keys[len(self) - (index % len(self) + 1)]
        del self.items[index]

    def clear(self):
        """Clear the hall of fame."""
        del self.items[:]
        del self.keys[:]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __str__(self):
        return str(self.items)

##############################  MODIFIED ALGORITHMS  ##################################################################

def varOrFeasible(population, toolbox, lambda_, sub_div, good_indexes_original, cxpb, mutpb, limit_size):
    """
    Function to perform crossover,mutation or reproduction operations based on varOr function from DEAP library:

    Parameters:
        population : list
            A list of individuals to vary.
        toolbox : class, deap.base.Toolbox
            contains the evolution operators.
        lambda_ : int
            The number of children to produce
        sub_div : dict
            categories in which the population is divided
        good_indexes_original : int
            array contating the indexes of the filled categories
        cxpb : float
            The probability of mating two individuals.
        mutpb : float
            The probability of mutating an individual.
        limit_size : int
            size limit used to accept or not the mutation performed on an individual

    """

    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    sub_pop = subset_feasible(population)

    len_subpop = len(sub_pop)

    if sub_pop == []:
        print("Exploring for feasible individuals. Mutpb: {}, Cxpb:{}".format(mutpb, cxpb))
    else:
        if cxpb < 0.65:
            mutpb = mutpb - 0.01
            cxpb = cxpb + 0.01
        print("\n")
        print("{}/{} ({}%) FEASIBLE INDIVIDUALS".format(len_subpop, len(population), round(len(sub_pop) / len(population) * 100, 2)))
        print("Mutpb: {}, Cxpb:{}".format(mutpb, cxpb))

    good_indexes = list(good_indexes_original)
    good_list = list(good_indexes)
    for _ in range(lambda_):
        op_choice = random.random()
        cat = np.zeros((2))  ### selection of 2 different categories for crossover
        for i in range(2):
            if not good_list:
                good_list = list(good_indexes)
            used = random.choice(good_list)
            cat[i] = used
            good_list.remove(used)
        if op_choice < cxpb: # select individuals for crossover
            ind1, ind2 = map(toolbox.clone, [selBest(sub_div["cat{}".format(int(cat[0]))]), random.choice(
                sub_div["cat{}".format(int(cat[1]))])])  # select the best individual from the first category and a random one from the second
            while ind1.fitness.values[0] == ind2.fitness.values[0] and ind1.fitness.values[1] == ind2.fitness.values[1]:  # repeat category selection and mating to avoid clones
                cat = np.zeros(2)
                for i in range(2):
                    if not good_list:
                        good_list = list(good_indexes)
                    used = random.choice(good_list)
                    cat[i] = used
                    good_list.remove(used)
                ind1, ind2 = map(toolbox.clone,
                                 [selBest(sub_div["cat{}".format(int(cat[0]))]),
                                  random.choice(sub_div["cat{}".format(int(cat[1]))])])

            ind1, ind2 = toolbox.mate(ind1, ind2)  # Apply crossover
            if sum(ind1.fitness.wvalues) > sum(ind2.fitness.wvalues):  # add to the offspring the individual with the best fitness
                del ind1.fitness.values
                offspring.append(ind1)
            else:
                del ind2.fitness.values
                offspring.append(ind2)
        elif op_choice < cxpb + mutpb:
            ind = toolbox.clone(selBest(population))  # select best individual from the population
            old = ind
            ind, = toolbox.mutate(ind) # Apply mutation
            if sum(old.fitness.values) <= sum(ind.fitness.values) and len(ind) > limit_size: # accept mutated individual if has a better fitness or is smaller than the original
                ind = old
            del ind.fitness.values
            offspring.append(ind)
        else:  # Apply reproduction
            if len_subpop >= 1:
                offspring.append(random.choice(sub_pop))
            else:
                offspring.append(selBest(population))  # copy in the offspring the best individual in the population

    return offspring, len_subpop, cxpb, mutpb


def eaMuPlusLambdaTol(population, toolbox, mu, cxpb, mutpb, ngen, tol, limit_size, stats=None, halloffame=None, verbose=__debug__):
    """
    Modification of eaMuPlusLambda function from DEAP library. Modifications include:
        - use of tolerance value for the first fitness function below which the evolution is stopped
        - reseeding of the population if for len(min_fit_history) the best fitness function remain approximately
          constant (1e-6 interval) or every 200 generations and if there are at least 10 feasible (penalty = 0)
          individuals in the population

    Original Description:


    This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population, for_feasible=True)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    min_fit = np.array(logbook.chapters["fitness"].select("min"))
    mh = 0
    min_fit_history = np.zeros((30))
    min_fit_history[mh] = sum(min_fit[0])
    # Begin the generational process
    gen = 1
    len_feas = 0
    while gen < ngen + 1 and (min_fit[-1][0] > tol or min_fit[-1][-1] > 0.0):

        if (abs(min_fit_history[0] - min_fit_history[-1] < 1e-6) or gen % 200 == 0) and len_feas > 10:
            print("\n")
            print("----------------------------- RESEEDING -------------------------------")
            print("\n")
            mutpb = 0.7 # reset mutation and crossover rates
            cxpb = 0.2
            chosen = toolbox.select(population, int(len(population) / 3), to_mate=False)
            old_entropy = 0
            for i in range(200):
                pop = POP(toolbox.population(n=len(population) - len(chosen)))
                if pop.entropy > old_entropy:  # and len(pop.indexes) == len(pop.categories) - 1:
                    best_pop = pop.items
                    old_entropy = pop.entropy

            invalid_ind = [ind for ind in best_pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            population = chosen + best_pop
            min_fit_history = np.zeros((30))
            mh = 0

        sub_div, good_index = subset_diversity(population)
        offspring, len_feas, cxpb, mutpb = varOrFeasible(population, toolbox, mu, sub_div, good_index, cxpb, mutpb, limit_size)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring, for_feasible=True)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu, to_mate=False)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        min_fit = np.array(logbook.chapters["fitness"].select("min"))
        if mh == len(min_fit_history):
            mh = 0
            min_fit_history = np.zeros((30))
        min_fit_history[mh] = sum(min_fit[-1])
        mh += 1
        if verbose:
            print(logbook.stream)
        gen += 1

    return population, logbook

############################ MODIFIED STATISTICS FUNCTIONS  #########################################################

def Min(pop):
    """
    The old Min function from the DEAP library was returning incorrect data in case of multiobjective fitness function.
    The stats weren't about one individual but it was printing the minimum value found for each objective separately, also
    if they didn't belong to the same individual.
    """
    min = pop[0]
    w = np.array([-0.5, -1.0])
    for ind in pop:
        if ind[-1] == 0:
            if min[-1] == 0 and sum(ind[0:2]*w) > sum(min[0:2]*w):
                min = ind
            elif min[-1] != 0:
                min = ind
        elif ind[-1] < min[-1]:
            min = ind
    return min

def Max(inds):
    max = inds[0]
    for fit in inds:
        if fit[-1] == 0:
            if max[-1] == 0 and sum(fit[0:2]) > sum(max[0:2]):
                max = fit
        elif sum(fit) > sum(max):
            max = fit
    return max

########################### GENETIC OPERATORS FOR MULTIPLE TREE OUTPUT   #####################################

def xmate(ind1, ind2):
    """From [2] and modified"""
    ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2


def xmut(ind, expr, unipb, shrpb, inspb, psetR, psetT):
    """From [2] and modified. Addedd several mutations possibilities."""
    choice = random.random()

    if choice < unipb:
        indx1 = gp.mutUniform(ind[0], expr, pset=psetR)
        ind[0] = indx1[0]
        indx2 = gp.mutUniform(ind[1], expr, pset=psetT)
        ind[1] = indx2[0]
    elif unipb <= choice < unipb + shrpb:
        indx1 = gp.mutShrink(ind[0])
        ind[0] = indx1[0]
        indx2 = gp.mutShrink(ind[1])
        ind[1] = indx2[0]
    elif unipb + shrpb <= choice < unipb + shrpb + inspb:
        indx1 = gp.mutInsert(ind[0], pset=psetR)
        ind[0] = indx1[0]
        indx2 = gp.mutInsert(ind[1], pset=psetT)
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
    return ind,


