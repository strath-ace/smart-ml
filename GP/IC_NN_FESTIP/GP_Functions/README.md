This folder contains the scripts used by the GP:
* GP_PrimitiveSet.py contains the primitives used by the GP
* GP_Functions.py contains all the functions and classes used by both the IGP and SGP

Regarding the IGP functions, the main improvements introduced in comparison to a standard Genetic Programming algorithm are:
* The individuals in the population are divided in different categories according to their length.
* Such categories are considered in the InclusiveTournament selection mechanism, which performs a double tournament selection on each category to maintain diversity.
* The crossover operations are performed between individuals of different categories in order to maintain diversity in the population
* Two fitness functions are used. The first one it the primary objective function while the second one is a penalty function used to take into account the imposed constraints. All the selection mechanisms and the hall of fame insertion works accordingly to this penalty measure.
* The adopted evolutionary strategy consists in first find feasible individuals (penalty=0) and then evolve the population towards individuals with the lowest objective function (first fitness function).

This code was developed thanks to concepts and ideas discussed in [1], [2] and [3].

## References

1. Entropy-Driven Adaptive Representation. J. P. Rosca. Proceedings of the Workshop on Genetic Programming: From Theory to Real-World Applications, 23-32. 1995
2. Exploration and Exploitation in Evolutionary Algorithms: a Survey. M. Crepinsek, S.Liu, M. Mernik. ACM Computer Surveys. 2013
3. Theoretical and Numerical Constraint-Handling Techniques used with Evolutionary Algorithms: A Survey of the State of the Art. C. A. Coello Coello. Computer Methods in Applied Mechanics and Engineering 191. 2002
