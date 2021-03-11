# Genetic Progamming for Intelligent Control

This software has been released under the MPL-2.0 and LGPL-3.0 licenses. It includes source code from DEAP(https://github.com/deap/deap) which is released under the LGPL-3.0 license.

## Description
The code contained in this directory is the one used to obtain the results presented in [1] at the WCCI/IJCNN 2020 conference. 
To reproduce the results run the files "MAIN_CdScenario.py", "MAIN_GustScenario.py" and "MAIN_DensityScenario.py".

The developed code is based on the DEAP library and makes use of Genetic Programming to evaluate online a control law for a 
Goddard Rocket test case in 3 different failure scenarios.

The main improvements introduced in comparison to a standard Genetic Programming algorithm are:
* The individuals in the population are divided in different categories according to their length.
* Such categories are considered in the InclusiveTournament selection mechanism, which performs a double tournament selection on each category to maintain diversity.
* The crossover operations are performed between individuals of different categories in order to maintain diversity in the population
* Two fitness functions are used. The first one it the primary objective function while the second one is a penalty function used to take into account the imposed constraints. All the selection mechanisms and the hall of fame insertion works accordingly to this penalty measure.
* The adopted evolutionary strategy consists in first find feasible individuals (penalty=0) and then evolve the population towards individuals with the lowest objective function (first fitness function).

This code was developed thanks to concepts and ideas discussed in [2], [3] and [4].
  
The folder Goddard_Models contains the used reference trajectory, and the necessary models to run the scripts.

# Citation
If you use any part of the code, please cite [1].


## Main libraries used (Python):
  * DEAP 1.2.2: https://github.com/deap/deap
  * Scipy 1.3.2
  * Numpy 1.18.1
  * Matplotlib 3.1.1
  * Multiprocessing 3.8.3

## References
1. F. Marchetti, E. Minisci, A. Riccardi. Towards Intelligent Control via Genetic Programming. The 2020 IEEE International Joint Conference on Neural Network Proceedings. 2020
2. Entropy-Driven Adaptive Representation. J. P. Rosca. Proceedings of the Workshop on Genetic Programming: From Theory to Real-World Applications, 23-32. 1995
3. Exploration and Exploitation in Evolutionary Algorithms: a Survey. M. Crepinsek, S.Liu, M. Mernik. ACM Computer Surveys. 2013
4. Theoretical and Numerical Constraint-Handling Techniques used with Evolutionary Algorithms: A Survey of the State of the Art. C. A. Coello Coello. Computer Methods in Applied Mechanics and Engineering 191. 2002
