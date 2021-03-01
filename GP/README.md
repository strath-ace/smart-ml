# Genetic Programming

This software has been released under the MPL-2.0 and LGPL-3.0 licenses. It includes source code from DEAP(https://github.com/deap/deap) which is released under the LGPL-3.0 license.

## Description

Algorithms for the intelligent control of launch vehicles using primarly Genetic Programming along with other Machine Learning techiniques.

Please refer to the subfolder for detailed explanation of the algorithms and their usage:
* **IGP** (Inclusive Genetic Programming) contains the new GP heuristic which was introduced in [2] and the formally formulated in [3].
* **IntGPC** (Intelligent Genetic Programming Control) contains the algorithm developed for the work presented in reference [1]. It is an implementation of the online use of GP to perform intelligent control applied on the Goddard ascent vehicle.
* **IntHGPNNC** (Intelligent Hybrid Genetic Programming Neural Network Control) contains the algorithm developed for the work presented in reference [2]. It is an implementation of the online use of a NN to optimize online a control law produced using GP tp perform intelligent control. It is applied on the ascent trajectory of the FESTIP vehicle.
*  **SGP** (Standard Genetic Programming) contains a standard implementation of GP used to compare the results of the IGP.

## References
1. Marchetti, F., Minisci, E., Riccardi, A. . Towards Intelligent Control via Genetic Programming. 2020 International Joint Conference on Neural Networks (IJCNN) (2020)
2. Marchetti, F., Minisci, E. . A Hybrid Neural Network-Genetic Programming Intelligent Control Approach. In: B. Filipič, E. Minisci, M. Vasile (eds.) Bioinspired Optimization Methods and Their Applications. BIOMA 2020. Springer, Cham, Brussels (2020) 
3. Marchetti F., Minisci E. . Inclusive Genetic Programming. In: Genetic Programming. EuroGP 2021. Lecture Notes in Computer Science. Springer, Cham (2021)
