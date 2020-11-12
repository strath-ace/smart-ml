# Hybrid Neural Network-GeneticProgramming Intelligent Control Approach

This software has been released under the MPL-2.0 and LGPL-3.0 licenses. It includes source code from DEAP(https://github.com/deap/deap) which is released under the LGPL-3.0 license.

## Description
The code contained in this directory is the one used to obtain the results presented in [1] at the BIOMA 2020 conference. 

To reproduce the results from scratches run the code in the following order:
1) Inside the folder 1_OfflineCreationGPLaw:
* The scripts IGP_OfflineLawCreation.py and SGP_OfflineLawCreation.py creates 10 different control laws each on the disturbance scenarios contained in Datasets/GP_creationSet.npy, using the Inclusive Genetic Programming (IGP) and the Standard Genetic Prgramming (SGP).
* The script Test_GPLawRobustness.py tests the created control laws on the 500 different disturbance scenarios contained in Datasets/training_points.npy .
* The folder ResultsBIOMA2020 contains the control laws generated for the work presented in [1].
2) Inside the folder 2_Optimization_GPLaw:
* The script OfflineGPLawTuner.py optimize the GP control law created at the previous step, using an optimization algorithm contained in the scipy.optimize.minimze library. In [1] BFGS and NM were used. The results of the optimization are stored in a dataset used to train the NN for the next step of the process. The datasets produced for [1] are the files Datasets/dataset_forNN_500samplesTEST_1percent_BFGS_hof4.npy and Datasets/dataset_forNN_500samplesTEST_1percent_NM_hof4.npy.
* The script TreeTunerUtils.py contains the functions used by the script OfflineGPLawTuner.py.
3) Inside the folder 3_NN+GP_Control:
* The script OnlineControl_NN.py performs the control simulation on the disturbance scenarios contained in Datasets/TestSetNN.npy, using simultaneously the NN which always optimizes the GP control law and the GP control law non optimized. The NN can be either trained from scratches or the model produced for [1] can be loaded. 

The folder Datasets contains:
* GP_creationSet.npy: the disturbance scenarios used to create the GP control law offline.
[Datasets/GP_creationSet.png]
* training_points.npy: the disturbance scenarios used to optimize the GP control law 
[Datasets/training_points.png]
* TestSetNN.npy: the disturbance scenarios used to test the controller
[Datasets/TestSetNN.png]
* dataset_forNN_500samplesTEST_1percent_BFGS_hof4.npy
* dataset_forNN_500samplesTEST_1percent_NM_hof4.npy
* visualize_dataset.py: script used to produce the images of the datasets


The developed code is based on the DEAP library and makes use of Genetic Programming to evaluate online a control law for a 
Goddard Rocket test case in 3 different failure scenarios.

The main improvements introduced in comparison to a standard Genetic Programming algorithm are:
* The individuals in the population are divided in different categories according to their length.
* Such categories are considered in the InclusiveTournament selection mechanism, which performs a double tournament selection on each category to maintain diversity.
* The crossover operations are performed between individuals of different categories in order to maintain diversity in the population
* Two fitness functions are used. The first one it the primary objective function while the second one is a penalty function used to take into account the imposed constraints. All the selection mechanisms and the hall of fame insertion works accordingly to this penalty measure.
* The adopted evolutionary strategy consists in first find feasible individuals (penalty=0) and then evolve the population towards individuals with the lowest objective function (first fitness function).

This code was developed thanks to concepts and ideas discussed in [2], [3] and [4].
  

# Citation
If you use any part of the code, please cite [1].


## Main libraries used (Python):
  * DEAP 1.2.2: https://github.com/deap/deap
  * Scipy 1.3.2
  * Numpy 1.18.1
  * Matplotlib 3.1.1
  * Multiprocessing 3.8.3

## References
1. [TO BE PUBLISHED] F. Marchetti, E. Minisci. A Hybrid Neural Network-Genetic Programming Intelligent Control Approach. Bioinspired Optimization Methods and Their Applications. BIOMA 2020.


