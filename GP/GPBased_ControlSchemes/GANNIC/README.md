# Genetically Adapted Neural Network-Based Intelligent Controller (GANNIC)

The code in this folder is use for the design and application of the GANNIC control scheme. The GANNIC scheme is 
composed by a Neural Network (NN) which is used as nonlinear controller. The weights of the NN are updated online
using a set of differential equations found offline using Genetic Programming (GP). For more information please
refer to [1]. If you use the code in this folder please cite [1].

To reproduce the results in [1] run main_GANNIC.py inside the FESTIP_reentry folder. After the simulations are 
performed, the algorithm post_processing/find_best.py can be run to find at which generation, for each 
simulation, the best individual is found. The output of this script is used to run the 
post_processing/GANNIC_results_analysis.py script. This last script is used to produce the plots of the 
best individuals found in each simulation.

Folder structure:
* FESTIP_reentry fodler: contains the FESTIP vehicle models and the main script to run the results presented in [1]
* post_processing folder: contains the code used to post-process the produced results
* functions_GANNIC.py: contains the functions used to run the GANNIC scheme
* utils.py: contains some other functions
* utils_NN.py: contains the functions used to create and update the NN

## References
1. [SUBMITTED TO] Marchetti F., Minisci E., Genetically Adapted Neural Network-Based Intelligent Controller for Reentry Vehicle 
Guidance Control. Engineering Applications of Artificial Intelligence. 2023.