# Genetic Programming for Intelligent Control of a Goddard Rocket

The code contained in this directory is the one used to obtain the results presented in [1] at the WCCI/IJCNN 2020 conference. 
To reproduce the results run the files "MAIN_CdScenario.py", "MAIN_GustScenario.py" and "MAIN_DensityScenario.py".

The developed code is based on the DEAP library and makes use of Genetic Programming to evaluate online a control 
law for a Goddard Rocket test case in 3 different failure scenarios.

The folder Goddard_Models contains the used reference trajectory, and the necessary models to run the scripts.

If you use any part of the code, please cite [1].

## References
1. F. Marchetti, E. Minisci, A. Riccardi. Towards Intelligent Control via Genetic Programming. The 2020 IEEE International Joint Conference on Neural Network Proceedings. 2020