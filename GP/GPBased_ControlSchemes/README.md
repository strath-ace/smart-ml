# Genetic Programming based Control Schemes

## Description

Algorithms for the intelligent and non intelligent control of launch vehicles using primarily Genetic Programming along with other Machine Learning techniques.

Please refer to the subfolder for detailed explanation of the algorithms and their usage:
* **Intelligent_nonIntelligent_GPControl** (Intelligent and non Intelligent Genetic Programming Control) contains the 
algorithms developed for the work presented in [1], [3] and [5]. [1] is an implementation of the online use of GP 
to perform intelligent control applied on the Goddard ascent vehicle. [3] employs GP to find offline the guidance 
control law of the FESTIP considering uncertainties. In [5] IGP is applied to generate offline the control law for an 
harmonic oscillator and an inverted pendulum. 
* **IntHGPNNC** (Intelligent Hybrid Genetic Programming Neural Network Control) contains the algorithm developed for 
the work presented in [2]. It is an implementation of the online use of a NN to optimize online a control law produced using GP to perform intelligent control. It is applied on the ascent trajectory of the FESTIP vehicle.
* **GANNIC** (Genetically Adapted Neural Network-Based Intelligent Controller) contains the code developed for the 
Francesco Marchetti's PhD Thesis and presented in [4].
* **OPGD_IGP** (Operators Gradient Descent Inclusive Genetic Programming) contains the code for the approach presented
in [5]. The OPGD was implemented as in [6].

## References
1. Marchetti F., Minisci E., Riccardi A. . Towards Intelligent Control via Genetic Programming. 2020 International Joint Conference on Neural Networks (IJCNN) (2020)
2. Marchetti F., Minisci E. . A Hybrid Neural Network-Genetic Programming Intelligent Control Approach. In: B. Filipič, E. Minisci, M. Vasile (eds.) Bioinspired Optimization Methods and Their Applications. BIOMA 2020. Springer, Cham, Brussels (2020) 
3. Marchetti F., Minisci E. (2021). Genetic programming guidance control system for a reentry vehicle under uncertainties. Mathematics, 9(16), 1–17. https://doi.org/10.3390/math9161868
4. [SUBMITTED TO] Marchetti F., Minisci E., Genetically Adapted Neural Network-Based Intelligent Controller for Reentry Vehicle 
Guidance Control. Engineering Applications of Artificial Intelligence. 2023.
5. [SUBMITTED TO] Marchetti F., Pietropolli G., Camerota Verdù F.J., Castelli M. , Minisci E. . Control Law Automatic Design 
Through Parametrized Genetic Programming with Adjoint State Method Gradient Evaluation. Applied Soft Computing. 2023
6. F.-J. Camerota-Verdù, G. Pietropolli, L. Manzoni, M. Castelli, Parametrizing gp trees for better symbolic
regression performance through gradient descent, in: Proceedings of the Genetic and Evolutionary Computation Conference
Companion, 2023.