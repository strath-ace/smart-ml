# Intelligent and Non-Intelligent Control Schemes based on Genetic Programming

This folder contains the control schemes developed using GP and applied on different test cases. The folder is structured
as follows:
* ASOC_paper: contains part of the code used to produce the results presented in [3]. In particular, the IGP applied 
to design the control law of the harmonic oscillator and the inverted pendulum on a cart. This application is not intelligent
control.
* FESTIP: contains the code used to produce the results presented in [2]. It uses the IGP to design the control
law of the FESTIP-FSS5 reusable launch vehicle. This application is not intelligent
* Goddard: contains the code used to produce the results presented in [1]. It uses the GP to design the control
law of a goddard rocket. It can be considered intelligent control.


## References
1. Marchetti F., Minisci E., Riccardi A. . Towards Intelligent Control via Genetic Programming. 2020 International Joint Conference on Neural Networks (IJCNN) (2020)
2. Marchetti, F., Minisci, E. & Riccardi, A. Single-stage to orbit ascent trajectory optimisation with reliable evolutionary initial guess. Optim Eng 24, 291–316 (2023). https://doi.org/10.1007/s11081-021-09698-w
3. [SUBMITTED TO] Marchetti F., Pietropolli G., Camerota Verdù F.J., Castelli M. , Minisci E. . Control Law Automatic Design 
Through Parametrized Genetic Programming with Adjoint State Method Gradient Evaluation. Applied Soft Computing. 2023
