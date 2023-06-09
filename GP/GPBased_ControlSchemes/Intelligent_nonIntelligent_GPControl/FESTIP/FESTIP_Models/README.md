# FESTIP Models

This folder contains the models and necessary data of the FESTIP-FSS5 ascent vehicle. The models were taken from [1] and are contained inside the script models_FESTIP.py.
The uncertainties implemented in the aerodynamic and atmospheric models were taken from [2].

The files crowd_cl.mat, crowd_cd, crowd_cm.mat, crowd_alpha.mat, crowd_mach.mat contain the aerodynamic coefficients used in the aerodynamic model. These are not the original files used in [1] but a modified version produced in [3]. 
The file impulse.dat contains the specific impulse values used by the thrust model. 
The file reference_trajectory.mat contains the reference trajectory produced in [3].


## References
1. D’Angelo, S., Minisci, E., Di Bona, D., Guerra, L.: Optimization Methodology forAscent Trajectories of Lifting-Body Reusable Launchers. Journal of Spacecraft and Rockets 37(6) (2000)
2. Pescetelli, F., Minisci, E., Maddock, C., Taylor, I., Brown, R.E.: Ascent trajectory optimisation  for  a  single-stage-to-orbit  vehicle  with  hybrid  propulsion.  In:  18th AIAA/3AF International Space Planes and Hypersonic Systems and Technologies Conference 2012. pp. 1–18 (2012)
3. Marchetti, F., Minisci, E. & Riccardi, A. Single-stage to orbit ascent trajectory optimisation with reliable evolutionary initial guess. Optim Eng 24, 291–316 (2023). https://doi.org/10.1007/s11081-021-09698-w
