# Hybrid Neural Network-Genetic Programming Intelligent Control Approach

The code contained in this directory is the one used to obtain the results presented in [1] at the BIOMA 2020 conference. The developed code is based on the DEAP library for the Genetic Programming (GP) part and Tensorflow for the Neural Network (NN). As described in [1] the overall procedure to use the code is summarized in the picture below

![alt text](https://github.com/strath-ace-labs/smart-ml/blob/master/GP/IntHGPNNC/FESTIP/Ascent/procedure.png)

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

A more detailed description on the folders contents can be found inside them.

The default settings of the code are the same used in [1]. It is possible to run the 3 section separately to reproduce the results of the paper.
To produce completely new results, the sequence described above must be followed.
To test the control approach on another vehicle or plant, the paths in the scripts must be modified to the folder where your models are contained and you have to copy and modify accordingly the propagated functions.

If you use any part of the code, please cite [1].


## References
1. F. Marchetti, E. Minisci. A Hybrid Neural Network-Genetic Programming Intelligent Control Approach. Bioinspired Optimization Methods and Their Applications. BIOMA 2020.


