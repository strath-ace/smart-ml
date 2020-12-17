About the contents of the folder:
* GP_creationSet.npy: the disturbance scenarios used to create the GP control law offline.
![alt text](https://github.com/strath-ace-labs/smart-ml/blob/master/NN%2BGP_IntelligentControl/Datasets/GP_creationSet.png)
* training_points.npy: the disturbance scenarios used to optimize the GP control law 
![alt text](https://github.com/strath-ace-labs/smart-ml/blob/master/NN%2BGP_IntelligentControl/Datasets/training_points.png)
* TestSetNN.npy: the disturbance scenarios used to test the controller
![alt text](https://github.com/strath-ace-labs/smart-ml/blob/master/NN%2BGP_IntelligentControl/Datasets/TestSetNN.png)
* dataset_forNN_500samplesTEST_1percent_BFGS_hof4.npy: dataset created to train the NN using the BFGS algorithm
* dataset_forNN_500samplesTEST_1percent_NM_hof4.npy: dataset created to train the NN using the NM algorithm
* visualize_dataset.py: script used to produce the images of the datasets