Neuroevolution of 2D beam.

This script runs a topology optimisation of a 2D beam structure. A neural network is used to approximate update sensitivities for an OC update. 

The network is trained via neuroevolution. This script replicates work done in the following paper:

https://www.honda-ri.de/pubs/pdf/951.pdf

This script is also heavily influced by code avaialbe here:

http://www.topopt.mek.dtu.dk/Apps-and-software/Topology-optimization-codes-written-in-Python

CMA-ES is used to update the weights and baises of a fixed topology neural network. DEAP is used for the implementation of CMA-ES.

This takes a few hours to solve.

GIF of the optimisation:
![](GIF.gif)
