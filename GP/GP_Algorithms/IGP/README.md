# Inclusive Genetic Programming

The Inclusive Genetic Programming is a novel heuristic which was formulated in [1]. 
The evolutionary process at the core of the IGP is based on a modified version of the evolutionary strategy *μ + λ* [2]. The differences from 
the standard version consist in: 
1. the creation of the niches at the beginning of the evolutionary process and every time after a new offspring is generated; 
2. the use of the Inclusive Reproduction; 
3. the use of the Inclusive Tournament.

The niches are created in an evenly distributed manner (linearly divided) between the maximum and minimum length (number of nodes) of the individuals in the 
population, then the individuals are assigned to the respective niche according to their length. The same number of niches is kept during the evolutionary process, 
but their size (the interval of individuals lengths that they cover) and the amount of individuals inside them change at every generation. The variation of size of 
the niches allows for a shifting of the individuals between contiguous niches every time maximum and minimum lengths of the individuals in the population changes. 
Once the niches are created, both the reproduction and selection are performed considering individuals from different niches in order to maintain the population 
diversity. 

The Inclusive Reproduction consists in applying either crossover, mutation or 1:1 reproduction (the individual is passed unaltered to the offspring) using the 
individuals in the different niches. If the crossover is selected, a one point crossover is applied between two individuals which are selected from two different
niches. About the two individuals chosen, one is the best of the considered niche, in order to pass the best performing genes to the future generation and the 
other is selected randomly in order to maintain a certain degree of diversity in the population. Moreover, a mechanism to avoid breeding between the same or very 
similar individuals is used. If the mutation is selected, a mutation operator is applied to an individual randomly chosen from the chosen niche. 
Finally, if the 1:1 reproduction is selected, a randomly chosen individual from the chosen niche is passed to the offspring. The niches selected in all three 
previously described operations (crossover, mutation and 1:1 reproduction) are picked from a list of exploitable niches, which is continuously updated in order to 
avoid selecting always from the same niches. 

The Inclusive Tournament consists in performing a Double Tournament [3] on each niche. For the Inclusive Tournament the niches are selected in a sequential 
manner and the double tournament on each niche is performed at most *t* times where *t* is the number of individuals inside the considered niche, to avoid having 
clones in the final population.

## Folder structure

* Examples folder: contains the code to produce the results presented in [1]
* IGP_Functions.py: contains the functions and classes for the IGP
* Recombination_operators.py: contains the recombination functions, i.e. crossover and mutation, used in the IGP and SGP

## Citation

If you use and part of the cose, please cite [1].

## References
1. Marchetti F., Minisci E. . Inclusive Genetic Programming. In: Genetic Programming. EuroGP 2021. Lecture Notes in Computer Science. Springer, Cham (2021)
2. Beyer, H.G., Schwefel, H.P. . Evolution strategies – A comprehensive introduction.Natural Computing1(1), 3–52 (2002). https://doi.org/10.1023/A:1015059928466
3. Luke, S., Panait, L. . Fighting bloat with nonparametric parsimony pressure. In: Guervos, J.J.M., Adamidis, P., Beyer, H.G., Schwefel, H.P., Fernandez-Villacanas,J.L. (eds.)  Parallel  Problem  Solving  from  Nature  —  PPSN  VII.  pp.  411–421. Springer Berlin Heidelberg, Berlin, Heidelberg (2002)
