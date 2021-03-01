# Standard Genetic Programming

The script SGP_Functions.py contains a modified version of the eaMuPlusLambda and selDoubleTournament functions from the DEAP library. The introduced modifications 
are made to:

* eaMuPlusLambdaTolSimple:
** implemented stopping criteria based on tolerance on minimum value of fitness function
** use of custom POP class to keep track of the evolution of the population
** implemented a way to modified the introduced disturbances used in the FESTIP Ascent scenario.

* xselDoubleTournament:
** modified version of Double Tournament to consider individuals composed by multiple trees
