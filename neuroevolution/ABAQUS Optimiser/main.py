# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:09:58 2019

This runs the full inner loop of NETO. Some fine tuning needs to be done to ensure that everything is running as it should be.

@author: John Hutcheson
"""

import multiprocessing 
from deap import tools, creator, base, cma, algorithms
import numpy as np
import time
import os 
import shutil


# --- Functions relating to the ANN --- #

def updateModel(model,weights,bias):
    model.layers[0].set_weights(weights)
    model.layers[1].set_weights(bias)
    return model

def normalise(inputs,maxVal,minVal): # Normalise inputs to zero mean in range [-1,1]
    inputs[inputs>maxVal]=maxVal
    inputs[inputs<minVal]=minVal
    inputsZA = inputs - np.average(inputs) # Zero averaged
    inputsNorm = 2*(inputsZA-minVal)/(maxVal-minVal)-1
    return inputsNorm

def generateModel(inp,hid,out):
    from keras.models import Sequential
    from keras.layers.core import Dense
    model = Sequential()
    model.add(Dense(hid, input_dim=inp,activation='sigmoid'))
    model.add(Dense(out,activation='relu'))
#    params = (inp*hid) + hid +(inp*out)
#    varVecInit = np.random.uniform(low=-10, high=10, size=(params,))
#    weights = [np.reshape(varVecInit[:100],(inp,hid)),varVecInit[100:110]] # weights and biases
#    bias = [np.array(varVecInit[-10:])[:,np.newaxis],np.array(0,)[np.newaxis]] # weights and biases
#    model = updateModel(model,weights,bias)
    return model


def updateAnn(model,params,inp,hid):
    paramsArray = np.asarray(params)
    weights = [np.reshape(paramsArray[:100],(inp,hid)),paramsArray[100:110]] # need to get this into the correct shape
    bias = [np.array(paramsArray[-10:])[:,np.newaxis],np.array(0,)[np.newaxis]]
    ann = updateModel(model,weights,bias) # need to define initial model.
    return ann


def calcSens(ann,LSF):
    outputs = []
    inputs = LSF
    for i,_ in enumerate(inputs):
        inpVec = inputs[i]
        sens = ann.predict(np.array([inpVec,]))
        if sens >0:    
            outputs.append(0)
        else:
            outputs.append(sens)
    return np.squeeze(np.asarray(outputs)) # returns the sensitivities.


def getObj():
    return np.loadtxt('obj.csv',delimiter=',')
    

def getLSF():
    LSF = np.loadtxt('LSF.csv',delimiter=',')
    # TODO: add normalise function here, so that LSF values are pre-normalised. Use map
    # apply function to each row of numpy array. LSF =  ...; 
    elDen = np.ones(len(LSF))
    return list(LSF), list(elDen) # output these parameters as lists instead of arrays and see if this helps with the ann.predict function.     


def runAbaqus(homeDir,copyDir,pasteDir,i,LSF):
        jobDir = shutil.copytree(copyDir,pasteDir + str(i)) # copy the original folder for use in this analysis.
        os.chdir(jobDir) # change working directory to new folder
        os.system('abaqus cae noGui=generateInp.py') # generate the input file in new folder
        os.system('abaqus job=Job-1 interactive ask_delete=OFF') # run the analysis
        os.system('abaqus viewer noGui=getElements.py -- ' + str(i) ) # runs macro and passes i value to it # extract the data from the obd.
        if LSF == True:
            LSF,elDen = getLSF()
            os.chdir(homeDir) # change working directory to new folder
            return LSF, elDen
        elif LSF == False:
            obj = getObj()
            os.chdir(homeDir) 
            shutil.rmtree(jobDir) # must remove the job folder once fitness has been calculated. Not deleting the folder... 
            return obj
        

def BESO(sens,elDen,ert,vf): # it might be better to use dictionaries for this but lets see if it works with lists 
   vh = sum(elDen)/len(elDen)
   nv = max(vf,vh*(1-ert))
   lo,hi = min(sens),max(sens)
   tv = nv*len(elDen)
   print(vh,nv,lo,hi,tv)
   while (hi-lo)/hi > 1.0e-5:
       th=lo+hi/2
       for i,_ in enumerate(elDen):
           if sens[i]>th:
               elDen[i]=1  
           else:
               elDen[i]=0.001 # sets density based on th value
       if sum((elDen) - tv) >0: lo = th
       else: hi = th
   return elDen    


def evaluate(x, lsfList, vol, elDen, inp, hid, out, ert, homeDir, copyDir, pasteDir): # custom define rastrigin function. This runs
    import tensorflow as tf
    from keras import backend as K    
    session = tf.Session()
    K.set_session(session)
    i = multiprocessing.current_process()._identity # creates unique identity for naming the job folder.
    ann1 = generateModel(inp,hid,out)
    ann2 = updateAnn(ann1,x,inp,hid) # [complete].
    # calculate the sensitivities for each element with the network
    sens = calcSens(ann2,lsfList) # [complete]
    K.clear_session()
    if len(set(sens))==1:
        obj = 1e3 # might need to change the value of this # TODO: check value of obj when sens is all the same.
    else:
        # Use sensitivities to generate a structure (BESO)
        BESO(sens,elDen,ert,vol) # [complete]
        # update the input file with new structure and run analysis.
        # Analyse the structure 
        obj = runAbaqus(homeDir,copyDir,pasteDir,i,False)
    return obj, # comma must go here 

def main(elDen,LSF,vol,inp, hid, out, ert, homeDir, copyDir, pasteDir):
    np.random.seed(128)
    toolbox.register("evaluate", evaluate, lsfList=LSF, vol=vol, elDen=elDen, inp=inp, hid=hid, out=out, ert=ert, homeDir=homeDir, copyDir=copyDir, pasteDir=pasteDir) # having an evaluation function that is part of a class is very dangerous since the object does not have the same state in all processes.
    #initial= np.random.uniform(low=-0.01, high=0.01, size=(120,))
    initial = np.zeros(120) # check this ...
    strategy = cma.Strategy(initial, sigma=0.01/3, lambda_=18, mu=9) # must multiply centroid by N.
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    # --- Run CMA-ES --- #
    algorithms.eaGenerateUpdate(toolbox, ngen=2, stats=stats, halloffame=hof) # program runs to this line...
    # --- --- #


# --- Other global definitions --- #    

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()


if __name__=='__main__':
    
    t1=time.time()
    # --- Specify Job Directories --- #
    homeDir = r'C:\Users\pqb18127\OneDrive\PhD\Python\DEAP\multiprocessing\MP Abaqus and DEAP Test'
    copyDir = r'C:\Users\pqb18127\OneDrive\PhD\Python\DEAP\multiprocessing\MP Abaqus and DEAP Test\original'
    pasteDir = r'C:\Users\pqb18127\OneDrive\PhD\Python\DEAP\multiprocessing\MP Abaqus and DEAP Test\process_'
    # --- Get LSF Data --- #
    pasteDir2 = homeDir + '\evalLSF_'
    LSF, elDen = runAbaqus(homeDir,copyDir,pasteDir2,1,True) # creates initial LSF data
    # --- User Inputs --- #
    vol = 0.5
    inp = 10
    hid = 10
    out = 1
    ert = 0.05
    pool = multiprocessing.Pool(processes=6)
    toolbox.register("map",pool.map)
    main(elDen,LSF,vol,inp, hid, out, ert, homeDir, copyDir, pasteDir)
    t2= time.time()
    print(str(t2-t1))
