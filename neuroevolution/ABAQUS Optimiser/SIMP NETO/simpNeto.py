# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:09:58 2019

This runs the full inner loop of NETO. Some fine tuning needs to be done to ensure that everything is running as it should be.

This has been modified to stop the previous network from being used as the starting point for the next iteration. 

@author: John Hutcheson
""" 

import multiprocessing 
from deap import tools, creator, base, cma, algorithms
import numpy as np
import time
import os 
import shutil

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # this should prevent tensor flow from accessing gpu...


# --- Functions relating to the ANN --- #

def updateModel(model,weights,bias):
    model.layers[0].set_weights(weights)
    model.layers[1].set_weights(bias)
    return model


def normalise(mat,initial):
    if type(initial) == int:
        initial = mat.copy()
    else:
        initial = initial.copy()
    mat = mat.copy()
    if len(set(np.ravel(mat)))==1:
        return np.zeros_like(mat)
    else:
        return ((mat-np.min(initial)) / (np.max(initial) - np.min(initial)))*2 - 1


def generateModel(inp,hid,out):
    from keras.models import Sequential
    from keras.layers.core import Dense
    model = Sequential()
    model.add(Dense(hid, input_dim=inp,activation='sigmoid'))
    model.add(Dense(out,activation='relu'))
    return model


def updateAnn(model,params,inp,hid):
    paramsArray = np.asarray(params)
    weights = [np.reshape(paramsArray[:100],(inp,hid)),paramsArray[100:110]] # need to get this into the correct shape
    #weights = [np.reshape(paramsArray[:20],(inp,hid)),paramsArray[20:30]] # need to get this into the correct shape
    bias = [np.array(paramsArray[-10:])[:,np.newaxis],np.array(0,)[np.newaxis]]
    ann = updateModel(model,weights,bias) # need to define initial model.
    return ann


def calcSens(ann,LSF):
    outputs = []
    inputs = LSF
    for i,_ in enumerate(inputs):
        #inpVec = inputs[i][-2:] # This will only select the last two elements of each array (strain energy and density).
        inpVec = inputs[i]
        sens = ann.predict(np.array([inpVec,]))
        outputs.append(sens)
    return np.squeeze(np.asarray(outputs)) # returns the sensitivities.


def getSens(x,inp,hid,out,lsfList):
    import tensorflow as tf
    from keras import backend as K    
    session = tf.Session()
    K.set_session(session)
    ann1 = generateModel(inp,hid,out)
    ann2 = updateAnn(ann1,x,inp,hid)
    sens = calcSens(ann2,lsfList)
    K.clear_session()
    return sens


def closeDir(homeDir,jobDir):
    os.chdir(homeDir)
    time.sleep(0.1) # might be able to remove this ... 
    shutil.rmtree(jobDir)


def setDir(copyDir,pasteDir):
         jobDir = shutil.copytree(copyDir,pasteDir)
         os.chdir(jobDir)
         return jobDir
        
        
def runAbaqus(script_1,script_2):
    os.system('abaqus cae noGui=' + script_1) 
    os.system('abaqus job=Job-1 interactive ask_delete=OFF') 
    os.system('abaqus viewer noGui='+ script_2) 

def getObj():
    return np.loadtxt('obj.csv',delimiter=',')
    

def getLSF(initial_LSF): # here
    
    initial_disps = initial_LSF[0]
    initial_seng = initial_LSF[1]
    initial_elDen = initial_LSF[2]
    
    LSF = np.loadtxt('LSF.csv',delimiter=',')
    LSF[:,0:8] = normalise(LSF[:,0:8],initial_disps)
    LSF[:,8] = normalise(LSF[:,8],initial_seng)
    LSF[:,9] = normalise(LSF[:,9],initial_elDen)
    elDen = np.loadtxt('elDen.csv')# sets initial element density to full.# read this from elDen.csv # TODO: THis is new
    np.savetxt('normLSF.csv',LSF,delimiter=',')
    return list(LSF), list(elDen) # output these parameters as lists instead of arrays and see if this helps with the ann.predict function.     


def getLSF2(initial_LSF):
    initial_disps = initial_LSF[0]
    initial_seng = initial_LSF[1]
    initial_elDen = initial_LSF[2]
    
    LSF = np.loadtxt('LSF.csv',delimiter=',')
    LSF[:,0:8] = normalise(LSF[:,0:8],initial_disps)
    LSF[:,8] = normalise(LSF[:,8],initial_seng)
    LSF[:,9] = normalise(LSF[:,9],initial_elDen)
    elDen = np.loadtxt('elDenNew.csv')# sets initial element density to full.# read this from elDen.csv # TODO: THis is new
    np.savetxt('normLSF.csv',LSF,delimiter=',')
    return list(LSF), list(elDen) # output these parameters as lists instead of arrays and see if this helps with the ann.predict function.     

# Old BESO function.
# this BESO function will not work ...
#def BESO(sens,elDen,vf): # TODO: Something funky is happending here, tv is not correct...
#   elDenNew = elDen.copy()
#   sens[sens==0]=1e-4
#   lo,hi = min(sens),max(sens)
#   tv = vf*len(elDen)
#   while (hi-lo)/hi > 1.0e-9: # This was at 1.0e-5.
#       th=(lo+hi)/2
#       for i,_ in enumerate(elDen):
#           if sens[i]>th:
#               elDenNew[i]=sens[i]  
#           else:
#               elDenNew[i]=0.001 # sets density based on th value
#       if (sum(elDenNew) - tv) >0: lo = th
#       else: hi = th
#   return elDenNew


def BESO(sens,elDen,vf): # this is actually the SIMP function.
    l1=0
    l2=1e9
    elDen = -np.abs(elDen)
    move=0.9 # adjust this as necessary... 
    elDenNew = np.zeros(len(elDen))
    dv = 1
    elDen=np.array(elDen)
    if np.sum(sens) == 0 :
        elDenNew = elDenNew
    else:
        while (l2-l1)/(l1+l2)>1e-3: 
            lmid=0.5*(l2+l1)
            print(lmid)
            elDenNew[:]= np.maximum(0.0,np.maximum(elDen-move,np.minimum(1.0,np.minimum(elDen+move,elDen*np.sqrt(-sens/dv/lmid))))) 
            if np.sum(elDenNew[:]) > (vf * len(elDen)):
                l1 = lmid
            else:
                l2 = lmid
    elDenNew=list(elDenNew)
    return elDenNew


def evaluate(x, lsfList, vol, elDen, inp, hid, out, homeDir, case, initLSF, iteration): 
    i = iteration
    processNo = os.getpid()
    
    # Pre-filt should happen in here. (In case 1 and case 2)
    
    # Filter only needs to occur here.
    if case == 'Case_1': # Initial run (global)
        script_1 = 'generateInp5.py'
        script_2 = 'getResults.py'
        copyDir = homeDir + '\original'
        pasteDir = homeDir + '\evalLSF_' + str(i)
        setDir(copyDir,pasteDir)
        
        runAbaqus(script_1,script_2)
        LSF,elDen = getLSF(initLSF)
        return LSF,elDen
    
    
    elif case == 'Case_2': # Initial run (local)
        script_1 = 'generateInp4.py'
        script_2 = 'getResults.py'
        copyDir = homeDir + '\\final'
        pasteDir = homeDir + '\evalLSF_' + str(i)
        setDir(copyDir,pasteDir)
        shutil.rmtree(copyDir) # Remove the final folder once its contents has been copied to an eval folder.
        runAbaqus(script_1,script_2)
        LSF,elDen = getLSF2(initLSF)
        print('Starting volume: ' + str(sum(elDen)))
        return LSF,elDen
        
    
    elif case == 'Case_3': # Final run (local) 
        copyDir = homeDir + '\evalLSF_'+ str(i)
        pasteDir = homeDir + '\\final'
        script_1 ='generateInp4.py'
        script_2 = 'getResults.py'

        setDir(copyDir,pasteDir)
        
        sens = getSens(x,inp,hid,out,lsfList)
        
        np.savetxt('sens.csv',sens,delimiter=',')
        
        
        os.system('abaqus cae noGui=filterScript.py') # run the filter.
        filtSens = np.loadtxt('filtSens.csv')
        
        elDenNew = BESO(filtSens,elDen,vol)
        np.savetxt('elDenNew.csv',elDenNew,delimiter=',')
        
        runAbaqus(script_1,script_2) # analyse the new strucutre 
        LSF,elDen = getLSF2(initLSF) # get LSF and elDen from new structure.
        os.chdir(homeDir)
        print('Finishing volume: ' + str(sum(elDen)))
        return LSF,elDen
    
    
    elif case == 'Case_4': # CMA-ES standard run.
        sens = getSens(x,inp,hid,out,lsfList) 
        if len(set(sens))== 1:
            obj = 1e5 
        else:
            elDenNew = BESO(sens,elDen,vol)
            script_1 ='generateInp4.py'
            script_2 = 'getResults.py'
            copyDir = homeDir + '\original'
            pasteDir = homeDir + '\process_' + str(processNo)
            jobDir = setDir(copyDir,pasteDir)
            
            np.savetxt('sens.csv',sens,delimiter=',')
            
            os.system('abaqus cae noGui=filterScript.py')
            filtSens = np.loadtxt('filtSens.csv')
            elDenNew = BESO(filtSens,elDen,vol)
            
            np.savetxt('elDenNew.csv',elDenNew,delimiter=',') 
            
            runAbaqus(script_1,script_2)
            obj = getObj()
            closeDir(homeDir,jobDir)
        return obj,


def main(elDen,LSF,vol,inp, hid, out, homeDir, case, initLSF, iteration):
    np.random.seed(20)
    toolbox.register("evaluate", evaluate, lsfList=LSF, vol=vol, elDen=elDen, inp=inp, hid=hid, out=out, homeDir=homeDir, case=case, initLSF=initLSF, iteration=iteration) # having an evaluation function that is part of a class is very dangerous since the object does not have the same state in all processes.

    N = (inp * hid) + (hid * out) + hid

    if iteration ==0:
        strategy = cma.Strategy([0]*N, sigma=0.01, lambda_=18, mu=9)
    else:
        strategy = cma.Strategy(fittest[-1], sigma=0.01, lambda_=18, mu=9)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # --- Run CMA-ES --- #
    algorithms.eaGenerateUpdate(toolbox, ngen=50, stats=stats, halloffame=hof) 
    return hof

    # --- --- #

# --- Other global definitions --- #    

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()



if __name__=='__main__':
    
    t1=time.time()
    # --- Specify Job Directories --- #
    homeDir = os.getcwd()
    iterations=4
    fittest = []
    vol = 0.8
    inp = 10
    hid = 10
    out = 1
    
    for i in range(iterations):
        
        print('Iteration ' + str(i))
        
        if i ==0:
            initial_LSF = (1,1,1)
            LSF, elDen = evaluate([], [], vol, [], inp, hid, out, homeDir, 'Case_1', initial_LSF, i)
            # run evaluate for case 1
            
            LSFarray = np.array(LSF)
            initial_disps = LSFarray[:,0:8]
            initial_seng = LSFarray[:,8]
            initial_elDen = np.array((1.0,-1.0)) # gives a variation in the element density.
            initial_LSF = (initial_disps,initial_seng,initial_elDen)
            
        else:
            LSF, elDen = evaluate([], [], vol, [], inp, hid, out, homeDir, 'Case_2', initial_LSF, i)
        
        print('Starting Volume: ' + str(round(sum(elDen))))
        #pool = multiprocessing.Pool(processes=6)
        #toolbox.register("map",pool.map)
        
        hof = main(elDen,LSF,vol,inp, hid, out, homeDir, 'Case_4', initial_LSF, i)
    
        fittest.append(hof[0])
                    
        # Final evaulation.
        evaluate(fittest[-1], LSF, vol, elDen, inp, hid, out, homeDir, 'Case_3', initial_LSF, i)
        
    print('Optimisation Complete!')
       
    t2= time.time()
    
    print('Optimisation took ' + str(t2-t1) + ' s.')
