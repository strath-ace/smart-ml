#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:22:10 2019

Neuroevolution of beam structure with CMA-ES.

@author: John Hutcheson
"""

# --- Import Libraries --- #

from __future__ import division

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import numpy as np

from deap import algorithms, base, creator, tools, cma

from keras.models import Sequential
from keras.layers.core import Dense

from matplotlib import pyplot as plt
from matplotlib import colors

import random

import time

# --- Define Functions --- #


def updateModel(model,weights1,weights2):
    
    model.layers[0].set_weights(weights1)
    model.layers[1].set_weights(weights2)
    
    return model


def generateModel(inp,hid,out):
    
    model = Sequential()
    model.add(Dense(hid, input_dim=inp,activation='sigmoid'))
    model.add(Dense(out,activation='relu'))
    
    varVecInit = np.ones([120,]) * random.uniform(-1,1) # could change this to randomise everything

    weights1Init = [np.reshape(varVecInit[:100],(inp,hid)),varVecInit[100:110]] # need to get this into the correct shape
    weights2Init = [np.array(varVecInit[-10:])[:,np.newaxis],np.array(0,)[np.newaxis]]
    
    model = updateModel(model,weights1Init,weights2Init)
    
    return model


def importWeights(weights):
    weightsData = np.loadtxt('bestNets.txt',delimiter=',')
    return weightsData


def generateModel2(weights,inp,hid,out): # import weights and bias from text file and create a new model. used in other scripts...
    
    data = weights
    
    weights1 = [np.reshape(data[:100],(inp,hid)),data[100:110]] # make a separate function for these two lines.
    weights2 = [np.array(data[-10:])[:,np.newaxis],np.array(0,)[np.newaxis]]
    
    model = generateModel(inp,hid,out)
    updateModel(model,weights1,weights2)
    
    return model


def importModel(weights): # import weights and bias from text file and create a new model
    
    data = np.loadtxt(weights)
    
    weights1 = [np.reshape(data[:100],(inp,hid)),data[100:110]] # make a separate function for these two lines.
    weights2 = [np.array(data[-10:])[:,np.newaxis],np.array(0,)[np.newaxis]]
    
    model = generateModel(inp,hid,out)
    updateModel(model,weights1,weights2)
    
    return model


def lk():
    
    E=1
    nu=0.3
    k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
    
    return (KE)


def analyseStructure(penal,x,edofMat,Emax,Emin,KE,u,free,iK,jK,ndof,f,nelx,nely): 
    
    sK=((KE.flatten()[np.newaxis]).T*(Emin+np.power(x,penal)*(Emax-Emin))).flatten(order='F') # stiuffness matrix x the element density. if the element is not dense, it provides no strucutral stiffness. 
    K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc() # stiffness matrix
    K = K[free,:][:,free]
    u[free,0]=spsolve(K,f[free,0]) # node displacement matrix.
    ce = (np.dot(u[edofMat].reshape(nelx*nely,8),KE) * u[edofMat].reshape(nelx*nely,8) ).sum(1) # problem using edofMat.
    elStrainEnergy = (Emin+x**penal*(Emax-Emin))*ce
    strucStrainEnergy=elStrainEnergy.sum()
    
    return u, strucStrainEnergy, elStrainEnergy


def oc(nelx,nely,x,dc,dv,volfrac):
    
    l1=0
    l2=1e9
    move=0.9 # adjust this as necessary... 
    xnew = np.zeros(nelx*nely)
    
    if np.sum(dc) == 0 :
        xnew = xnew
    else:
        while (l2-l1)/(l1+l2)>1e-3: 
            lmid=0.5*(l2+l1)
            xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid))))) 
            if np.sum(xnew[:]) > (volfrac * nelx * nely):
                l1 = lmid
            else:
                l2 = lmid

    return (xnew)


def generateStructure(dc,dcList,dv,xInit,Hs,H,nelx,nely,volfrac): # remove xPhys to get it to work normally.
    
    xPhys = xInit.copy() # update this at every step?? check topopt.
    
    dv[:] = np.ones(nely*nelx)
    dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0] 
    dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0] 
    dc[dc==0] = -1e-5 
    dcList.append(dc) 
    xPhysUpdate = oc(nelx,nely,xPhys,dc,dv,volfrac) # should xPhys here be the matrix of the structure at this particular step?
    xPhysUpdate[:]=np.asarray(H*xPhysUpdate[np.newaxis].T/Hs)[:,0]
    
    return xPhysUpdate


def hofNet(model,hof): # Generates trained network from hof entry
    
    weights = np.asarray(hof)
    weights1 = [np.reshape(weights[:100],(inp,hid)),weights[100:110]] # need to get this into the correct shape
    weights2 = [np.array(weights[-10:])[:,np.newaxis],np.array(0,)[np.newaxis]]
    updatedModel = updateModel(model,weights1,weights2) # need to define initial model.
    
    return updatedModel


def net2struc(net,LSF,dcList,dv,xInit,Hs,H,nelx,nely,volfrac):
    
    outputs = []
    inputs = LSF
    
    for i,_ in enumerate(inputs):
        inpVec = np.asarray(inputs[i]).T 
        outputs.append(net.predict(inpVec)) 
    
    sensitivity = np.squeeze(np.asarray(outputs))
    
    dc = -np.abs(sensitivity)

    struc = generateStructure(dc,dcList,dv,xInit,Hs,H,nelx,nely,volfrac) # check what dc list is for...
    
    return struc


def evaluateModel(individual,LSF,edofMat,dcList,run,model,Emax,Emin,KE,u,free,iK,jK,ndof,f,dv,xInit,Hs,H,nelx,nely):
    
    upNet = hofNet(model,individual)
    xPhys = net2struc(upNet,LSF,dcList,dv,xInit,Hs,H,nelx,nely,volfrac)
    _,strainEnergy,_ = analyseStructure(penal,xPhys,edofMat,Emax,Emin,KE,u,free,iK,jK,ndof,f,nelx,nely) # penal,xPhys,edofMat,Emax,Emin,KE,u,free,iK,jK,ndof,f
    fitness = strainEnergy
        
    return fitness,


def main(inp,hid,out,nelx,nely,penal,Emin,Emax,volfrac,rmin,iterations):
    
    H, Hs, dv, Emin, Emax, iK, jK, ndof, u, f, free, KE, xInit, edofMat = initFEA(nelx,nely,penal,Emin,Emax,volfrac,rmin)
    initialDisp,_,initialStrainEnergy = analyseStructure(penal,xInit,edofMat,Emax,Emin,KE,u,free,iK,jK,ndof,f,nelx,nely) # penal,x,edofMat,Emax,Emin,KE,u,free,iK,jK,ndof,f
    model = generateModel(inp,hid,out)
    
    dcList = []
    bestNets =[]
    hof = []
    
    plt.ioff() # Turn off plotting (graphs are saved as .PNG)
    fig,ax = plt.subplots()
    
    run = 1
    
    initialDispMax = max(initialDisp)
    initialDispMin = min(initialDisp)
    
    initialStrainEnergyMax = max(initialStrainEnergy)
    initialStrainEnergyMin = min(initialStrainEnergy)
    
    # Controls for the optimisation
    N = (inp * hid) + (out * hid) + hid # this has been updated ...
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # weights the importance of the fitness.
    
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    
    fitList = [] # list for the fitness values of each generation to be stored in.
    
    for i in range(iterations):
        if i ==0:    
            fittest = [0]*N
            
            initDispNorm = normalise(initialDisp,initialDispMax,initialDispMin)
            initStrainNorm = normalise(initialStrainEnergy,initialStrainEnergyMax,initialStrainEnergyMin)
            initXNorm = normalise(xInit,1,-1)
            
            LSFdata = lsfFunc(initDispNorm,initStrainNorm,initXNorm,nelx,nely)
            
        else:
            fittest = hof[0]
            
            disp,_,strainEnergy = analyseStructure(penal,bestStruc,edofMat,Emax,Emin,KE,u,free,iK,jK,ndof,f,nelx,nely)
             
            dispNorm = normalise(disp,initialDispMax,initialDispMin)
            strainNorm = normalise(strainEnergy,initialStrainEnergyMax,initialStrainEnergyMin)
            xNorm = normalise(bestStruc,1,-1)
            
            LSFdata = lsfFunc(dispNorm,strainNorm,xNorm,nelx,nely) # these must be the normalised values.
    
        np.random.seed(128)
        toolbox.register("evaluate", evaluateModel,LSF=LSFdata,edofMat=edofMat,dcList=dcList,run=run,model=model,Emax=Emax,Emin=Emin,KE=KE,u=u,free=free,iK=iK,jK=jK,ndof=ndof,f=f,dv=dv,xInit=xInit,Hs=Hs,H=H,nelx=nelx,nely=nely)
        
        strategy = cma.Strategy(fittest, sigma=0.01, lambda_=18, mu=9) # must multiply centroid by N.
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
       
        #pop,logbook = algorithms.eaGenerateUpdate(toolbox, ngen=3, stats=stats, halloffame=hof) # what is the output of this?
        
        
        NGEN = 2
        
        fbest = np.ndarray((NGEN,1))
        
        for gen in range(NGEN):
            # Generate a new population
            population = toolbox.generate()
            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Update the strategy with the evaluated individuals
            toolbox.update(population)
            
            hof.update(population)
            fbest[gen] = hof[0].fitness.values
            fitList.append(fbest)
            
        
        #update network
        bestNet = hofNet(model,hof[0])
        
        #generate structure
        bestStruc = net2struc(bestNet,LSFdata,dcList,dv,xInit,Hs,H,nelx,nely, volfrac)
        
        # generate best strucutre for plotting here
        
        xPhysPlot = np.reshape(bestStruc,[nelx,nely])
        ax.imshow(-xPhysPlot.T, cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
        fig.savefig('iteration_' + str(run) + '.png')
        
        bestNets.append(np.array(hof[0]))
        
        run = run + 1
        
    return hof, bestNets, fitList


def lsfFunc(disp,strain,x,nelx,nely): # Creates list of local state feature inputs for the MLP. Takes normalised values.
    
    nNodeY = nely+1
    nNodeY2 = nNodeY*2
    LSFData = []

    for j in range(nelx):
        for i in range(nely):
            LSFvec = []
            k = j*nNodeY2
            
            LSFvec.append(disp[(2*i)+k])
            LSFvec.append(disp[(2*i+1)+k])
            LSFvec.append(disp[((2*i)+nNodeY2)+k])
            LSFvec.append(disp[((2*i+1)+nNodeY2)+k])
            LSFvec.append(disp[((2*i+2)+nNodeY2)+k])
            LSFvec.append(disp[((2*i+3)+nNodeY2)+k])
            LSFvec.append(disp[(2*i+2)+k])
            LSFvec.append(disp[(2*i+3)+k])
            LSFvec.append((strain[i+j*nely])[np.newaxis])
            LSFvec.append((x[i+j*nely])[np.newaxis])
            
            LSFData.append(LSFvec)
    
    return LSFData


def normalise(inputs,maxVal,minVal): # Normalise inputs to zero mean in range [-1,1]
    
    inputs[inputs>maxVal]=maxVal
    inputs[inputs<minVal]=minVal
    
    inputsZA = inputs - np.average(inputs) # Zero averaged
    inputsNorm = 2*(inputsZA-minVal)/(maxVal-minVal)-1
    
    return inputsNorm


def initFEA(nelx,nely,penal,Emin,Emax,volfrac,rmin): # initialises the FEA analysis.
    
    # --- Initialise Variables --- #
    xInit=np.ones(nely*nelx,dtype=float) # the design variable starts as a matrix of the uniform volume fraction.
    ndof = 2*(nelx+1)*(nely+1) # number of degrees of freedom in the FE mesh. 2 DOF per node.
    edofMat=np.zeros((nelx*nely,8),dtype=int) # empty matrix to store degrees of freedom.
    KE=lk()
        
    for elx in range(nelx):
        for ely in range(nely):
            el = ely+elx*nely
            n1=(nely+1)*elx+ely
            n2=(nely+1)*(elx+1)+ely
            edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1]) # I am not sure what the numbers in this matrix represent.
    
    edofMat = edofMat.astype(int)
    
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat,np.ones((8,1))).flatten()
    jK = np.kron(edofMat,np.ones((1,8))).flatten()    

    # BC's and support
    dofs=np.arange(2*(nelx+1)*(nely+1))
    fixed=np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
    free=np.setdiff1d(dofs,fixed)
    
    # Solution and RHS vectors
    f=np.zeros((ndof,1))
    u=np.zeros((ndof,1))
    
    # Set load
    f[1,0]=-1
    
    # Filter: Build (and assemble) the index + data vectors for the coo matrix format
    nfilter=nelx*nely*((2*(np.ceil(rmin)-1)+1)**2)
    iH = np.zeros(int(nfilter)) # error here nFilter is a float 64 number and not an integer.
    jH = np.zeros(int(nfilter))
    sH = np.zeros(int(nfilter))
    cc=0
    
    for i in range(nelx):
        for j in range(nely):
            row=i*nely+j
            kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
            kk2=int(np.minimum(i+np.ceil(rmin),nelx))
            ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
            ll2=int(np.minimum(j+np.ceil(rmin),nely))
            for k in range(kk1,kk2):
                for l in range(ll1,ll2):
                    col=k*nely+l
                    fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                    iH[cc]=row
                    jH[cc]=col
                    sH[cc]=np.maximum(0.0,fac)
                    cc=cc+1
                             
    H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()    
    Hs=H.sum(1) # This stuff is required for sensitivity filtering. 
    dv = np.ones(nely*nelx)
    
    return H, Hs, dv, Emin, Emax, iK, jK, ndof, u, f, free, KE, xInit, edofMat


if __name__ == "__main__": # This script will run a 2D compliance based optimisation.
    
    # --- User Inputs --- #

    inp = 10
    hid = 10
    out = 1
    nelx = 70
    nely = 28
    penal = 3
    Emin=1e-9
    Emax=1.0
    volfrac = 0.4
    rmin = 2.5
    iterations = 3
    
    # --- Run the optimisation --- #
    
    # Plot controls    
    t1=time.time()
    hof,bestNets,fitList = main(inp,hid,out,nelx,nely,penal,Emin,Emax,volfrac,rmin,iterations)
    t2=time.time()
    print('Optimisation took: ' + str(t2-t1) + ' s.')
    
    # Save best networks for future use.
    np.savetxt('bestNets.txt',bestNets,delimiter=',')

    # Plot the best fitness of each generation over time.
    
    fig2,ax2 = plt.subplots()
    
    fitPlot = np.vstack(fitList)
    ax2.plot(fitPlot)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Minimum Strain Energy')
    ax2.set_title('Convergence History')
    
    fig2.savefig('convergence history.png')
