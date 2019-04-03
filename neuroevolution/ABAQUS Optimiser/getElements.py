# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:24:07 2019

This script should be run from within abaqus to extract the required information.

Could adapt this for multiprocessing. 

Get element / node information from abaqus odb file. 

could do a numpy append??

@author: pqb18127
"""

# --- Import libraries --- #

from odbAccess import openOdb
import numpy as np
import time

# --- Define functions --- #

def getLSF(Mdb,Iter): # this runs an FEA and creates the inputs for the ANNs.
        
    # Run analysis on model 

        Mdb.Job('Design_Job'+str(Iter),'Model-1').submit() # submit a new job
        Mdb.jobs['Design_Job'+str(Iter)].waitForCompletion() # wait for the job to complete

        
        # Open odb
        opdb = openOdb('Design_Job'+str(Iter)+'.odb')
        seng = opdb.steps['Step-1'].frames[-1].fieldOutputs['ESEDEN'.upper()].values # ESEDEN is element strain energies
        disp = opdb.steps['Step-1'].frames[-1].fieldOutputs['U'].values
        
        # Get element and node numbers for indexing.
        
        elementSet = opdb.rootAssembly.instances['Part-1-1'.upper()].elementSets['Set-1'.upper()].elements # need to add the upper() function for this to work.
        solidElSet = opdb.rootAssembly.instances['Part-1-1'.upper()].elementSets['Set-1'.upper()].elements # list of element numbers for solid elements
        
        solidEls = []
        elDens = [] # empty list for element density to be stored.
        elementNodes=[]
        
        # need to make a list of all of the elements in the 
        for el in solidElSet:
            solidEls.append(el.label)
        
        
        for el in elementSet:
            elementNodes.append(el.connectivity) 
            if(el.label in solidEls): # if solid el set contains el.label then assign density of 1.
                elDens.append(1)
            else:
                elDens.append(0.001)
            
            
        strainEner = []
        for el in seng:
            strainEner.append(el.data)
        
        u1 =[]
        u2 =[]    
        
        for node in disp:
            u1.append(node.data[0])
            u2.append(node.data[1])
        
        # Save the results to a file for inspection.
        
        LSF = []
        for i in range(len(elementNodes)):
            elLSF = []
            length = len(elementNodes[i])
            for j in range(length):
                elLSF.append(u1[elementNodes[i][j]-1]) # check the use of -1 here ... 
                elLSF.append(u2[elementNodes[i][j]-1])
            if length==3: # duplicate third node displacements for 3 node element.
                elLSF.append(u1[elementNodes[i][2]-1])
                elLSF.append(u2[elementNodes[i][2]-1])
                
            elLSF.append(strainEner[i]) # add the strain energy of the element
            elLSF.append(elDens[i])
            LSF.append(elLSF) # append element LSF to global LSF list.
            
        
        #histKey = re.sub(r'\W+', '',str(opdb.steps['Step-1'].historyRegions.keys())) # try and get this working last.
        #histKey = str(opdb.steps['Step-1'].historyRegions.keys())
        obj = [opdb.steps['Step-1'].historyRegions['assembly Assembly'].historyOutputs['ALLWK'].data[-1][1]] # save as list of one element. 
        np.savetxt('LSF.csv',LSF,delimiter=',')
        np.savetxt('obj.csv',obj,delimiter=',')
        
if __name__ == '__main__':
    
    mddb = openMdb(getInput('Input CAE file:',default='cantilever.cae'))
    mdl = mddb.models['Model-1']
    mdl.FieldOutputRequest('SEDensity','Step-1',variables=('ELEDEN', )) # request output fields.
    mdl.HistoryOutputRequest('ExtWork','Step-1',variables=('ALLWK', ))
    
    iter = 1
    
    getLSF(mddb,iter)
