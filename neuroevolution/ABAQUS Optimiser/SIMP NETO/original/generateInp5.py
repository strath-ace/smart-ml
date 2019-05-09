# -*- coding: utf-8 -*-

# slb and vlb are lists of the element numbers that should be assigned to each section.

# This script takes an abaqus cae model and writes an input file based on the list of element densities described in 'elDen.csv'

import numpy as np
import step


## Function of preparing filter map (Fm={elm1:[[el1,el2,...],[wf1,wf2,...]],...})
def preFlt(Rmin,Elmts,Nds,Fm):
    
    # Calculate element centre coordinates
    elm, c0 = np.zeros(len(Elmts)), np.zeros((len(Elmts),3))
    for i in range(len(elm)):
        elm[i] = Elmts[i].label
        nds = Elmts[i].connectivity
        for nd in nds: c0[i] = np.add(c0[i],np.divide(Nds[nd].coordinates,len(nds)))
    # Weighting factors
    for i in range(len(elm)):
        Fm[elm[i]] = [[],[]]
        for j in range(len(elm)):
            dis = np.square(np.sum(np.power(np.subtract(c0[i],c0[j]),2)))
            if dis<Rmin:
                Fm[elm[i]][0].append(elm[j])
                Fm[elm[i]][1].append(Rmin - dis)
            Fm[elm[i]][1] = np.divide(Fm[elm[i]][1],np.sum(Fm[elm[i]][1])) # does this need to be saved as a list to work??
            Fm[elm[i]][1] = Fm[elm[i]][1].tolist() # added to convert datatype to a list
    print(Fm)


def generateInp():
    Mdb = openMdb('cantilever.cae')
    # Replace format database stuff.
    mdl = Mdb.models['Model-1']
    part = mdl.parts['Part-1']
    Elmts = part.elements
    
    Xe=np.ones(len(Elmts)) # should load this in
    np.savetxt('elDen.csv',Xe,delimiter=',')
    
    
    # Assign new materials and section based on Xe.
    elDic = dict()    
    for i,j in enumerate(Xe):
        if j in elDic: # append the new number to the existing array at this slot
            elDic[j].append((i+1))
        else: # create a new array in this slot
            elDic[j] = [i+1]
    
    for i in elDic.keys():
        
        ident = str(i).replace(".", "")
        
        matName = 'Material' + ident
        secName = 'Sec' + ident
        setName = 'Set' + ident
        
        mdl.Material(matName).Elastic(((i, 0.3), )) # assigns young's modulus based on key.        
        mdl.HomogeneousSolidSection(secName,matName) # Name and material
        
        part.SectionAssignment(part.Set(setName,part.elements),secName)
        part.SectionAssignment(part.SetFromElementLabels(setName,elDic[i]),secName)
    
    # Define output request
    mdl.FieldOutputRequest('SEDensity','Step-1',variables=('ELEDEN', ))
    mdl.HistoryOutputRequest('ExtWork','Step-1',variables=('ALLWK', ))
    
    Mdb.jobs['Job-1'].writeInput()
    # run prefilter in here
    Nds = part.nodes
    Fm={}
    Rmin = 3
    preFlt(Rmin,Elmts,Nds,Fm)
    #np.savetxt('Fm.csv',Fm,delimiter=',')
            
generateInp()

# Now run the prefilter to generate the prefilter matrix for this structural configuration.

