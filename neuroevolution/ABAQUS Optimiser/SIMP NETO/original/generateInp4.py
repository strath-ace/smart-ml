# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Replaces generateInp3

"""
    
import numpy as np
import step
from collections import Counter

def generateInp(Xe):
    Mdb = openMdb('cantilever.cae')
    
    # Replace format database stuff.
    mdl = Mdb.models['Model-1']
    part = mdl.parts['Part-1']
    Elmts = part.elements
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

Xe = np.loadtxt('elDenNew.csv')
generateInp(Xe)