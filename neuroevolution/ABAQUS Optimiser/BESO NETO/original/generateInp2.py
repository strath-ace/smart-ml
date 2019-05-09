# -*- coding: utf-8 -*-

# slb and vlb are lists of the element numbers that should be assigned to each section.

# This script takes an abaqus cae model and writes an input file based on the list of element densities described in 'elDen.csv'

import numpy as np
import step
import json



def fmtMdb(Mdb):
    mdl = Mdb.models['Model-1']
    part = mdl.parts['Part-1']
    # Build sections and assign solid section
    mdl.Material('Material01').Elastic(((1.0, 0.3), ))
    mdl.HomogeneousSolidSection('sldSec','Material01')
    mdl.Material('Material02').Elastic(((0.001**3, 0.3), ))
    mdl.HomogeneousSolidSection('voidSec','Material02')
    part.SectionAssignment(part.Set('ss',part.elements),'sldSec') # is there a vs equivalence of this
    # Define output request
    mdl.FieldOutputRequest('SEDensity','Step-1',variables=('ELEDEN', ))
    mdl.HistoryOutputRequest('ExtWork','Step-1',variables=('ALLWK', ))


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
    fmtMdb(Mdb)
    mdl = Mdb.models['Model-1']
    part = mdl.parts['Part-1']
    Elmts = part.elements
    Xe=np.ones(len(Elmts)) # should load this in
    np.savetxt('elDen.csv',Xe,delimiter=',')
    vlb, slb = [], []
    for el in Elmts:
        if Xe[el.label-1] == 1.0: slb.append(el.label) # TODO: check that the index for this is correct.
        else: vlb.append(el.label)
    if len(vlb)>0 and len(slb)>0:
        part.SectionAssignment(part.SetFromElementLabels('ss',slb),'sldSec')
        part.SectionAssignment(part.SetFromElementLabels('vs',vlb),'voidSec') 
    elif len(slb)>0 and len(vlb)==0:
        part.SectionAssignment(part.SetFromElementLabels('ss',slb),'sldSec')
    elif len(slb)==0 and len(vlb)>0: 
        part.SectionAssignment(part.SetFromElementLabels('vs',vlb),'voidSec')
    Mdb.jobs['Job-1'].writeInput()
    # run prefilter in here
    Nds = part.nodes
    Fm={}
    Rmin = 3
    preFlt(Rmin,Elmts,Nds,Fm)
    #np.savetxt('Fm.csv',Fm,delimiter=',')
    with open('Fm.json', 'w') as fp:
        json.dump(Fm, fp)
    
    
    
            
generateInp()

# Now run the prefilter to generate the prefilter matrix for this structural configuration.

