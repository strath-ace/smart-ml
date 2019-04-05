# -*- coding: utf-8 -*-

# slb and vlb are lists of the element numbers that should be assigned to each section.

# This script takes an abaqus cae model and writes an input file based on the list of element densities described in 'elDen.csv'

import numpy as np
import step


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


def generateInp(Xe):
    Mdb = openMdb('cantilever.cae')
    fmtMdb(Mdb)
    mdl = Mdb.models['Model-1']
    part = mdl.parts['Part-1']
    Elmts = part.elements
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
    

Xe = np.loadtxt('elDen.csv')
generateInp(Xe)
