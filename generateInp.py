# -*- coding: utf-8 -*-

# slb and vlb are lists of the element numbers that should be assigned to each section.
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
    part.SectionAssignment(part.Set('ss',part.elements),'sldSec')
    # Define output request
    mdl.FieldOutputRequest('SEDensity','Step-1',variables=('ELEDEN', ))
    mdl.HistoryOutputRequest('ExtWork','Step-1',variables=('ALLWK', ))

def generateInp(slb,vlb):
    Mdb = openMdb('cantilever.cae')
    fmtMdb(Mdb)
    mdl = Mdb.models['Model-1']
    part = mdl.parts['Part-1']
    part.SectionAssignment(part.SetFromElementLabels('ss',slb),'sldSec')
    part.SectionAssignment(part.SetFromElementLabels('vs',vlb),'voidSec')
    Mdb.jobs['Job-1'].writeInput()
    
slb = np.loadtxt('slb.csv',dtype='int',delimiter=',') # 
vlb = np.loadtxt('vlb.csv',dtype='int',delimiter=',') # would be nice to have slb and vlb saved in the same file and loaded in together.

generateInp(slb,vlb)

#os.system('abaqus cae noGui=generateInp.py')

# need to open the correct model database for this to runn 