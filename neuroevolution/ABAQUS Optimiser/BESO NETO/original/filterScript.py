# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:23:16 2019

Filter script.

Run this with a CAE database open to filter the sensitivity values.

@author: pqb18127
"""

import numpy as np
import json

## Function of filtering sensitivities
def fltAe(Ae,Fm):
    raw = Ae.copy()

    for el in Fm.keys():
        elInt = int(float(el))
        print(str(elInt))
        #print(Ae[elInt])
        Ae[elInt] = 0.0
        for i in range(len(Fm[el][0])): Ae[elInt]+=raw[Fm[el][0][i]]*Fm[el][1][i]
        
        
# Load sensitivities from csv
Ae = list(np.loadtxt('sens.csv'))
#Ae = np.loadtxt('sens.csv')
Ae = np.array([0] + Ae) # adds blank line to the top of the list to ensure that it is alligned correctly. (element labels dont start from zero and the list does)


# Load pre filter data
with open('Fm.json', 'r') as fp:
    Fm = json.load(fp)

fltAe(Ae,Fm)

# save sensitivities
Ae=list(Ae)
Ae.pop(0)

np.savetxt('filtSens.csv',Ae,delimiter=',')