# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
from odbAccess import *
import visualization
import sys

# This script is used to retrieve data from an abaqus .odb file.
# Useful info: https://stackoverflow.com/questions/22750272/can-i-pass-command-line-arguments-to-abaqus-python 
# https://stackoverflow.com/questions/35723362/error-creating-writefieldreport-results-in-abaqus

def getDisps1(i):

    o1= session.openOdb(name='C:/Users/pqb18127/OneDrive/PhD/Python/DEAP/multiprocessing/MP Abaqus and DEAP Test/analysis_' + str(i) + '/Job-1.odb')
    session.viewports['Viewport: 1'].setValues(displayedObject=o1)
    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=1)
    odb = session.odbs['C:/Users/pqb18127/OneDrive/PhD/Python/DEAP/multiprocessing/MP Abaqus and DEAP Test/analysis_' + str(i) + '/Job-1.odb']
    session.fieldReportOptions.setValues(reportFormat=COMMA_SEPARATED_VALUES)
    session.writeFieldReport(fileName='outputData' + str(i) + '.csv', append=OFF, 
        sortItem='Node Label', odb=odb, step=0, frame=1, outputPosition=NODAL, 
        variable=(('U', NODAL), ))

if __name__ == '__main__':

    i = sys.argv[-1]
    getDisps1(i)
