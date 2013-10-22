#!/usr/bin/sh python

import numpy as np
np.set_printoptions(linewidth=125)
import os
import fileops

basedir=os.path.join(os.path.expanduser("~"),'SimulationResults/TimeSeries/PecoraMethod/FinalPaperExamples/')

# double pendulum example
names = ['x','y','z','w']
confmatrix = np.zeros((len(names),len(names)))
for fname in os.listdir(basedir):
    if '.png' not in fname and 'numlags4' in fname:
        ind1=names.index(fname[-9])
        ind2=names.index(fname[-8])
        outdict = fileops.loadPickle(basedir+fname)
        confmatrix[ind1,ind2] = outdict['forwardconf'][-1][-1]
        confmatrix[ind2,ind1] = outdict['inverseconf'][-1][-1]
print("      ".join(names))
for row in confmatrix:
    print(", ".join(["{:0.2f}".format(x).replace("'","") for x in row]))

# diamond example
names = ['x','y','z','w','s','u','v','p']
confmatrix = np.zeros((len(names),len(names)))
for fname in os.listdir(basedir):
    if '.png' not in fname and 'd020' in fname:
        ind1=names.index(fname[-9])
        ind2=names.index(fname[-8])
        outdict = fileops.loadPickle(basedir+fname)
        confmatrix[ind1,ind2] = outdict['forwardconf'][-1][-1]
        confmatrix[ind2,ind1] = outdict['inverseconf'][-1][-1]
print("      ".join(names))
for row in confmatrix:
    print(", ".join(["{:0.2f}".format(x).replace("'","") for x in row]))


