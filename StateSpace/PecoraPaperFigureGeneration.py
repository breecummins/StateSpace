#!/usr/bin/sh python

import PecoraViz as PV
import os

basedir=os.path.join(os.path.expanduser("~"),'SimulationResults/TimeSeries/PecoraMethod/FinalPaperExamples/')

# for fname in os.listdir(basedir):
#     if '.png' not in fname and 'noise' in fname:
#         # PV.plotContinuityConfWrapper(basedir,fname,[0,0])
#         PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])

noiseprop = [0.01,0.02,0.03,0.04]
for s in ['xy','xz','xw','yz','yw','zw']:
    fnames = []
    for f in os.listdir(basedir):
        if '.png' not in f and s in f and 'noise' in f:
            fnames.append(f)
    PV.plotContinuityConfWrapper_SaveFigs_Noise(basedir,fnames,noiseprop,[0,0])

