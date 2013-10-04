#!/usr/bin/sh python

import PecoraViz as PV
import os

# basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPpaperexample/'
# for fname in os.listdir(basedir):
#     if '.png' not in fname:
#         PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])

# basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/Diamondpaperexample/'
# for fname in os.listdir(basedir):
#     if '.png' not in fname and 'Capped' in fname:
#         PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])

basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/RosslerExample/'
for fname in os.listdir(basedir):
    if '.png' not in fname and 'Rotated' in fname:
        PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])

# basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/LorenzExample/'
# for fname in os.listdir(basedir):
#     if '.png' not in fname:
#         PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])
