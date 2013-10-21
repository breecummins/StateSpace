#!/usr/bin/sh python

import PecoraViz as PV
import os

basedir=os.path.join(os.path.expanduser("~"),'SimulationResults/TimeSeries/PecoraMethod/FinalPaperExamples/')
for fname in os.listdir(basedir):
    if '.png' not in fname and 'd0265' in fname:
        # PV.plotContinuityConfWrapper(basedir,fname,[0,0])
        PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])

# basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPpaperexample/'
# for fname in os.listdir(basedir):
#     if '.png' not in fname:
#         PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])

# basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/Diamondpaperexample/'
# for fname in os.listdir(basedir):
#     if '.png' not in fname and 'Mult' in fname:
#         PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])

# basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DoublePendulumDiamondExample/'
# for fname in os.listdir(basedir):
#     if '.png' not in fname:
#         PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])

# basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/RosslerExample/'
# for fname in os.listdir(basedir):
#     if '.png' not in fname and 'Harder' in fname:
#         PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])

# basedir=os.path.join(os.path.expanduser("~"),'SimulationResults/TimeSeries/PecoraMethod/LorenzExample/')
# for fname in os.listdir(basedir):
#     if '.png' not in fname and 'Rotated' in fname:
#         PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])
