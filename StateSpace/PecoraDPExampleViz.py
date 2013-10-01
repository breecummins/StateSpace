#!/usr/bin/sh python

import PecoraViz as PV
import os

basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPpaperexample/'
for fname in os.listdir(basedir):
    PV.plotContinuityConfWrapper(basedir,fname,[0,0])
