import numpy as np
import os
import matplotlib.pyplot as plt
import StateSpace.StateSpaceReconstructionPlots as SSRPlots
import StateSpace.StateSpaceReconstruction as SSR
import processPhysData as PD
import StateSpace.PecoraViz as PV


def plotContConf():
    basedir=os.path.join(os.path.expanduser("~"),'SimulationResults/TimeSeries/HeartRateData/')
    for fname in os.listdir(basedir):
        if '.pickle' in fname and 'lag100' in fname:
            print(fname)
            PV.plotContinuityConfWrapper(basedir,fname,[0,0])
            PV.plotContinuityConfWrapper_SaveFigs(basedir,fname,[0,0])

def plotRates(ts,start=0,stop=None):
    if not stop:
        stop = ts.shape[0]
    timeindex = range(start,stop)
    plt.plot(timeindex,ts[start:stop,0],'r')
    plt.hold('on')
    plt.plot(timeindex,ts[start:stop,1],'b')
    plt.legend(['Heart rate','Breathing rate'])
    plt.xlim([start,stop])
    plt.show()

def autocorrViz(ts):
    autocorr0 = SSR.getAutocorrelation(ts[:,0],int(0.1*ts.shape[0]))
    autocorr1 = SSR.getAutocorrelation(ts[:,1],int(0.1*ts.shape[0]))
    SSRPlots.plotAutocorrelation(autocorr0,"heart rate autocorr")
    SSRPlots.plotAutocorrelation(autocorr1,"breathing rate autocorr")

if __name__ == "__main__":
    # plotContConf()
    ts = PD.extractRates()
    plotRates(ts)
    plotRates(ts,90,720)
    for l in [5,50,100]:
        SSRPlots.plotShadowManifold(ts[90:720,0], 3, l, show=0,hold=0,style='b-',titlestr='Heart rate, lag {0}'.format(l),scatter=False, color=None,smooth=1)
        SSRPlots.plotShadowManifold(ts[90:720,1], 3, l, show=1,hold=0,style='b-',titlestr='Breathing rate, lag {0}'.format(l),scatter=False, color=None,smooth=1)
     # autocorrViz(ts[90:720,:])

