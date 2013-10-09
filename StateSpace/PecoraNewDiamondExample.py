import numpy as np
import random
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops

def chooseLagsForSims(finaltime,tsprops=None,Tp=400):
    if tsprops == None:
        tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200  
    eqns,names,ts = newDiamondTS(finaltime)
    Mlens = ( np.round( ts.shape[0]*tsprops ) ).astype(int)
    lags = SSR.chooseLags(ts,Mlens,Tp)
    return lags

def continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,numlags,fname=''):
    forwardconf, inverseconf, epsM1, epsM2, forwardprobs, inverseprobs = PM.convergenceWithContinuityTestFixedLagsFixedEps(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags[0],lags[1],tsprops=tsprops,epsprops=epsprops)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compinds[0]],names[compinds[1]])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compinds[0]],names[compinds[1]])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','numlags','lags','ts','tsprops','epsprops','epsM1','epsM2','forwardprobs','inverseprobs']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

def newDiamondTS(finaltime=1200.0,dt=0.025):
    from Rossler import solveDiamondRotatedInteralMult
    timeseries = solveDiamondRotatedInteralMult([1.0,2.0,3.0,2.0,5.0,4.0,3.0,0.75],finaltime,dt)
    eqns = 'Diamond with internal mult by y'
    names = ['x','y','z','w','s','u','v','p']
    return eqns,names,timeseries

def runDiamond(finaltime=1200.0,remote=1):
    print('Beginning batch run for diamond equations....')
    if remote:
        basedir = '/home/bcummins/'
    else:
        basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/Diamondpaperexample/'
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) 
    names = ['x','y','z','w','s','u','v','p']
    compind1 = [0,0,0,0,1,1,1,1,4,4,4,5,5,7,7,7]
    compind2 = [4,5,6,7,4,5,6,7,5,6,7,6,7,6,3,2]
    tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200
    numlags = 8
    eqns,names,ts = newDiamondTS(finaltime)
    basefname = 'DiamondInternalMult_1200time_mixedlags_'
    lags= [[100,60],[100,60],[100,60],[100,185],[100,60],[100,60],[100,60],[100,185],[60,60],[60,60],[60,185],[60,60],[60,185],[185,60],[185,100],[185,115]]
    for k,c1 in enumerate(compind1):
        print('------------------------------------')
        print(names[c1] + ' and ' + names[compind2[k]])
        print('------------------------------------')
        compinds = [c1,compind2[k]]
        fname = basefname + names[c1] + names[compind2[k]] + '.pickle'
        continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags[k],numlags,fname=basedir+fname) 

if __name__ == '__main__':
    runDiamond()
    # chooseLagsForSims(1200.0)
    # #below, choose lags from autocorrelation
    # import StateSpaceReconstructionPlots as SSRPlots
    # eqns,names,ts = newDiamondTS(1200.0)
    # T = 400
    # autocorr = SSR.getAutocorrelation(ts[:,7],T)
    # SSRPlots.plotAutocorrelation(autocorr,'p')
