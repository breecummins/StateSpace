import numpy as np
import random
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops

def chooseLagsForSims(finaltime,tsprops=None,Tp=200,varchange=1):
    if tsprops == None:
        tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200 
    if varchange:       
        eqns,names,ts = rosslerVarChangeTS(finaltime)
    else:
        eqns,names,ts = rosslerTS(finaltime)
    Mlens = ( np.round( ts.shape[0]*tsprops ) ).astype(int)
    lags = SSR.chooseLags(ts,Mlens,Tp)
    return lags

def continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,numlags=3,fname=''):
    forwardconf, inverseconf, epsM1, epsM2, forwardprobs, inverseprobs = PM.convergenceWithContinuityTestFixedLagsFixedEps(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags[0][0],lags[0][1],tsprops=tsprops,epsprops=epsprops)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compinds[0]],names[compinds[1]])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compinds[0]],names[compinds[1]])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','numlags','lags','ts','tsprops','epsprops','epsM1','epsM2','forwardprobs','inverseprobs']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

def rosslerTS(finaltime=1200.0,dt=0.025):
    from Rossler import solveRossler
    timeseries = solveRossler([5.0,4.0,3.0],finaltime,dt)
    eqns = 'Rossler'
    names = ['s','u','v']
    return eqns,names,timeseries

def rosslerVarChangeTS(finaltime=1200.0,dt=0.025):
    from Rossler import solveRosslerVarChange
    timeseries = solveRosslerVarChange([5.0,4.0,3.0],finaltime,dt)
    eqns = 'Rossler with variable change'
    names = ['s','u','v']
    return eqns,names,timeseries

def runRossler(finaltime=1200.0,remote=1,varchange=1):
    print('Beginning batch run for Rossler equations....')
    if remote:
        basedir = '/home/bcummins/'
    else:
        basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/RosslerExample/'
    tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) 
    compind1 = [0,0,1]
    compind2 = [1,2,2]
    if varchange:
        eqns,names,ts = rosslerVarChangeTS(finaltime)
        basefname = 'RosslerVarChange_1200time_samelags_'
        lags= [[60,60],[60,60],[60,60]]
    else:
        eqns,names,ts = rosslerTS(finaltime)
        basefname = 'Rossler_1200time_mixedlags_'
        lags= [[60,60],[60,35],[60,35]]
    for k,c1 in enumerate(compind1):
        print('------------------------------------')
        print(names[c1] + ' and ' + names[compind2[k]])
        print('------------------------------------')
        compinds = [c1,compind2[k]]
        fname = basefname + names[c1] + names[compind2[k]] + '.pickle'
        continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,[lags[k]],fname=basedir+fname)

if __name__ == '__main__':
    runRossler()
    # chooseLagsForSims(1200.0)