import numpy as np
import random
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops

def chooseLagsForSims(finaltime,tsprops=None,Tp=200,varchange=0,driven=0,rotated=1):
    if tsprops == None:
        tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200 
    if varchange:       
        eqns,names,ts = rosslerVarChangeTS(finaltime)
    elif driven:
        eqns,names,ts = drivenRosslerTS(finaltime)
    elif rotated:
        eqns,names,ts = rotatedRosslerTS(finaltime)
    else:
        eqns,names,ts = rosslerTS(finaltime)
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

def rosslerTS(finaltime=1200.0,dt=0.025):
    from Rossler import solveRossler
    timeseries = solveRossler([5.0,4.0,3.0],finaltime,dt)
    eqns = 'Rossler'
    names = ['s','u','v']
    return eqns,names,timeseries

def drivenRosslerTS(finaltime=1200.0,dt=0.025):
    from Rossler import solveDrivenRossler
    timeseries = solveDrivenRossler([5.0,4.0,3.0],finaltime,dt)
    eqns = 'Driven Rossler'
    names = ['s','u','v']
    return eqns,names,timeseries

def rotatedRosslerTS(finaltime=1200.0,dt=0.025):
    from Rossler import solveRotatedRossler
    timeseries = solveRotatedRossler([5.0,4.0,3.0],finaltime,dt)
    eqns = 'Rotated Rossler'
    names = ['s','u','v']
    return eqns,names,timeseries

def rosslerVarChangeTS(finaltime=1200.0,dt=0.025):
    from Rossler import solveRosslerVarChange
    timeseries = solveRosslerVarChange([5.0,4.0,3.0],finaltime,dt)
    eqns = 'Rossler with variable change'
    names = ['s','u','v']
    return eqns,names,timeseries

def runRossler(finaltime=1200.0,remote=1,varchange=0,driven=0,rotated=1):
    print('Beginning batch run for Rossler equations....')
    if remote:
        basedir = '/home/bcummins/'
    else:
        basedir=os.path.join(os.path.expanduser("~"),'SimulationResults/TimeSeries/PecoraMethod/RosslerExample/')
    tsprops = np.arange(0.3,1.05,0.1) # for finaltime = 1200
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) 
    numlags=3
    compind1 = [0,0,1]
    compind2 = [1,2,2]
    if varchange:
        eqns,names,ts = rosslerVarChangeTS(finaltime)
        basefname = 'RosslerVarChange_1200time_samelags_'
        lags= [[60,60],[60,60],[60,60]]
    elif driven:
        eqns,names,ts = drivenRosslerTS(finaltime)
        basefname = 'DrivenHarderRossler_1200time_samelags_'
        lags= [[60,60],[60,60],[60,60]]
    elif rotated:
        eqns,names,ts = rotatedRosslerTS(finaltime)
        basefname = 'RotatedRossler_1200time_samelags_'
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
        continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags[k],numlags,fname=basedir+fname)

if __name__ == '__main__':
    runRossler()
    # chooseLagsForSims(1200.0)