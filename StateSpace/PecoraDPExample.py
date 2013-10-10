import numpy as np
import random
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops

def chooseLagsForSims(finaltime,tsprops=None,Tp=150):
    if tsprops == None:
        tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200        
    eqns,names,ts = doublependulumTS(finaltime)
    Mlens = ( np.round( ts.shape[0]*tsprops ) ).astype(int)
    lags = SSR.chooseLags(ts,Mlens,Tp)
    return lags

def continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,numlags=4,fname=''):
    forwardconf, inverseconf, epsM1, epsM2, forwardprobs, inverseprobs = PM.convergenceWithContinuityTestFixedLagsFixedEps(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags[0][0],lags[0][1],tsprops=tsprops,epsprops=epsprops)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compinds[0]],names[compinds[1]])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compinds[0]],names[compinds[1]])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','numlags','lags','ts','tsprops','epsprops','epsM1','epsM2','forwardprobs','inverseprobs']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

def doublependulumTS(finaltime=600.0,dt=0.025):
    from DoublePendulum import solvePendulum
    timeseries = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt)
    eqns = 'Double pendulum'
    names = ['x','y','z','w']
    return eqns,names,timeseries

def runDP(epsprops,compinds,fname,lags=[[100,100]],basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPpaperexample/',finaltime=1200.0):
    eqns,names,ts = doublependulumTS(finaltime)
    tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 

def remoteRun_DP(finaltime=1200.0):
    print('Beginning batch run for double pendulum equations....')
    basedir = '/home/bcummins/'
    epsprops1=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) 
    epsprops2=np.array([0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.0075]) #for x and y
    names = ['x','y','z','w']
    compind1 = [0,0,0,1,1,2]
    compind2 = [1,2,3,2,3,3]
    basefname = 'DP_1200time_numlags4_fixedlags_fixedeps_'
    lags= [[100,100],[100,115],[100,100],[100,115],[100,100],[115,100]]
    for k,c1 in enumerate(compind1):
        print('------------------------------------')
        print(names[c1] + ' and ' + names[compind2[k]])
        print('------------------------------------')
        compinds = [c1,compind2[k]]
        fname = basefname + names[c1] + names[compind2[k]] + '.pickle'
        if k > 0:
            runDP(epsprops1,compinds,fname,[lags[k]],basedir=basedir,finaltime=finaltime)
        else:
            runDP(epsprops2,compinds,fname,[lags[k]],basedir=basedir,finaltime=finaltime)

if __name__ == '__main__':
    remoteRun_DP(1200.0)
    # chooseLagsForSims(1200.0)