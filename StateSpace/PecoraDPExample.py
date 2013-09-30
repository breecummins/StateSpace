import numpy as np
import random
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops
import PecoraViz as PV

def continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname='',numlags=5):
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

def getTSDP(finaltime):
    eqns,names,ts = doublependulumTS(finaltime)
    tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200
    return eqns,names,ts,tsprops

def localRun_zw_DP(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPpaperexample/',finaltime=1200.0):
    eqns,names,ts,tsprops = getTSDP(finaltime)
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for z and w
    compinds = [2,3]
    lags = [[100,100]] #fixed lags
    fname = 'DP_1200time_samefixedlags_fixedeps_zw.pickle'
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 

def localRun_xw_DP(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPpaperexample/',finaltime=1200.0):
    eqns,names,ts,tsprops = getTSDP(finaltime)
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for x and w
    compinds = [0,3]
    lags = [[100,100]] #fixed lags
    fname = 'DP_1200time_samefixedlags_fixedeps_xw.pickle'
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

def localRun_xy_DP(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPpaperexample/',finaltime=1200.0):
    eqns,names,ts,tsprops = getTSDP(finaltime)
    epsprops=np.array([0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.0075]) #for x and y
    compinds = [0,1]
    lags = [[100,100]] #fixed lags
    fname = 'DP_1200time_samefixedlags_fixedeps_xy.pickle'
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

def remoteRun_DP(finaltime):
    print('Beginning batch run for double pendulum equations....')
    basedir = '/home/bcummins/'
    print('------------------------------------')
    print('z and w')
    print('------------------------------------')
    localRun_zw_DP(basedir,finaltime)
    print('------------------------------------')
    print('x and w')
    print('------------------------------------')
    localRun_xw_DP(basedir,finaltime)
    print('------------------------------------')
    print('x and y')
    print('------------------------------------')
    localRun_xy_DP(basedir,finaltime)

if __name__ == '__main__':
    remoteRun_DP(1200.0)
    # PV.plotContinuityConfWrapper(basedir,fname)
