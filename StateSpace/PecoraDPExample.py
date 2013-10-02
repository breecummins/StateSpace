import numpy as np
import random
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops
import PecoraViz as PV

def chooseLagsForSims(finaltime,tsprops=None,Tp=150):
    if tsprops == None:
        tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200        
    eqns,names,ts = doublependulumTS(finaltime)
    Mlens = ( np.round( ts.shape[0]*tsprops ) ).astype(int)
    lags = SSR.chooseLags(ts,Mlens,Tp)
    return lags

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

def runDP(epsprops,compinds,fname,lags=[[100,100]],basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPpaperexample/',finaltime=1200.0):
    eqns,names,ts = doublependulumTS(finaltime)
    tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 

def remoteRun_DP(finaltime):
    print('Beginning batch run for double pendulum equations....')
    basedir = '/home/bcummins/'
    print('------------------------------------')
    print('x and y')
    print('------------------------------------')
    epsprops=np.array([0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.0075]) #for x and y
    compinds = [0,1]
    fname = 'DP_1200time_samefixedlags_fixedeps_xy.pickle'
    runDP(epsprops,compinds,fname,basedir=basedir,finaltime=finaltime)
    print('------------------------------------')
    print('z and w')
    print('------------------------------------')
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for z and w
    compinds = [2,3]
    fname = 'DP_1200time_samefixedlags_fixedeps_zw.pickle'
    runDP(epsprops,compinds,fname,basedir=basedir,finaltime=finaltime)
    print('------------------------------------')
    print('x and w')
    print('------------------------------------')
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for x and w
    compinds = [0,3]
    fname = 'DP_1200time_samefixedlags_fixedeps_xw.pickle'
    runDP(epsprops,compinds,fname,basedir=basedir,finaltime=finaltime)
    print('------------------------------------')
    print('y and w')
    print('------------------------------------')
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for x and w
    compinds = [1,3]
    fname = 'DP_1200time_samefixedlags_fixedeps_yw.pickle'
    runDP(epsprops,compinds,fname,basedir=basedir,finaltime=finaltime)
    print('------------------------------------')
    print('x and z')
    print('------------------------------------')
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for x and w
    compinds = [0,2]
    fname = 'DP_1200time_samefixedlags_fixedeps_xz.pickle'
    runDP(epsprops,compinds,fname,basedir=basedir,finaltime=finaltime)
    print('------------------------------------')
    print('y and z')
    print('------------------------------------')
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for x and w
    compinds = [1,2]
    fname = 'DP_1200time_samefixedlags_fixedeps_yz.pickle'
    runDP(epsprops,compinds,fname,basedir=basedir,finaltime=finaltime)

if __name__ == '__main__':
    # remoteRun_DP(1200.0)
    chooseLagsForSims(1200.0)
