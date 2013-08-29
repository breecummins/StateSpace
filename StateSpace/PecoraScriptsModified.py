import numpy as np
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops

def lorenzTS(finaltime=80.0,dt=0.01):
    from LorenzEqns import solveLorenz
    timeseries = solveLorenz([1.0,0.5,0.5],finaltime,dt)
    eqns = 'Lorenz'
    names = ['x','y','z']
    return eqns,names,timeseries

def doublependulumTS(finaltime=600.0,dt=0.025):
    from DoublePendulum import solvePendulum
    timeseries = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt)
    eqns = 'Double pendulum'
    names = ['x','y','z','w']
    return eqns,names,timeseries

def doublependulummodifiedTS(finaltime=600.0,dt=0.025):
    from DoublePendulumModified import solvePendulum
    timeseries = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt)
    eqns = 'Double pendulum modified'
    names = ['x','y','z','w']
    return eqns,names,timeseries

def doublependulummodifiedTS_withnoise(finaltime=600.0,dt=0.025):
    from DoublePendulumModified import solvePendulum
    timeseries = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt)
    for j in range(timeseries.shape[1]):
        s = np.std(timeseries[:,j])
        timeseries[:,j] += -0.01*s + 0.02*s*np.random.random(timeseries[:,j].shape)
    eqns = 'Double pendulum with noise'
    names = ['x','y','z','w']
    return eqns,names,timeseries

def continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname='',numlags=5):
    '''
    Will work with any of the equations, Lorenz, double pendulum, or modified double pendulum.

    '''
    if len(lags) == 1:
        forwardconf, inverseconf, epsM1, epsM2, forwardprobs, inverseprobs = PM.convergenceWithContinuityTestFixedLags(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags[0][0],lags[0][1],tsprops=tsprops,epsprops=epsprops)
    else:
        forwardconf, inverseconf, epsM1, epsM2, forwardprobs, inverseprobs = PM.convergenceWithContinuityTestMultipleLags(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags,tsprops=tsprops,epsprops=epsprops)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compinds[0]],names[compinds[1]])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compinds[0]],names[compinds[1]])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','numlags','lags','ts','tsprops','epsprops','epsM1','epsM2','forwardprobs','inverseprobs']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

def continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname='',numlags=5):
    '''
    Will work with any of the equations, Lorenz, double pendulum, or modified double pendulum.

    '''
    if len(lags) == 1:
        forwardconf, inverseconf, epsM1, epsM2, forwardprobs, inverseprobs = PM.convergenceWithContinuityTestFixedLagsFixedEps(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags[0][0],lags[0][1],tsprops=tsprops,epsprops=epsprops)
    else:
        forwardconf, inverseconf, epsM1, epsM2, forwardprobs, inverseprobs = PM.convergenceWithContinuityTestMultipleLagsFixedEps(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags,tsprops=tsprops,epsprops=epsprops)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compinds[0]],names[compinds[1]])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compinds[0]],names[compinds[1]])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','numlags','lags','ts','tsprops','epsprops','epsM1','epsM2','forwardprobs','inverseprobs']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

def testLagsAtDifferentLocationsAndLengths(finaltime=1200.0):
    '''
    Only for modified double pendulum.

    '''
    eqns,names,ts = doublependulummodifiedTS(finaltime)
    tsprops = [0.1,0.2,0.4,0.6,0.8] 
    nums = [40]
    compind1=2
    for p in tsprops:
        L = int(round(p*ts.shape[0]))
        for N in nums:
            lags = SSR.testLagsWithDifferentChunks(ts[:,compind1],L,N,int(0.5*L))
            print('z: {0} trials of length {1} in a length {2} timeseries.'.format(N,L,ts.shape[0]))
            print( ( np.min(lags), np.max(lags), np.mean(lags) ) )
    # compind1=3
    # for p in tsprops:
    #     L = int(round(p*ts.shape[0]))
    #     for N in nums:
    #         lags = SSR.testLagsWithDifferentChunks(ts[:,compind1],L,N)
    #         print('w: {0} trials of length {1} in a length {2} timeseries.'.format(N,L,ts.shape[0]))
    #         print( ( np.min(lags), np.max(lags), np.mean(lags) ) )

def chooseLagsForSims(compinds,finaltime,tsprops,Tp=None):
    '''
    Only for modified double pendulum.

    '''
    eqns,names,ts = doublependulummodifiedTS(finaltime)
    Mlens = ( np.round( ts.shape[0]*tsprops ) ).astype(int)
    lags = SSR.chooseLags(ts[:,compinds[0]],ts[:,compinds[1]],Mlens,Tp)
    return lags

def getTS(finaltime):
    '''
    Only for modified double pendulum.

    '''
    eqns,names,ts = doublependulummodifiedTS(finaltime)
    tsprops = np.arange(0.5,0.85,0.1) # for finaltime = 1200
    # tsprops = np.arange(0.3,0.85,0.1) # for finaltime = 2400
    return eqns,names,ts,tsprops

def localRun_zw(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/',finaltime=1200.0):
    '''
    Only for modified double pendulum.

    '''
    eqns,names,ts,tsprops = getTS(finaltime)
    epsprops=np.array([0.005,0.0075,0.01,0.0125,0.015,0.02,0.04]) #for z and w
    compinds = [2,3]
    lags = [[5600,5600]] #fixed lags
    # lags = [[int(0.15*t*ts.shape[0])]*2 for t in tsprops] #changing lags
    fname = 'DPMod_1200time_samefixedlags_zw.pickle'
    #changing eps
    continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 
    #fixed eps
    # continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 

def localRun_xw(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/',finaltime=1200.0):
    '''
    Only for modified double pendulum.

    '''
    eqns,names,ts,tsprops = getTS(finaltime)
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for x and w
    compinds = [0,3]
    lags = [[100,5600]] #fixed lags
    # lags = [[100,int(0.15*t*ts.shape[0])] for t in tsprops] #changing lags
    fname = 'DPMod_1200time_difffixedlags_xw.pickle'
    #changing eps
    continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)
    #fixed eps
    # continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

def localRun_xy(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/',finaltime=1200.0):
    '''
    Only for modified double pendulum.

    '''
    eqns,names,ts,tsprops = getTS(finaltime)
    epsprops=np.array([0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.0075]) #for x and y
    compinds = [0,1]
    lags = [[100,100]] #fixed lags
    fname = 'DPMod_1200time_samefixedlags_xy.pickle'
    #changing eps
    continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)
    #fixed eps
    # continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

def getTS_withnoise(finaltime):
    '''
    Only for modified double pendulum with noise.

    '''
    eqns,names,ts = doublependulummodifiedTS_withnoise(finaltime)
    tsprops = np.arange(0.5,0.85,0.1) # for finaltime = 1200
    # tsprops = np.arange(0.3,0.85,0.1) # for finaltime = 2400
    return eqns,names,ts,tsprops

def localRun_zw_withnoise(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/',finaltime=1200.0):
    '''
    Only for modified double pendulum with noise.

    '''
    eqns,names,ts,tsprops = getTS_withnoise(finaltime)
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]) #for z and w
    compinds = [2,3]
    lags = [[5600,5600]] #fixed lags
    # lags = [[int(0.15*t*ts.shape[0])]*2 for t in tsprops] #changing lags
    fname = 'DPMod_1200time_withnoise_samefixedlags_zw.pickle'
    #changing eps
    continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 
    #fixed eps
    # continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 

def localRun_xw_withnoise(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/',finaltime=1200.0):
    '''
    Only for modified double pendulum with noise.

    '''
    eqns,names,ts,tsprops = getTS_withnoise(finaltime)
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6]) #for x and w
    compinds = [0,3]
    lags = [[100,5600]] #fixed lags
    # lags = [[100,int(0.15*t*ts.shape[0])] for t in tsprops] #changing lags
    fname = 'DPMod_1200time_withnoise_difffixedlags_xw.pickle'
    #changing eps
    continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)
    #fixed eps
    # continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

def localRun_xy_withnoise(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/',finaltime=1200.0):
    '''
    Only for modified double pendulum with noise.

    '''
    eqns,names,ts,tsprops = getTS_withnoise(finaltime)
    epsprops=np.array([0.001,0.005,0.01,0.05,0.1,0.2,0.5]) #for x and y
    compinds = [0,1]
    lags = [[100,100]] #fixed lags
    fname = 'DPMod_1200time_withnoise_samefixedlags_xy.pickle'
    #changing eps
    continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)
    #fixed eps
    # continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

def remoteRun(finaltime):
    '''
    Only for modified double pendulum.

    '''
    print('Beginning batch run for modified double pendulum equations....')
    basedir = '/home/bcummins/'
    print('------------------------------------')
    print('z and w')
    print('------------------------------------')
    localRun_zw(basedir,finaltime)
    print('------------------------------------')
    print('x and w')
    print('------------------------------------')
    localRun_xw(basedir,finaltime)
    print('------------------------------------')
    print('x and y')
    print('------------------------------------')
    localRun_xy(basedir,finaltime)

def remoteRun_withnoise(finaltime):
    '''
    Only for modified double pendulum.

    '''
    print('Beginning batch run for modified double pendulum equations with noise....')
    basedir = '/home/bcummins/'
    print('------------------------------------')
    print('z and w')
    print('------------------------------------')
    localRun_zw_withnoise(basedir,finaltime)
    print('------------------------------------')
    print('x and w')
    print('------------------------------------')
    localRun_xw_withnoise(basedir,finaltime)
    print('------------------------------------')
    print('x and y')
    print('------------------------------------')
    localRun_xy_withnoise(basedir,finaltime)

if __name__ == '__main__':
    remoteRun_withnoise(1200.0)
    # remoteRun(1200.0)
    # ###################
    # compinds = [2,3]
    # finaltime = 2400.0
    # tsprops = np.arange(0.2,0.95,0.1)
    # lags = chooseLagsForSims(compinds,finaltime,tsprops,0.5)
    # ###################
    # localRun_zw()
    # testLagsAtDifferentLocationsAndLengths(2400.0)
