import numpy as np
import random
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

def doublependulummodifiedTS_changedbeta(finaltime=600.0,dt=0.025):
    from DoublePendulumModified import solvePendulum
    timeseries = solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt,beta=1.2)
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

def twoLinesScrambledNoisy(skip=0,noise=0):
    times = np.arange(0,2.0,0.01)
    line1 = 0.1*times
    line2 = -0.7*times
    std = np.std(line2)
    line2 = line2 - noise*std + 2*noise*std*np.random.random(line2.shape)
    if skip > 0:
        newinds = random.sample(range(0,len(line2),skip),int(len(line2)/skip))
        line2[::skip] = line2[newinds]
    return line1,line2

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

def chooseLagsForSimsDP(compinds,finaltime,tsprops,Tp=None):
    '''
    Only for double pendulum.

    '''
    eqns,names,ts = doublependulumTS(finaltime)
    Mlens = ( np.round( ts.shape[0]*tsprops ) ).astype(int)
    lags = SSR.chooseLags(ts[:,compinds[0]],ts[:,compinds[1]],Mlens,Tp)
    return lags

def getTSDP(finaltime):
    '''
    Only for double pendulum.

    '''
    eqns,names,ts = doublependulumTS(finaltime)
    tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200
    return eqns,names,ts,tsprops

def localRun_zw_DP(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPchangedbeta/',finaltime=1200.0):
    '''
    Only for double pendulum.

    '''
    eqns,names,ts,tsprops = getTSDP(finaltime)
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for z and w
    compinds = [2,3]
    lags = [[100,100]] #fixed lags
    # lags = [[int(0.15*t*ts.shape[0]),115] for t in tsprops] #changing lags
    fname = 'DP_1200time_samefixedlags_fixedeps_zw.pickle'
    #changing eps
    # continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 
    #fixed eps
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 

def localRun_xw_DP(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPchangedbeta/',finaltime=1200.0):
    '''
    Only for double pendulum.

    '''
    eqns,names,ts,tsprops = getTSDP(finaltime)
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for x and w
    compinds = [0,3]
    lags = [[100,100]] #fixed lags
    fname = 'DP_1200time_samefixedlags_fixedeps_xw.pickle'
    #changing eps
    # continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)
    #fixed eps
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

def localRun_xy_DP(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPchangedbeta/',finaltime=1200.0):
    '''
    Only for double pendulum.

    '''
    eqns,names,ts,tsprops = getTSDP(finaltime)
    epsprops=np.array([0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.0075]) #for x and y
    compinds = [0,1]
    lags = [[100,100]] #fixed lags
    fname = 'DP_1200time_samefixedlags_fixedeps_xy.pickle'
    #changing eps
    # continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)
    #fixed eps
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

def chooseLagsForSims(compinds,finaltime,tsprops,Tp=None):
    '''
    Only for modified double pendulum.

    '''
    eqns,names,ts = doublependulummodifiedTS(finaltime)
    Mlens = ( np.round( ts.shape[0]*tsprops ) ).astype(int)
    lags = SSR.chooseLags(ts[:,compinds[0]],ts[:,compinds[1]],Mlens,Tp)
    return lags

def chooseLagsForSims_changedbeta(compinds,finaltime,tsprops,Tp=None):
    '''
    Only for modified double pendulum.

    '''
    eqns,names,ts = doublependulummodifiedTS_changedbeta(finaltime)
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

def getTS_changedbeta(finaltime):
    '''
    Only for modified double pendulum.

    '''
    eqns,names,ts = doublependulummodifiedTS_changedbeta(finaltime)
    tsprops = np.arange(0.2,1.1,0.2) # for finaltime = 1200
    # tsprops = np.arange(0.3,0.85,0.1) # for finaltime = 2400
    return eqns,names,ts,tsprops

def localRun_zw(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPModified/',finaltime=1200.0):
    '''
    Only for modified double pendulum.

    '''
    # eqns,names,ts,tsprops = getTS(finaltime)
    eqns,names,ts,tsprops = getTS_changedbeta(finaltime)
    # epsprops=np.array([0.005,0.0075,0.01,0.0125,0.015,0.02,0.04]) #for z and w, old beta
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for z and w
    compinds = [2,3]
    # lags = [[5600,5600]] #fixed lags for old params
    lags = [[104,104]] #fixed lags for beta = 1.2
    # lags = [[int(0.15*t*ts.shape[0])]*2 for t in tsprops] #changing lags
    fname = 'DPMod_1200time_samefixedlags_fixedeps_beta1-2_zw.pickle'
    #changing eps
    # continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 
    #fixed eps
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 

def localRun_xw(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPModified/',finaltime=1200.0):
    '''
    Only for modified double pendulum.

    '''
    # eqns,names,ts,tsprops = getTS(finaltime)
    eqns,names,ts,tsprops = getTS_changedbeta(finaltime)
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) #for x and w
    compinds = [0,3]
    # lags = [[100,5600]] #fixed lags for old params
    lags = [[104,104]] #fixed lags for beta = 1.2
    # lags = [[100,int(0.15*t*ts.shape[0])] for t in tsprops] #changing lags
    fname = 'DPMod_1200time_difffixedlags_fixedeps_beta1-2_xw.pickle'
    #changing eps
    # continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)
    #fixed eps
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

def localRun_xy(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPModified/',finaltime=1200.0):
    '''
    Only for modified double pendulum.

    '''
    # eqns,names,ts,tsprops = getTS(finaltime)
    eqns,names,ts,tsprops = getTS_changedbeta(finaltime)
    epsprops=np.array([0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.0075]) #for x and y
    compinds = [0,1]
    # lags = [[100,100]] #fixed lags for old params
    lags = [[104,104]] #fixed lags for beta = 1.2
    fname = 'DPMod_1200time_samefixedlags_fixedeps_beta1-2_xy.pickle'
    #changing eps
    # continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)
    #fixed eps
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

def getTS_withnoise(finaltime):
    '''
    Only for modified double pendulum with noise.

    '''
    eqns,names,ts = doublependulummodifiedTS_withnoise(finaltime)
    tsprops = np.arange(0.5,0.85,0.1) # for finaltime = 1200
    # tsprops = np.arange(0.3,0.85,0.1) # for finaltime = 2400
    return eqns,names,ts,tsprops

def localRun_zw_withnoise(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPModified/withnoise/',finaltime=1200.0):
    '''
    Only for modified double pendulum with noise.

    '''
    eqns,names,ts,tsprops = getTS_withnoise(finaltime)
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]) #for z and w
    compinds = [2,3]
    lags = [[5600,5600]] #fixed lags
    # lags = [[int(0.15*t*ts.shape[0])]*2 for t in tsprops] #changing lags
    fname = 'DPMod_1200time_withnoise_samefixedlags_fixedeps_zw.pickle'
    #changing eps
    # continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 
    #fixed eps
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 

def localRun_xw_withnoise(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPModified/withnoise/',finaltime=1200.0):
    '''
    Only for modified double pendulum with noise.

    '''
    eqns,names,ts,tsprops = getTS_withnoise(finaltime)
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6]) #for x and w
    compinds = [0,3]
    lags = [[100,5600]] #fixed lags
    # lags = [[100,int(0.15*t*ts.shape[0])] for t in tsprops] #changing lags
    fname = 'DPMod_1200time_withnoise_difffixedlags_fixedeps_xw.pickle'
    #changing eps
    # continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)
    #fixed eps
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

def localRun_xy_withnoise(basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DPModified/withnoise/',finaltime=1200.0):
    '''
    Only for modified double pendulum with noise.

    '''
    eqns,names,ts,tsprops = getTS_withnoise(finaltime)
    epsprops=np.array([0.001,0.005,0.01,0.05,0.1,0.2,0.5]) #for x and y
    compinds = [0,1]
    lags = [[100,100]] #fixed lags
    fname = 'DPMod_1200time_withnoise_samefixedlags_fixedeps_xy.pickle'
    #changing eps
    # continuityTesting(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)
    #fixed eps
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname)

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
    Only for modified double pendulum with noise.

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

def localRun_DP(finaltime):
    '''
    Only for double pendulum.

    '''
    print('Beginning batch run for double pendulum equations....')
    print('------------------------------------')
    print('z and w')
    print('------------------------------------')
    localRun_zw_DP(finaltime=finaltime)
    print('------------------------------------')
    print('x and w')
    print('------------------------------------')
    localRun_xw_DP(finaltime=finaltime)
    print('------------------------------------')
    print('x and y')
    print('------------------------------------')
    localRun_xy_DP(finaltime=finaltime)

def remoteRun_DP(finaltime):
    '''
    Only for double pendulum.

    '''
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

def test2LinesScrambled():
    import PecoraViz as PV
    def makeOutput(line1,line2,summary):
        ts = np.array([line1,line2]).transpose()
        d = continuityTestingFixedEps('Two lines',['x','y'],ts,[0,1],np.arange(0.2,1.1,0.2),np.array([0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5]),[[1,1]],numlags=3)
        print('---------------------------------------------------------')
        print(summary)
        print('---------------------------------------------------------')
        print("Epsilon as proportions of (changing) standard deviations: {0}.".format(d['epsprops']))
        print("Forward continuity confidence: ")
        print(d['forwardconf'])
        print("Inverse continuity confidence: ")
        print(d['inverseconf'])
        PV.plotOutput(d['forwardconf'],d['inverseconf'],d['epsprops'],d['tsprops'],len(line1),d['forwardtitle'],d['inversetitle'],logs = [0,0])
    summary = "Two lines, complete scrambling."
    line1,line2 = twoLinesScrambledNoisy(1)
    makeOutput(line1,line2,summary)
    summary = "Two lines, one-half scrambling."
    line1,line2 = twoLinesScrambledNoisy(2)
    makeOutput(line1,line2,summary)
    summary = "Two lines, one-quarter scrambling."
    line1,line2 = twoLinesScrambledNoisy(4)
    makeOutput(line1,line2,summary)
    summary = "Two lines, one-eighth scrambling."
    line1,line2 = twoLinesScrambledNoisy(8)
    makeOutput(line1,line2,summary)
    summary = "Two lines, one-twentieth scrambling."
    line1,line2 = twoLinesScrambledNoisy(20)
    makeOutput(line1,line2,summary)
    summary = "Two lines, one-fifieth scrambling."
    line1,line2 = twoLinesScrambledNoisy(50)
    makeOutput(line1,line2,summary)
    summary = "Two lines, no scrambling."
    line1,line2 = twoLinesScrambledNoisy(0)
    makeOutput(line1,line2,summary)

def test2LinesNoisy():
    import PecoraViz as PV
    np.set_printoptions(linewidth=125)
    def makeOutput(line1,line2,summary):
        ts = np.array([line1,line2]).transpose()
        d = continuityTestingFixedEps('Two lines',['x','y'],ts,[0,1],np.arange(0.2,1.1,0.2),np.array([0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5]),[[1,1]],numlags=3)
        print('---------------------------------------------------------')
        print(summary)
        print('---------------------------------------------------------')
        print("Epsilon as proportions of (changing) standard deviations: {0}.".format(d['epsprops']))
        print("Forward continuity confidence: ")
        print(d['forwardconf'])
        print("Inverse continuity confidence: ")
        print(d['inverseconf'])
        PV.plotOutput(d['forwardconf'],d['inverseconf'],d['epsprops'],d['tsprops'],len(line1),d['forwardtitle'],d['inversetitle'],logs = [0,0])
    summary = "Two lines, no noise."
    line1,line2 = twoLinesScrambledNoisy(0,0)
    makeOutput(line1,line2,summary)
    summary = "Two lines, 0.001*std noise."
    line1,line2 = twoLinesScrambledNoisy(0,0.001)
    makeOutput(line1,line2,summary)
    summary = "Two lines, 0.01*std noise."
    line1,line2 = twoLinesScrambledNoisy(0,0.01)
    makeOutput(line1,line2,summary)
    summary = "Two lines, 0.1*std noise."
    line1,line2 = twoLinesScrambledNoisy(0,0.1)
    makeOutput(line1,line2,summary)
    summary = "Two lines, 0.25*std noise."
    line1,line2 = twoLinesScrambledNoisy(0,0.25)
    makeOutput(line1,line2,summary)
    summary = "Two lines, 0.5*std noise."
    line1,line2 = twoLinesScrambledNoisy(0,0.5)
    makeOutput(line1,line2,summary)
    summary = "Two lines, 1.0*std noise."
    line1,line2 = twoLinesScrambledNoisy(0,1)
    makeOutput(line1,line2,summary)

if __name__ == '__main__':
    # remoteRun_DP(1200.0)
    # remoteRun_withnoise(1200.0)
    # remoteRun(1200.0)
    # ###################
    # compinds = [1,2]
    # finaltime = 1200.0
    # tsprops = np.arange(0.2,0.95,0.1)
    # lags = chooseLagsForSims_changedbeta(compinds,finaltime,tsprops,0.5)
    # ###################
    # localRun_zw()
    # testLagsAtDifferentLocationsAndLengths(2400.0)
    ###################
    test2LinesNoisy()
    test2LinesScrambled()
    # ###################
    # compinds = [1,2]
    # finaltime = 1200.0
    # tsprops = np.arange(0.2,0.95,0.1)
    # lags = chooseLagsForSimsDP(compinds,finaltime,tsprops)
    # ###################
    # localRun_DP(1200.0)
