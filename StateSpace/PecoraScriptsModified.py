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
    eqns = 'Double pendulum'
    names = ['x','y','z','w']
    return eqns,names,timeseries

def testDoublePendulum(masterts,mastereps,fname='',lags=None):
    eqns,names,ts = doublependulumTS(finaltime=1200.0)
    compind1=2
    compind2=3
    numlags = 5 #num dims
    forwardconf, inverseconf = PM.convergenceWithContinuityTest(ts[:,compind1],ts[:,compind2],numlags,lags,masterts=masterts,mastereps=mastereps)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compind1],names[compind2])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compind1],names[compind2])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','numlags','lags','ts','masterts','mastereps']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

def testDoublePendulumModified(masterts,mastereps,fname='',lags=None):
    eqns,names,ts = doublependulummodifiedTS(finaltime=1200.0)
    compind1=0
    compind2=3
    numlags = 5 #num dims
    forwardconf, inverseconf = PM.convergenceWithContinuityTest(ts[:,compind1],ts[:,compind2],numlags,lags,masterts=masterts,mastereps=mastereps)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compind1],names[compind2])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compind1],names[compind2])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','numlags','lags','ts','masterts','mastereps']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

if __name__ == '__main__':
    masterts = np.arange(0.5,0.85,0.1)
    mastereps=np.array([0.005,0.0075,0.01,0.0125,0.015,0.02,0.04])
    basedir = '/home/bcummins/'
    fname = 'DPMod_1200time_twofixedlags_xw.pickle'
    testDoublePendulumModified(masterts,mastereps,fname=basedir+fname,lags = [[100,5600]])
    # outdict = testDoublePendulum(masterts,mastereps)
    # eqns,names,ts = doublependulummodifiedTS(finaltime=1200.0)
    # compind1=2
    # for p in [0.1,0.2,0.4]:
    #     L = int(round(p*ts.shape[0]))
    #     for N in [20]:
    #         lags = PM.testLagsWithDifferentChunks(ts[:,compind1],L,N)
    #         print('z: {0} trials of length {1} in a length {2} timeseries.'.format(N,L,ts.shape[0]))
    #         print( ( np.min(lags), np.max(lags), np.mean(lags) ) )

    # compind1=3
    # for p in [0.1,0.2,0.4]:
    #     L = int(round(p*ts.shape[0]))
    #     for N in [20]:
    #         lags = PM.testLagsWithDifferentChunks(ts[:,compind1],L,N)
    #         print('w: {0} trials of length {1} in a length {2} timeseries.'.format(N,L,ts.shape[0]))
    #         print( ( np.min(lags), np.max(lags), np.mean(lags) ) )
