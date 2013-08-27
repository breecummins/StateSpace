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

def testDoublePendulum(compinds,tsprops,epsprops,fname='',lags=None):
    eqns,names,ts = doublependulumTS(finaltime=1200.0)
    numlags = 5 #num dims
    if len(lags) == 1:
        forwardconf, inverseconf = PM.convergenceWithContinuityTestFixedLags(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags[0][0],lags[0][1],tsprops=tsprops,epsprops=epsprops)
    else:
        forwardconf, inverseconf = PM.convergenceWithContinuityTestMultipleLags(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags,tsprops=tsprops,epsprops=epsprops)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compinds[0]],names[compinds[1]])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compinds[0]],names[compinds[1]])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','numlags','lags','ts','tsprops','epsprops']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

def testDoublePendulumModified(compinds,tsprops,epsprops,fname='',lags=None):
    eqns,names,ts = doublependulummodifiedTS(finaltime=1200.0)
    numlags = 5 #num dims
    if len(lags) == 1:
        forwardconf, inverseconf = PM.convergenceWithContinuityTestFixedLags(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags[0][0],lags[0][1],tsprops=tsprops,epsprops=epsprops)
    else:
        forwardconf, inverseconf = PM.convergenceWithContinuityTestMultipleLags(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags,tsprops=tsprops,epsprops=epsprops)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compinds[0]],names[compinds[1]])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compinds[0]],names[compinds[1]])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','numlags','lags','ts','tsprops','epsprops']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

if __name__ == '__main__':
    # tsprops = np.arange(0.5,0.85,0.1)
    # epsprops=np.array([0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.0075]) #for x and y
    # compinds = [0,1]
    # lags = [[100,100]]
    # basedir = '/home/bcummins/'
    # fname = 'DPMod_1200time_samefixedlags_xy.pickle'
    # testDoublePendulumModified(compinds,tsprops,epsprops,fname=basedir+fname,lags = lags)
    # outdict = testDoublePendulum(tsprops,epsprops)
    
    eqns,names,ts = doublependulummodifiedTS(finaltime=1200.0)
    compind1=2
    for p in [0.1,0.2,0.4,0.6,0.8]:
        L = int(round(p*ts.shape[0]))
        for N in [20]:
            lags = PM.testLagsWithDifferentChunks(ts[:,compind1],L,N)
            print('z: {0} trials of length {1} in a length {2} timeseries.'.format(N,L,ts.shape[0]))
            print( ( np.min(lags), np.max(lags), np.mean(lags) ) )

    compind1=3
    for p in [0.1,0.2,0.4,0.6,0.8]:
        L = int(round(p*ts.shape[0]))
        for N in [20]:
            lags = PM.testLagsWithDifferentChunks(ts[:,compind1],L,N)
            print('w: {0} trials of length {1} in a length {2} timeseries.'.format(N,L,ts.shape[0]))
            print( ( np.min(lags), np.max(lags), np.mean(lags) ) )