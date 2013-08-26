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

if __name__ == '__main__':
    masterts = np.arange(0.6,1.05,0.1)
    mastereps=np.array([0.05,0.075,0.1,0.15,0.2])
    # testLorenz(masterts,mastereps)
    testDoublePendulum(masterts,mastereps,fname='/home/bcummins/DP_1200time_biggestlag_zw.pickle',lags = [[3306,114],[2964,114],[2964,114],[5244,114],[6954,114]])
    # outdict = testDoublePendulum(masterts,mastereps)