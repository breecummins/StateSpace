import numpy as np
import PecoraMethod as PM
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

def testLorenz(masterts,mastereps,fname=''):
    eqns,names,ts = lorenzTS()
    compind1=0
    compind2=1
    numlags = 3 #num dims
    lagsize = 8
    # lagsize = SSR.numlagsFromFirstZeroOfAutocorrelation(ts[:,compind1])
    # print(lagsize)
    Mx = SSR.makeShadowManifold(ts[:,compind1], numlags, lagsize)
    My = SSR.makeShadowManifold(ts[:,compind2], numlags, lagsize)
    N = int(0.1*Mx.shape[0])
    forwardconf, inverseconf = PM.convergenceWithContinuityTest(Mx,My,N,masterts,mastereps)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compind1],names[compind2])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compind1],names[compind2])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','N','numlags','lagsize','ts','masterts','mastereps']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

def testDoublePendulum(masterts,mastereps,fname=''):
    eqns,names,ts = doublependulumTS()
    compind1=2
    compind2=3
    numlags = 5 #num dims
    lagsize = 28
    # lagsize = SSR.numlagsFromFirstZeroOfAutocorrelation(ts[:,compind1])
    # print(lagsize)
    M1 = SSR.makeShadowManifold(ts[:,compind1], numlags, lagsize)
    M2 = SSR.makeShadowManifold(ts[:,compind2], numlags, lagsize)
    N = int(0.1*M1.shape[0])
    forwardconf, inverseconf = PM.convergenceWithContinuityTest(M1,M2,N,masterts,mastereps)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compind1],names[compind2])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compind1],names[compind2])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','N','numlags','lagsize','ts','masterts','mastereps']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

if __name__ == '__main__':
    masterts = np.arange(0.2,1.1,0.2)
    mastereps=np.array([0.01,0.05,0.1,0.2])
    # testLorenz(masterts,mastereps)
    # testDoublePendulum(masterts,mastereps,fname='/home/bcummins/DP600ptszw.pickle')
    outdict = testDoublePendulum(masterts,mastereps)