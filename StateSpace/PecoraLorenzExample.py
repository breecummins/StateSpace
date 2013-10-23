import numpy as np
import random
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops

def chooseLagsForSims(finaltime,tsprops=None,Tp=1000,rotated=1):
    if tsprops == None:
        tsprops = np.arange(0.3,1.05,0.1) # for finaltime = 1200  
    if rotated:      
        eqns,names,ts = rotatedLorenzTS(finaltime)
    else:
        eqns,names,ts = lorenzTS(finaltime)
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

def lorenzTS(finaltime=1200.0,dt=0.025):
    from LorenzEqns import solveLorenz
    timeseries = solveLorenz([1.0,0.5,0.5],finaltime,dt)
    eqns = 'Lorenz eqns'
    names = ['x','y','z']
    return eqns,names,timeseries

def rotatedLorenzTS(finaltime=1200.0,dt=0.025):
    from LorenzEqns import solveRotatedLorenz
    timeseries = solveRotatedLorenz([1.0,0.5,0.5],finaltime,dt)
    eqns = 'Rotated Lorenz'
    names = ['x','y','z']
    return eqns,names,timeseries

def runLorenz(finaltime=1200.0,remote=1,rotated=1):
    print('Beginning batch run for Lorenz equations....')
    if remote:
        basedir = '/home/bcummins/'
    else:
        basedir=os.path.join(os.path.expanduser("~"),'SimulationResults/TimeSeries/PecoraMethod/LorenzExample/')
    numlags = 3
    tsprops = np.arange(0.3,1.05,0.1) # for finaltime = 1200
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) 
    def runme(eqns,names,ts,basefname,lags):
        for c1 in range(2):
            for c2 in range(c1+1,3):
                print('------------------------------------')
                print(names[c1] + ' and ' + names[c2])
                print('------------------------------------')
                fname = basefname + names[c1] + names[c2] + '.pickle'
                continuityTestingFixedEps(eqns,names,ts,[c1,c2],tsprops,epsprops,[lags[c1],lags[c2]],numlags,fname=basedir+fname)
    if rotated:
        eqns,names,ts = rotatedLorenzTS(finaltime)
        basefname = 'RotatedLorenz_1200time_samelags_'
        lags= [10,10,10]
        runme(eqns,names,ts,basefname,lags)
    else:
        eqns,names,ts = lorenzTS(finaltime)
        lags= [20,20,7]
        basefname = 'Lorenz_1200time_mixedlags_xylag020_'
        runme(eqns,names,ts,basefname,lags)
        lags= [40,40,7]
        basefname = 'Lorenz_1200time_mixedlags_xylag040_'
        runme(eqns,names,ts,basefname,lags)
        lags= [160,160,7]
        basefname = 'Lorenz_1200time_mixedlags_xylag160_'
        runme(eqns,names,ts,basefname,lags)

if __name__ == '__main__':
    runLorenz(rotated=0)
    # chooseLagsForSims(1200.0,rotated=0)
    # ################################
    # import StateSpaceReconstructionPlots as SSRPlots
    # eqs,ns,ts = lorenzTS(1200.)
    # autocorr = SSR.getAutocorrelation(ts[:,0],200)
    # SSRPlots.plotAutocorrelation(autocorr,'x')
    # autocorr = SSR.getAutocorrelation(ts[:,1],200)
    # SSRPlots.plotAutocorrelation(autocorr,'y')
    # autocorr = SSR.getAutocorrelation(ts[:,2],200)
    # SSRPlots.plotAutocorrelation(autocorr,'z')