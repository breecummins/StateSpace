import numpy as np
import random
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops

def chooseLagsForSims(finaltime,tsprops=None,Tp=1000,rotated=1):
    if tsprops == None:
        tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200  
    if rotated:      
        eqns,names,ts = rotatedLorenzTS(finaltime)
    else:
        eqns,names,ts = lorenzTS(finaltime)
    Mlens = ( np.round( ts.shape[0]*tsprops ) ).astype(int)
    lags = SSR.chooseLags(ts,Mlens,Tp)
    return lags

def continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,numlags,fname=''):
    forwardconf, inverseconf, epsM1, epsM2, forwardprobs, inverseprobs = PM.convergenceWithContinuityTestFixedLagsFixedEps(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags[0][0],lags[0][1],tsprops=tsprops,epsprops=epsprops)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compinds[0]],names[compinds[1]])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compinds[0]],names[compinds[1]])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','numlags','lags','ts','tsprops','epsprops','epsM1','epsM2','forwardprobs','inverseprobs']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

def lorenzTS(finaltime=1200.0,dt=0.025):
    from LorenzEqns import solveLorenzVarChange
    timeseries = solveLorenzVarChange([1.0,0.5,-0.5],finaltime,dt)
    eqns = 'Lorenz with variable change'
    names = ['s','u','v']
    return eqns,names,timeseries

def rotatedLorenzTS(finaltime=1200.0,dt=0.025):
    from LorenzEqns import solveRotatedLorenz
    timeseries = solveRotatedLorenz([1.0,0.5,0.5],finaltime,dt)
    eqns = 'Rotated Lorenz'
    names = ['s','u','v']
    return eqns,names,timeseries

def runLorenz(finaltime=1200.0,remote=1,rotated=1):
    print('Beginning batch run for Lorenz equations....')
    if remote:
        basedir = '/home/bcummins/'
    else:
        basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/LorenzExample/'
    numlags = 3
    tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200
    epsprops=np.arange(0.1,1.0,0.1) 
    compind1 = [0,0,1]
    compind2 = [1,2,2]
    if rotated:
        eqns,names,ts = rotatedLorenzTS(finaltime)
        basefname = 'RotatedLorenz_1200time_samelags_biggereps_'
        lags= [[10,10],[10,10],[10,10]]
    else:
        eqns,names,ts = lorenzTS(finaltime)
        basefname = 'Lorenz_1200time_mixedlags_125_100_biggereps_'
        lags= [[125,125],[125,100],[125,100]]
    for k,c1 in enumerate(compind1):
        print('------------------------------------')
        print(names[c1] + ' and ' + names[compind2[k]])
        print('------------------------------------')
        compinds = [c1,compind2[k]]
        fname = basefname + names[c1] + names[compind2[k]] + '.pickle'
        continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,[lags[k]],numlags,fname=basedir+fname)

if __name__ == '__main__':
    runLorenz()
    # chooseLagsForSims(1200.0)
    # ################################
    # import StateSpaceReconstructionPlots as SSRPlots
    # eqs,ns,ts = rotatedLorenzTS(400.)
    # autocorr = SSR.getAutocorrelation(ts[:,0],200)
    # SSRPlots.plotAutocorrelation(autocorr,'x')
    # autocorr = SSR.getAutocorrelation(ts[:,1],200)
    # SSRPlots.plotAutocorrelation(autocorr,'y')
    # autocorr = SSR.getAutocorrelation(ts[:,2],200)
    # SSRPlots.plotAutocorrelation(autocorr,'z')