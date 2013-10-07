import numpy as np
import random
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops

def chooseLagsForSims(finaltime,tsprops=None,Tp=400):
    if tsprops == None:
        tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200  
    eqns,names,ts = diamondTS(finaltime)
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

def diamondTS(finaltime=1200.0,dt=0.025):
    from DoublePendulumDiamond import solveDiamond
    timeseries = solveDiamond([1.0,2.0,1.0,2.0,2.0,2.0,1.0,2.0],finaltime,dt)
    eqns = 'Double Pendulum Diamond'
    names = ['x','y','z','w','u','v','p','q']
    return eqns,names,timeseries

def runDiamond(finaltime=1200.0,remote=1):
    print('Beginning batch run for double pendulum diamond equations....')
    if remote:
        basedir = '/home/bcummins/'
    else:
        basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/DoublePendulumDiamondExample/'
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) 
    tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200
    numlags = 8
    eqns,names,ts = diamondTS(finaltime)
    basefname = 'DPDiamond_1200time_mixedlags_'
    lags = [20,20,70,70,90,60,90,90]
    for c1 in range(7):
        for c2 in range(c1+1,8):
            print('------------------------------------')
            print(names[c1] + ' and ' + names[c2])
            print('------------------------------------')
            fname = basefname + names[c1] + names[c2] + '.pickle'
            continuityTestingFixedEps(eqns,names,ts,[c1,c2],tsprops,epsprops,[lags[c1],lags[c2]],numlags,fname=basedir+fname) 

if __name__ == '__main__':
    runDiamond()
    # chooseLagsForSims(1200.0)
    # #below, choose lags from autocorrelation
    # import StateSpaceReconstructionPlots as SSRPlots
    # eqns,names,ts = diamondVarChangeTS(1200.0)
    # T = 400
    # autocorr = SSR.getAutocorrelation(ts[:,7],T)
    # SSRPlots.plotAutocorrelation(autocorr,'s')
