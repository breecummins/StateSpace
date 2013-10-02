import numpy as np
import random
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops

def chooseLagsForSims(finaltime,tsprops=None,Tp=150):
    if tsprops == None:
        tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200        
    eqns,names,ts = diamondTS(finaltime)
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

def diamondTS(finaltime=1200.0,dt=0.025):
    from Rossler import solveDiamond
    timeseries = solveDiamond([1.0,2.0,3.0,2.0,5.0,4.0,3.0,3.0],finaltime,dt)
    eqns = 'Diamond'
    names = ['x','y','z','w','s','u','v','p']
    return eqns,names,timeseries

def runDiamond(epsprops,compinds,fname,lags,basedir='/Users/bree/SimulationResults/TimeSeries/PecoraMethod/Diamondpaperexample/',finaltime=1200.0):
    eqns,names,ts = diamondTS(finaltime)
    tsprops = np.arange(0.3,0.95,0.1) # for finaltime = 1200
    continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,fname=basedir+fname) 

def remoteRun_Diamond(finaltime=1200.0):
    print('Beginning batch run for diamond equations....')
    basedir = '/home/bcummins/'
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) 
    names = ['x','y','z','w','s','u','v','p']
    compind1 = [0,0,0,0,1,1,1,1,4,4,4,5,5,7,7,7]
    compind2 = [4,5,6,7,4,5,6,7,5,6,7,6,7,6,3,2]
    basefname = 'Diamond_1200time_mixedlags_'
    lags= [[100,60],[100,70],[100,30],[100,120],[100,60],[100,70],[100,30],[100,120],[60,70],[60,30],[60,120],[70,30],[70,120],[120,30],[120,100],[120,115]]
    for k,c1 in enumerate(compind1):
        print('------------------------------------')
        print(names[c1] + ' and ' + names[compind2[k]])
        print('------------------------------------')
        compinds = [c1,compind2[k]]
        fname = basefname + names[c1] + names[compind2[k]] + '.pickle'
        runDiamond(epsprops,compinds,fname,[lags[k]],basedir=basedir,finaltime=finaltime)

if __name__ == '__main__':
    # remoteRun_Diamond(1200.0)
    chooseLagsForSims(1200.0)
    # #below, choose lags from autocorrelation
    # import PecoraViz as PV
    # eqns,names,ts = diamondTS(1200.0)
    # T = 150
    # autocorr = SSR.getAutocorrelation(ts[:,4],T)
    # PV.plotAutocorrelation(autocorr,'s')
    # autocorr = SSR.getAutocorrelation(ts[:,7],T)
    # PV.plotAutocorrelation(autocorr,'p')
    # autocorr = SSR.getAutocorrelation(ts[:,6],T)
    # PV.plotAutocorrelation(autocorr,'v')
    # autocorr = SSR.getAutocorrelation(ts[:,5],T)
    # PV.plotAutocorrelation(autocorr,'u')
    # autocorr = SSR.getAutocorrelation(ts[:,0],T)
    # PV.plotAutocorrelation(autocorr,'x')
    # autocorr = SSR.getAutocorrelation(ts[:,3],T)
    # PV.plotAutocorrelation(autocorr,'w')