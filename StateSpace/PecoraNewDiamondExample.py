import numpy as np
import random, os
import PecoraMethodModified as PM
import StateSpaceReconstruction as SSR
import fileops

def chooseLagsForSims(finaltime,tsprops=None,Tp=400,unrotated=1):
    if tsprops == None:
        tsprops = np.arange(0.3,1.05,0.1) # for finaltime = 1200  
    if unrotated:
        eqns,names,ts = unrotatedDiamondTS(finaltime)
    else:
        eqns,names,ts = newDiamondTS(finaltime)
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

def newDiamondTS(finaltime=1200.0,dt=0.025):
    from Rossler import solveDiamondRotatedInteralMult
    timeseries = solveDiamondRotatedInteralMult([1.0,2.0,3.0,2.0,5.0,4.0,3.0,0.75],finaltime,dt)
    eqns = 'Diamond with internal mult by y'
    names = ['x','y','z','w','s','u','v','p']
    return eqns,names,timeseries

def unrotatedDiamondTS(finaltime=1200.0,dt=0.025):
    from Rossler import solveDiamond
    timeseries = solveDiamond([1.0,2.0,3.0,2.0,5.0,4.0,3.0,0.75],finaltime,dt)
    eqns = 'Diamond, unrotated Rossler'
    names = ['x','y','z','w','s','u','v','p']
    return eqns,names,timeseries

def runDiamond(finaltime=1200.0,remote=1,unrotated=1):
    #FIXME: Rewrite like PecoraDoublePendulumDiamondExample.py with a double for loop
    print('Beginning batch run for diamond equations....')
    if remote:
        basedir = '/home/bcummins/'
    else:
        basedir=os.path.join(os.path.expanduser("~"),'SimulationResults/TimeSeries/PecoraMethod/FinalPaperExamples/')
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) 
    compind1 = [0,0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,4,5,5,7,7,7]
    compind2 = [4,5,6,7,4,5,6,7,4,5,6,4,5,6,5,6,7,6,7,6,3,2]
    tsprops = np.arange(0.3,1.05,0.1) 
    numlags = 8
    if unrotated:
        eqns,names,ts = unrotatedDiamondTS(finaltime)
        basefname = 'DiamondUnrotated_1200time_mixedlags_' 
        lags= [[100,60],[100,60],[100,45],[100,100],[100,60],[100,60],[100,45],[100,100],[115,60],[115,60],[115,45],[100,60],[100,60],[100,45],[60,60],[60,45],[60,100],[60,45],[60,100],[100,45],[100,100],[100,115]]
    else:
        eqns,names,ts = newDiamondTS(finaltime)
        basefname = 'DiamondInternalMult_1200time_mixedlags_' 
        lags= [[100,60],[100,60],[100,60],[100,185],[100,60],[100,60],[100,60],[100,185],[115,60],[115,60],[115,60],[100,60],[100,60],[100,60],[60,60],[60,60],[60,185],[60,60],[60,185],[185,60],[185,100],[185,115]]
    for k,c1 in enumerate(compind1):
        print('------------------------------------')
        print(names[c1] + ' and ' + names[compind2[k]])
        print('------------------------------------')
        compinds = [c1,compind2[k]]
        fname = basefname + names[c1] + names[compind2[k]] + '.pickle'
        continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags[k],numlags,fname=basedir+fname) 

if __name__ == '__main__':
    runDiamond()
    # chooseLagsForSims(1200.0)
    # #below, choose lags from autocorrelation
    # import StateSpaceReconstructionPlots as SSRPlots
    # eqns,names,ts = newDiamondTS(1200.0)
    # T = 400
    # autocorr = SSR.getAutocorrelation(ts[:,7],T)
    # SSRPlots.plotAutocorrelation(autocorr,'p')
