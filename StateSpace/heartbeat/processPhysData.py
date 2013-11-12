import numpy as np
import random, os
import StateSpace.PecoraMethodModified as PM
import StateSpace.CaoNeighborRatio as CNR
import StateSpace.fileops as fileops

def extractRates(fname='physdata_13081200_NewSummary.csv'):
    f = open(fname,'r')
    f.readline()
    data = f.readlines()
    f.close()
    heartrate = []
    breathingrate = []
    for d in data: 
        words = d.split(',')
        heartrate.append(float(words[3]))
        breathingrate.append(float(words[5]))
    return np.vstack([heartrate,breathingrate]).transpose()

def getLagsDims(ts,lags=None):
    N = ts.shape[0]
    randinds = random.sample(range(N-int(N/10.)),int(N/10.))
    lags, dims = CNR.getLagDim2(ts,randinds=randinds,lags=lags)
    return lags, dims

def getLagsDimsWrapper(listoflags,start=0,stop=None):
    if not stop:
        stop = ts.shape[0]
    ts = extractRates()
    ts = ts[start:stop,:]
    for l in listoflags:
        print("--------------------------------------")
        print("Lags = {0}".format(l))
        print("--------------------------------------")
        lags,dims = getLagsDims(ts, lags=[l,l])

def continuityTestingFixedEps(eqns,names,ts,compinds,tsprops,epsprops,lags,numlags,fname=''):
    forwardconf, inverseconf, epsM1, epsM2, forwardprobs, inverseprobs = PM.convergenceWithContinuityTestFixedLagsFixedEps(ts[:,compinds[0]],ts[:,compinds[1]],numlags,lags[0],lags[1],tsprops=tsprops,epsprops=epsprops)
    forwardtitle = eqns + r', M{0} $\to$ M{1}'.format(names[compinds[0]],names[compinds[1]])
    inversetitle = eqns + r', M{1} $\to$ M{0}'.format(names[compinds[0]],names[compinds[1]])
    outdict = dict([(x,locals()[x]) for x in ['forwardconf','inverseconf','forwardtitle','inversetitle','numlags','lags','ts','tsprops','epsprops','epsM1','epsM2','forwardprobs','inverseprobs']])
    if fname:
        fileops.dumpPickle(outdict,fname)
    else:
        return outdict

def runHeartRate(lags,dims,remote=1):
    if remote:
        basedir = '/home/bcummins/'
    else:
        basedir=os.path.join(os.path.expanduser("~"),'SimulationResults/TimeSeries/HeartRateData/')
    eqns = 'Heart rate time series data'
    names = ['h','b']
    print('Beginning batch run for heart rate equations....')
    ts = extractRates()
    ts = ts[90:720,:]
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) 
    tsprops = np.arange(0.3,1.05,0.1) 
    for l in lags:
        for d in dims:
            print('------------------------------------')
            print("Lag %d, dimension %d" % (l,d))
            print('------------------------------------')
            fname = 'HeartRateTruncated_lag%02d_dim%02d.pickle' % (l,d)
            continuityTestingFixedEps(eqns,names,ts,[0,1],tsprops,epsprops,[l,l],d,fname=basedir+fname) 



if __name__ == "__main__":
    runHeartRate([100],range(3,9),remote=0)
    # getLagsDimsWrapper([1,5,10,20],90,720)


