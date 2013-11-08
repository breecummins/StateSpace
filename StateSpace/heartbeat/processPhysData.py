import numpy as np
import random
# import matplotlib.pyplot as plt
import StateSpace.StateSpaceReconstruction as SSR
# import StateSpace.StateSpaceReconstructionPlots as SSRPlots
import StateSpace.PecoraMethodModified as PM
import StateSpace.CaoNeighborRatio as CNR

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

def plotRates(ts,start=0,stop=None):
    if not stop:
        stop = len(hr)
    timeindex = range(start,stop)
    plt.plot(timeindex,ts[start:stop,0],'r')
    plt.hold('on')
    plt.plot(timeindex,ts[start:stop,1],'b')
    plt.legend(['Heart rate','Breathing rate'])
    plt.xlim([start,stop])
    plt.show()

def getLagsDims(ts,lags=None):
    N = ts.shape[0]
    randinds = random.sample(range(N-int(N/10.)),int(N/10.))
    lags, dims = CNR.getLagDim2(ts,randinds=randinds,lags=lags)
    return lags, dims

def getLagsDimsWrapper():
    ts = extractRates()
    print("--------------------------------------")
    print("Lags = 10")
    print("--------------------------------------")
    lags,dims = getLagsDims(ts, lags=[10,10])
    print("--------------------------------------")
    print("Lags = 20")
    print("--------------------------------------")
    lags,dims = getLagsDims(ts, lags=[20,20])
    print("--------------------------------------")
    print("Lags = 50")
    print("--------------------------------------")
    lags,dims = getLagsDims(ts, lags=[50,50])
    print("--------------------------------------")
    print("Lags = 100")
    print("--------------------------------------")
    lags,dims = getLagsDims(ts, lags=[100,100])

def autocorrViz(ts):
    autocorr0 = SSR.getAutocorrelation(ts[:,0],int(0.1*ts.shape[0]))
    autocorr1 = SSR.getAutocorrelation(ts[:,1],int(0.1*ts.shape[0]))
    SSRPlots.plotAutocorrelation(autocorr0,"heart rate autocorr")
    SSRPlots.plotAutocorrelation(autocorr1,"breathing rate autocorr")

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
    epsprops=np.array([0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) 
    tsprops = np.arange(0.3,1.05,0.1) 
    for l in lags:
        for d in dims:
            print('------------------------------------')
            print("Lag %d, dimension %d" % (l,d))
            print('------------------------------------')
            fname = 'HeartRate_lag%02d_dim%02d.pickle' % (l,d)
            continuityTestingFixedEps(eqns,names,ts,[0,1],tsprops,epsprops,[l,l],d,fname=basedir+fname) 



if __name__ == "__main__":
    runHeartRate([10,20,50],[3,4])
    # ts = extractRates()
    # SSRPlots.plotShadowManifold(ts[:,0], 3, 20, show=0,hold=0,style='b-',titlestr='Heart rate',scatter=False, color=None,smooth=1)
    # SSRPlots.plotShadowManifold(ts[:,1], 3, 20, show=1,hold=0,style='b-',titlestr='Breathing rate',scatter=False, color=None,smooth=1)
    # autocorrViz(ts)
    # plotRates(ts,90,720)
    # print("lags = {0}".format(lags))
    # print("dims = {0}".format(dims))

