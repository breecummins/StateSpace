import numpy as np
import random
import matplotlib.pyplot as plt
import StateSpace.StateSpaceReconstruction as SSR
import StateSpace.StateSpaceReconstructionPlots as SSRPlots
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

def autocorrViz(ts):
    autocorr0 = SSR.getAutocorrelation(ts[:,0],int(0.1*ts.shape[0]))
    autocorr1 = SSR.getAutocorrelation(ts[:,1],int(0.1*ts.shape[0]))
    SSRPlots.plotAutocorrelation(autocorr0,"heart rate autocorr")
    SSRPlots.plotAutocorrelation(autocorr1,"breathing rate autocorr")

def continuityTesting(hr,br,lags,dim):
    pass


if __name__ == "__main__":
    ts = extractRates()
    # SSRPlots.plotShadowManifold(ts[:,0], 3, 20, show=0,hold=0,style='b-',titlestr='Heart rate',scatter=False, color=None,smooth=1)
    # SSRPlots.plotShadowManifold(ts[:,1], 3, 20, show=1,hold=0,style='b-',titlestr='Breathing rate',scatter=False, color=None,smooth=1)
    # autocorrViz(ts)
    # plotRates(ts,90,720)
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
    # print("lags = {0}".format(lags))
    # print("dims = {0}".format(dims))


