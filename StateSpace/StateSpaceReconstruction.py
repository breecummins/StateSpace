import numpy as np
import random

def makeShadowManifoldSmooth(timeseries, numlags, lagsize):
    '''
    timeseries is a sequence of observations.
    numlags is a integer indicating the dimension of the shadow manifold
    to be constructed. 
    lagsize is the number of time points in each lag.
    It is required that numlags*lagsize << len(timeseries) to make a good
    reconstruction. 

    Reconstruct all time points.

    '''
    Mx = np.zeros((len(timeseries)-(numlags-1)*lagsize,numlags))
    for t in range(0,len(timeseries)-(numlags-1)*lagsize):
        inds = np.arange(t+(numlags-1)*lagsize,t-1,-lagsize)
        Mx[t,:] = timeseries[inds]
    return Mx

def makeShadowManifoldSkip(timeseries, numlags, lagsize):
    '''
    timeseries is a sequence of observations.
    numlags is a integer indicating the dimension of the shadow manifold
    to be constructed. 
    lagsize is the number of time points in each lag.
    It is required that numlags*lagsize << len(timeseries) to make a good
    reconstruction. 

    Reconstruct only time points that are multiples of lagsize.

    '''
    # excise non-multiples of lagsize
    timeseries = timeseries[::lagsize]
    lagsize = 1
    return makeShadowManifoldSmooth(timeseries, numlags, lagsize)

def makeShadowManifold(timeseries, numlags, lagsize, smooth=1):
    if smooth:
        return makeShadowManifoldSmooth(timeseries, numlags, lagsize)
    else:
        return makeShadowManifoldSkip(timeseries, numlags, lagsize)

def findFirstZero(arr):
    '''
    arr must be a list or 1D numpy array

    '''
    signs = np.sign(arr[1:]) + np.sign(arr[:-1])
    ind = np.argmax(np.abs(signs) < 2)
    if ind == 0 and np.abs(signs[0]) == 2: 
        # False positive: if there is no zero crossing, argmax will return 
        # the first index, so I have to manually check that signs[0] < 2.
        return None 
    elif np.abs(arr[ind]) < np.abs(arr[ind+1]):
        # return whichever of the two indices flanking zero has the smallest value
        return ind
    else:
        return ind+1

def getAutocorrelation(ts,T):
    N = len(ts)
    mu = np.mean(ts)
    s2 = np.var(ts,ddof=1) #unbiased estimator of variance
    autocc = []
    for k in range(1,T+1):
        autocc.append( ((ts[:N-k] - mu)*(ts[k:] - mu)).sum() / ( s2 *(N-k) ) )
    return autocc

def lagsizeFromFirstZeroOfAutocorrelation(ts,T=None):
    '''
    Calculate autocorrelation of a 1D timeseries and 
    return the first zero crossing in the first 40%
    of the time series. An error is thrown if there
    is no such zero.

    '''
    if T == None:
        autocorr = getAutocorrelation(ts,int(len(ts)*0.1))
        fz = findFirstZero(autocorr)
        if fz == None:
            autocorr = getAutocorrelation(ts,int(len(ts)*0.4))
            fz = findFirstZero(autocorr)
            if fz == None:
                raise ValueError('No zero in the autocorrelation for the first 40% of the time series.')
    else:
        autocorr = getAutocorrelation(ts,T)
        fz = findFirstZero(autocorr)
        if fz == None:
            raise ValueError('No zero in the autocorrelation for the first {0}% of the time series.'.format(int(T*100/len(ts))))
    return fz

def chooseLags(ts,Mlens,Tp=None):
    if len(ts.shape) == 1:
        N = 1
    elif len(ts.shape) == 2:
        N = ts.shape[1]
    lags = [[] for _ in range(len(Mlens))]
    for k,L in enumerate(Mlens):
        for j in range(N):
            lags[k].append(lagsizeFromFirstZeroOfAutocorrelation(ts[:L,j],Tp))
        print("Time series length: {0}, lags = {1}.".format(L,lags[k]))
    return lags

def testLagsWithDifferentChunks(ts,L,N,T=None):
    '''
    Get lag size from timeseries ts for chunks of 
    length L starting at N different locations.

    '''
    lags = []
    startlocs = random.sample(range(len(ts)-L),N)
    for s in startlocs:
        lags.append(lagsizeFromFirstZeroOfAutocorrelation(ts[s:s+L],T))
    return lags



