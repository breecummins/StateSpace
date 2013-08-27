import numpy as np

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

def findAllZeros(arr):
    '''
    arr must be a list or 1D numpy array

    '''
    signs = np.sign(arr[1:]) + np.sign(arr[:-1])
    ind = np.nonzero(np.abs(signs) < 2)[0]
    zeros = []
    for i in ind:        
        # record whichever of the two indices flanking zero has the smallest value
        if np.abs(arr[i]) < np.abs(arr[i+1]):
            zeros.append(i)
        else:
            zeros.append(i+1)
    return zeros

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
    return the first zero crossing in the first 30%
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

def evaluateSimilarity(int1,int2,N):
    '''
    Given two integers, decide whether they are similar or not. 
    N is a normalization factor.

    '''
    if float(int1)/int2 < 4./3 and float(int1)/int2 > 3./4:
        return [int(np.mean([int1,int2]))]
    else:
        if int1 < int2:
            multiplier = int(float(int2)/int1)
            return [int1,int(int1*multiplier)]
        else:
            multiplier = int(float(int1)/int2)
            return [int(int2*multiplier),int2]

def chooseLagSize(ts1,ts2):
    '''
    Choose lag sizes for the two time series ts1 and ts2, which
    are both 1D arrays of the same length, by calculating the 
    first zeros of the autocorrelations. Decide whether the lags 
    are close enough to be averaged, or must be considered 
    different values. 

    '''
    lag1 = lagsizeFromFirstZeroOfAutocorrelation(ts1)
    lag2 = lagsizeFromFirstZeroOfAutocorrelation(ts2)
    sim = evaluateSimilarity(lag1,lag2,len(ts1))
    if len(sim) == 1:
        newlags = sim*2 # list of length 2 with newlags[0] = newlags[1] = sim[0]
    else:
        newlags = sim
    print('Original lags: {0}'.format([lag1,lag2]))
    print('Modified lags: {0}'.format(newlags))
    accept = raw_input("Do you accept the modified lags? (y or n) ") 
    if accept == 'n':
        newlags = input("Enter a new length 2 list with the desired lags, such as [105,6900] or [23,23]. ")
    return newlags




