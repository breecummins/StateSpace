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
    return the first zero crossing. Check all lags from
    1 to T. Default is T = len(ts)/10. 

    '''
    if T == None:
        T = int(len(ts)/10.)
    autocc = getAutocorrelation(ts,T)
    return findFirstZero(autocc)

def getAllLags(ts):
    '''
    Find all the first zeros of the autocorrelation functions
    of the columns of ts, an mxn numpy array.

    '''
    lags = []
    for j in range(ts.shape[1]):
        lag = lagsizeFromFirstZeroOfAutocorrelation(ts[:,j],int(0.10*ts.shape[0]))
        if lag == None:
           lag = lagsizeFromFirstZeroOfAutocorrelation(ts[:,j],int(0.3*ts.shape[0]))
           if lag == None:
                raise ValueError('No zero in the first 0.30 of the time series for variable {0}'.format(j))
        lags.append(lag)
    return lags

def evaluateSimilarity(arr):
    '''
    Given a 1D integer array, quantize the integers into 1 or 2 groups that are
    divided by the mean.

    '''
    mu = np.mean(arr)
    if np.all( np.abs(arr - mu) < np.std(arr) ): 
        return [range(len(arr))]
    else:
        newarr = [np.array([j for j,a in enumerate(arr) if a < mu]), np.array([j for j,a in enumerate(arr) if a >= mu])]
        return newarr

def chooseLagSize(ts):
    '''
    Choose lag sizes for the columns of ts (mxn numpy array).
    Decide whether the first zeros of the autocorrelations of
    the columns of ts are similar or different. Ideally they 
    are all the same, but in reality this won't always happen.

    '''
    lags = getAllLags(ts)
    sim = evaluateSimilarity(lags)
    if len(sim) == 1:
        newlags = int(np.mean(lags))*np.ones(len(lags)).astype(int)
    else:
        newlags = np.zeros(len(lags)).astype(int)
        for s in sim:
            newlags[s] = int(np.mean(np.array(lags)[s]))
        u = sorted(list(set(list(newlags))))
        multiplier = int(float(u[1])/u[0])
        newlags[newlags==u[1]] = multiplier*u[0]
    print('Original calculated lags: {0}'.format(lags))
    print('New binned and averaged lags: {0}'.format(newlags))
    accept = raw_input("Do you accept the new lags? (y or n) ") 
    if accept == 'n':
        inlags = input("Enter a new numpy array of the same size, such as np.array([2,2,90,47]) for a 4 variable problem. ")
        if len(inlags) == len(newlags):
            newlags = inlags
        else:
            newlags = input("Try again. ")
    return newlags



