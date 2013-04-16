import numpy as np

def makeShadowManifold(timeseries, numlags, lagsize):
    '''
    timeseries is a sequence of observations.
    numlags is a integer indicating the dimension of the shadow manifold
    to be constructed. 
    lagsize is the number of time points in each lag.
    It is required that numlags*lagsize << len(timeseries) to make a good
    reconstruction. 

    '''
    Mx = np.zeros((len(timeseries)-(numlags-1)*lagsize+1,numlags))
    for t in range(0,len(timeseries)-(numlags-1)*lagsize):
        inds = np.arange(t+(numlags-1)*lagsize,t-1,-lagsize)
        Mx[t,:] = timeseries[inds]
    return Mx




