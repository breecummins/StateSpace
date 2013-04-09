def makeShadowManifold(timeseries, numlags, lagsize):
    '''
    timeseries is a sequence of observations.
    numlags is a integer indicating the dimension of the shadow manifold
    to be constructed. 
    lagsize is the number of time points in each lag.
    It is required that numlags*lagsize << len(timeseries) to make a good
    reconstruction. 

    '''
    for t in range(0,len(timeseries)-numlags*lagsize+1):
        yield timeseries[t:t+numlags*lagsize:lagsize]




