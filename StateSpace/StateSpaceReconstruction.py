def makeShadowManifold(timeseries, numlags):
    '''
    timeseries is a sequence of observations.
    numlags is a integer indicating the dimension of the shadow manifold
    to be constructed. It is required that numlags < len(timeseries).

    '''
    for t in range(len(timeseries)-numlags+1):
        yield timeseries[t,t+numlags]




