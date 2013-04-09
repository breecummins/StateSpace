import StateSpaceReconstruction as SSR
import numpy as np

def findClosest(poi,pts,N):
    '''
    Find the closest N points in pts (numpy array) to poi.

    '''
    dists = np.sqrt(((pts - poi)**2).sum(1))
    out = []
    for _ in range(N):
        i = dists.argmin()
        out.append((dists[i],i))
        dists[i] = np.Inf
    return zip(*out)

def makeExpWeights(dists):
    u = np.exp( -dists/dists[0])
    return u / u.sum()

def crossMap(ts1,ts2,numlags,lagsize,wgtfunc):
    '''
    Estimate timeseries 1 (ts1) from timeseries 2 and vice versa using a 
    cross-mapping technique between shadow manifolds.
    Construct a shadow manifold M from one time series.
    Find the nearest points to each point in M.
    Construct weights based on their distances to the point of
    interest. 
    Use the time indices of these points and the associated weights to 
    make a weighted-sum estimate of the contemporaneous points in the 
    other time series.

    '''
    M1 = SSR.makeShadowManifold(timeseries1,numlags,lagsize)
    M2 = SSR.makeShadowManifold(timeseries2,numlags,lagsize)
    def estSeries(M,ts):
        est=np.zeros(ts.shape)
        for k in range(M.shape[0]):
            poi = M[k,:]
            dists,inds = findClosest(poi,M,numlags+1)
            w = wgtfunc(dists)
            est[k] = w*ts[list(inds)]
        return est
    est1 = estSeries(M2,M1[:,-1])
    est2 = estSeries(M1,M2[:,-1])
    return est1, est2

def corrCoeff():
    pass

def testCausality(ts1,ts2,numlags,lagsize,listoflens,numiters,wgtfunc=makeExpWeights):
    '''
    Check for convergence (Sugihara) to infer causality between ts1 and ts2.
    numlags is the dimension of the embedding space for the reconstruction.
    Use time lags of size lagsize * dt to construct shadow manifolds. lagsize
    is an integer representing the index of the time lag.
    listoflens contains the lengths to use to show convergence 
    Example: range(100,10000,100)
    Each length will be run numiters times from different random starting 
    locations in the time series.
    The estimated time series will be constructed using the weighting function 
    handle given by wgtfunc.

    '''
    listoflens.sort()
    cap = min(len(ts1),len(ts2))
    lol = [l if l < cap for l in listoflens]
    for l in lol:
        # choose numiters starting indices
        # for each starting index, choose the appropriate chunks of ts1 and ts2
            # call crossMap with the chunks
            # calculate correlation coefficient of estimated time series with real 
        # record average over correlation coefficients
        
        




     