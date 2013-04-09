import StateSpaceReconstruction as SSR
import numpy as np

def findClosest(poi,pts,N):
    '''
    Find the closest N points in pts (numpy array) to poi.
    This method is faster than either of the other two below.

    '''
    dists = np.sqrt(((pts - poi)**2).sum(1))
    out = []
    for _ in range(N):
        i = dists.argmin()
        out.append((dists[i],i))
        dists[i] = np.Inf
    return zip(*out)

# def findClosest2(poi,pts,N):
#     '''
#     Find the closest N points in pts (numpy array) to poi.

#     '''
#     dists = np.sqrt(((pts - poi)**2).sum(1))
#     inds = dists.argsort()[:N]
#     return [(dists[i],i) for i in inds]

# def findClosest3(poi,pts,N):
#     '''
#     Find the closest N points in pts (numpy array) to poi.

#     '''
#     dists = np.sqrt(((pts - poi)**2).sum(1))
#     inds = bn.argpartsort(dists,N)[:N]
#     d = [(dists[i],i) for i in inds]
#     d.sort()
#     return d

def makeExpWeights(dists):
    u = np.exp( -dists/dists[0])
    return u / u.sum()

def estSeries(M,ts,npts,wgtfunc=makeExpWeights):
    est=np.zeros(ts.shape)
    for k in range(M.shape[0]):
        poi = M[k,:]
        dists,inds = findClosest(poi,M,npts)
        w = wgtfunc(dists)
        est[k] = w*ts[list(inds)]
    return est

def crossMap(ts1,ts2,numlags,lagsize,wgtfunc=makeExpWeights):
    '''
    Estimate timeseries 1 (ts1) from timeseries 2 and vice versa using a 
    cross-mapping technique.
    numlags is the dimension of the embedding space for the reconstruction.
    Use time lags of size lagsize * dt to construct shadow manifolds. lagsize
    is an integer representing the index of the time lag.

    '''
    # construct the shadow manifolds
    M1 = SSR.makeShadowManifold(timeseries1,numlags,lagsize)
    M2 = SSR.makeShadowManifold(timeseries2,numlags,lagsize)
    # reconstruct 1 from 2 (testing to see if 1 -> 2)
    est1 = estSeries(M2,M1[:,-1],numlags+1,wgtfunc)
    # reconstruct 2 from 1 (testing to see if 2 -> 1)
    est2 = estSeries(M1,M2[:,-1],numlags+1,wgtfunc)
    return est1, est2

def corrCoeff():
    pass

def testCausality(ts1,ts2,numlags,lagsize,listoflens,numiters,wgtfunc=makeExpWeights):
    '''
    Check for convergence (Sugihara) to infer causality between ts1 and ts2.

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
        
        




     