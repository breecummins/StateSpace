import numpy as np

def findClosest(poi,pts,N):
    '''
    Find the closest N points in pts (numpy array) to poi, 
    where poi is a member of pts.

    '''
    dists = np.sqrt(((pts - poi)**2).sum(1))
    out = []
    # The first point recovered will be poi itself.
    # We'll need to throw it out.
    for _ in range(N+1):
        i = dists.argmin()
        out.append((dists[i],i))
        dists[i] = np.Inf
    return zip(*out[1:])

def makeExpWeights(dists):
    u = np.exp( -dists/dists[0])
    return u / u.sum()

