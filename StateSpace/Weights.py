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

def typicalVolume(M1):
    '''
    Find the "typical volume" of points in M1.
    M1 is embedded in R^d, where d = M1.shape[1].
    Find the d+1 closest points in M1 to each point in 
    M1 and take the average distance along each axis.
    The product of these distances is the estimated
    volume of each point. Then take the average
    to get a typical volume over all of the points.

    '''
    d = M1.shape[1]
    dx = np.zeros(M1.shape)
    for k in range(M1.shape[0]):
        poi = M1[k,:]
        dists,inds=findClosest(poi,M1,d+1)
        dx[k,:] = np.mean(np.abs(M1[inds,:] - poi),0)
    return np.mean(dx.prod(1))
