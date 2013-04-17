import numpy as np
import Weights

def corrCoeffPearson(ts1,ts2):
    shift1 = ts1 - np.mean(ts1)
    shift2 = ts2 - np.mean(ts2)
    s12 = ( shift1*shift2 ).sum()
    s11 = ( shift1*shift1 ).sum()
    s22 = ( shift2*shift2 ).sum()
    return s12 / np.sqrt(s11*s22)

def RootMeanSquaredError(M1,M2):
    return np.sqrt( ((M1-M2)**2).sum() / M1.shape[0]*M1.shape[1])

def findClosestInclusive(poi,pts,N):
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

def findClosestExclusive(poi,pts,N):
    '''
    Find the closest N points in pts (numpy array) to poi, 
    where poi is not (known to be) a member of pts.

    '''
    dists = np.sqrt(((pts - poi)**2).sum(1))
    out = []
    for _ in range(N):
        i = dists.argmin()
        out.append((dists[i],i))
        dists[i] = np.Inf
    return zip(*out)

def HausdorffDistance(M1,M2):
    def calcMaxMin(X,Y):
        mm = 0.0
        for k in range(X.shape[0]):
            poi = X[k,:]
            d,junk = findClosestExclusive(poi,Y,1)
            if d[0] > mm:
                mm = d[0]
        return mm
    mm1 = calcMaxMin(M1,M2)
    mm2 = calcMaxMin(M2,M1)
    return max(mm1,mm2)

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
        dists,inds=findClosestInclusive(poi,M1,d+1)
        dx[k,:] = np.mean(np.abs(M1[inds,:] - poi),0)
    return np.mean(dx.prod(1))


