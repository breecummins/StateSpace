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
    '''
    Find the Hausdorff distance between two sets of points
    in R^d, where d = M1.shape[1] = M2.shape[1].

    '''
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

def countingMeasure(M1,M2):
    '''
    Count the number of nearest neighbors for points in M1 that share a time 
    index with a nearest neighbor of the contemporaneous point in M2. This
    is analogous to estimating M1 from M2 using Sugihara's method. 

    '''
    n = M1.shape[1]+1
    mycount = []
    for k in range(M1.shape[0]):
        poi1 = M1[k,:]
        junk,inds1 = findClosestInclusive(poi1,M1,n)
        poi2 = M2[k,:]
        junk,inds2 = findClosestInclusive(poi2,M2,n)
        mc = 0
        for j in range(n):
            if inds2[j] in inds1:
                mc += 1
        mycount.append(mc)
    return np.mean(np.array(mycount)) / n

def compareLocalDiams(M1,M2):
    def calcDiam(N):
        diam = 0
        for k in range(N.shape[0]):
            for j in range(N.shape[0]):
                d = np.sqrt(((N[k,:]-N[j,:])**2).sum())
                if d > diam:
                    diam=d
        return diam
    n = M1.shape[1]+2
    diamratioprod = []
    for k in range(M1.shape[0]):
        poi1 = M1[k,:]
        _,inds1 = findClosestExclusive(poi1,M1,n)
        diamN1 = calcDiam(M1[inds1,:])
        diamIm1 = calcDiam(M2[inds1,:])
        poi2 = M2[k,:]
        _,inds2 = findClosestExclusive(poi2,M2,n)
        diamN2 = calcDiam(M2[inds2,:])
        diamIm2 = calcDiam(M1[inds2,:])
        diamratioprod.append((diamIm1*diamIm2)/(diamN1*diamN2))
    return np.mean(np.array(diamratioprod))

        
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

if __name__=='__main__':
    import StateSpaceReconstruction as SSR
    from LorenzEqns import solveLorenz
    ts = solveLorenz([1.0,0.5,0.5],80.0)
    numlags = 3
    lagsize = 8
    M1 = SSR.makeShadowManifold(ts[:,0],numlags,lagsize)
    M2 = SSR.makeShadowManifold(ts[:,1],numlags,lagsize)
    meanprodratiodiams = compareLocalDiams(M1,M2)
    print(meanprodratiodiams)