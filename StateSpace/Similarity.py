import numpy as np
import Weights

def corrCoeffPearson(ts1,ts2):
    '''
    Compare a times series and its estimate.

    '''
    shift1 = ts1 - np.mean(ts1)
    shift2 = ts2 - np.mean(ts2)
    s12 = ( shift1*shift2 ).sum()
    s11 = ( shift1*shift1 ).sum()
    s22 = ( shift2*shift2 ).sum()
    return s12 / np.sqrt(s11*s22)

def RootMeanSquaredErrorTS(ts1,ts2):
    '''
    Compare a sampled manifold and its estimate at contemporaneous
    points using root mean squared error.

    '''
    return np.sqrt( ((ts1-ts2)**2).sum() / len(ts1))

def RootMeanSquaredErrorManifold(M1,M2):
    '''
    Compare a sampled manifold and its estimate at contemporaneous
    points using root mean squared error.

    '''
    return np.sqrt( ((M1-M2)**2).sum() / M1.shape[0]*M1.shape[1])

def HausdorffDistance(M1,M2):
    '''
    Compare a sampled manifold and its estimate using the Hausdorff
    distance. This ignores relationships between contemporary
    times. 
    
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

def countingMeasure(M1,M2,N):
    '''
    For each point y in M2, find the contemporaneous point x in M1 and the set
    of points containing the N nearest neighbors of the N nearest neighbors of x. 
    Count the points in this neighborhood that share a time index 
    with any of the N nearest neighbors of y. Normalize by N and average over
    the number of points in the manifold (M1.shape[1] = M2.shape[2]).
    This is analogous to estimating M1 from M2 using Sugihara's method and 
    testing for causality in the 1 -> 2 direction. In Sugihara's method, the N 
    nearest neighbors of y are assumed to be "close to" x. So I check for these 
    points in a larger radius around x. In terms of maps, this tests for a 1-1 map 
    in the M2 -> M1 direction.
    Also do the same calculation starting at x in M1.
    Outputs are returned in the M1 -> M2, M2 -> M1 order for consistency 
    with neighborDistance.

    '''
    def friendsoffriends(inds,M):
        indsbig = []
        for i in inds:
            junk,indsb = findClosestInclusive(M[i,:],M,N)
            indsbig.extend(indsb)
        return list(set(indsbig))

    def countme(inds,indsbig):
        mc = 0.0
        for j in range(N):
            if inds[j] in indsbig:
                mc += 1
        return mc

    mycount1 = np.zeros(M1.shape[0])
    mycount2 = np.zeros(M1.shape[0])
    for k in range(M1.shape[0]):
        poi1 = M1[k,:]
        junk,inds1 = findClosestInclusive(poi1,M1,N)
        inds1big = friendsoffriends(inds1,M1)
        poi2 = M2[k,:]
        junk,inds2 = findClosestInclusive(poi2,M2,N)
        inds2big = friendsoffriends(inds2,M2)
        mycount1[k] = countme(inds2,inds1big)
        mycount2[k] = countme(inds1,inds2big)
    return np.mean(mycount2) / N, np.mean(mycount1) / N

def neighborDistance(M1,M2,N):
    '''
    For each pair of contemporaneous points x and y in M1 and M2 respectively, 
    find the  N nearest neighbors of each. 
    Map the neighbors of x (y) into M2 (M1) and sum the Euclidean distances to y (x). 
    Normalize by the sum of the distances between y (x) and its n+1 nearest 
    neighbors. Subtract one which is the smallest possible value to shift the range onto
    the positive reals. Then average over the number of points in the manifolds. 
    A small distance (close to 0) is evidence for a topology preserving map 
    in the M1 -> M2 (M2 -> M1) direction, which implies that M2 (M1) can be 
    reconstructed from M1 (M2), which implies that 2 is a driver of 1 (Sugihara's assertion). 

    Output returned so that M1 -> M2 (sum in M2) is first, M2 -> M1 (sum in M1) second.

    '''
    ndistsx = np.zeros(M1.shape[0])
    ndistsy = np.zeros(M1.shape[0])
    for k in range(M1.shape[0]):
        x = M1[k,:]
        distsx,indsx = findClosestInclusive(x,M1,N)
        y = M2[k,:]
        distsy,indsy = findClosestInclusive(y,M2,N)
        ndistsx[k] = (np.sqrt(((M1[indsy,:] - x)**2).sum(1)).sum(0) / np.array(distsx).sum()) - 1.0
        ndistsy[k] = (np.sqrt(((M2[indsx,:] - y)**2).sum(1)).sum(0) / np.array(distsy).sum()) - 1.0
    return np.mean(ndistsy),np.mean(ndistsx)

def maxNeighborDistArray(M1,M2,N,poi=None):
    if not poi:
        poi = range(M1.shape[0])
    ndistsx = np.zeros(len(poi))
    ndistsy = np.zeros(len(poi))
    for k,ind in enumerate(poi):
        x = M1[ind,:]
        distsx,indsx = findClosestInclusive(x,M1,N)
        y = M2[ind,:]
        distsy,indsy = findClosestInclusive(y,M2,N)
        ndistsx[k] = (np.sqrt(((M1[indsy,:] - x)**2).sum(1))).max()
        ndistsy[k] = (np.sqrt(((M2[indsx,:] - y)**2).sum(1))).max()
    return ndistsy, ndistsx

def maxNeighborDistMean(M1,M2,N,poi=None):
    if not poi:
        poi = range(M1.shape[0])
    ndistsx = np.zeros(len(poi))
    ndistsy = np.zeros(len(poi))
    for k,ind in enumerate(poi):
        x = M1[ind,:]
        distsx,indsx = findClosestInclusive(x,M1,N)
        y = M2[ind,:]
        distsy,indsy = findClosestInclusive(y,M2,N)
        ndistsx[k] = (np.sqrt(((M1[indsy,:] - x)**2).sum(1))).max()
        ndistsy[k] = (np.sqrt(((M2[indsx,:] - y)**2).sum(1))).max()
    return np.mean(ndistsy), np.mean(ndistsx)

def maxNeighborDistMin(M1,M2,N,poi=None):
    if not poi:
        poi = range(M1.shape[0])
    ndistsx = np.zeros(len(poi))
    ndistsy = np.zeros(len(poi))
    for k,ind in enumerate(poi):
        x = M1[ind,:]
        distsx,indsx = findClosestInclusive(x,M1,N)
        y = M2[ind,:]
        distsy,indsy = findClosestInclusive(y,M2,N)
        ndistsx[k] = (np.sqrt(((M1[indsy,:] - x)**2).sum(1))).max()
        ndistsy[k] = (np.sqrt(((M2[indsx,:] - y)**2).sum(1))).max()
    return np.min(ndistsy), np.min(ndistsx)

def maxNeighborDistMax(M1,M2,N,poi=None):
    if not poi:
        poi = range(M1.shape[0])
    ndistsx = np.zeros(len(poi))
    ndistsy = np.zeros(len(poi))
    for k,ind in enumerate(poi):
        x = M1[ind,:]
        distsx,indsx = findClosestInclusive(x,M1,N)
        y = M2[ind,:]
        distsy,indsy = findClosestInclusive(y,M2,N)
        ndistsx[k] = (np.sqrt(((M1[indsy,:] - x)**2).sum(1))).max()
        ndistsy[k] = (np.sqrt(((M2[indsx,:] - y)**2).sum(1))).max()
    return np.max(ndistsy), np.max(ndistsx)

def meanNeighborDist(M1,M2,N,poi=None):
    if not poi:
        poi = range(M1.shape[0])
    ndistsx = np.zeros(len(poi))
    ndistsy = np.zeros(len(poi))
    for k,ind in enumerate(poi):
        x = M1[ind,:]
        distsx,indsx = findClosestInclusive(x,M1,N)
        y = M2[ind,:]
        distsy,indsy = findClosestInclusive(y,M2,N)
        ndistsx[k] = np.mean(np.sqrt(((M1[indsy,:] - x)**2).sum(1)))
        ndistsy[k] = np.mean(np.sqrt(((M2[indsx,:] - y)**2).sum(1)))
    return np.mean(ndistsy), np.mean(ndistsx)

#Below here are failed experiments

def compareLocalDiams(M1,M2):
    '''
    Method to calculate the local diameter change in both directions.

    Note: Unfortunately, the forward and backward diameters are not 
    related. I would need the smallest dimension in the backwards map, 
    not the largest, and I don't know how to reliably get that.

    ''' 
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

    Note: Never used this for anything.

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