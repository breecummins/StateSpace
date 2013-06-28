import numpy as np
import StateSpaceReconstruction as SSR

# This is an implementation of Cao's nearest neighbor method using data from a perfect circle
# and from a noisy circle. 

times = np.arange(0,10*np.pi,0.05)
ts = np.cos(times)
ts2 = ts + 0.1*np.random.normal(ts.shape)
lagsize = 30

def L1(poi,pts):
    try:    
        return (np.abs(pts - poi)).sum(1)
    except:
        return (np.abs(pts - poi)).sum()

def L2(poi,pts):
    try:
        return np.sqrt(((pts - poi)**2).sum(1))
    except:
        return np.sqrt(((pts - poi)**2).sum())

def Linf(poi,pts):
    try:
        return (np.abs(pts - poi)).max(1)
    except:
        return (np.abs(pts - poi)).max()

def findNearestNeighbor(poi,pts,norm=L2):
    '''
    Find the closest distinct neighbor in pts (numpy array) 
    to poi.

    '''
    dists = norm(poi,pts)
    # Throw out all pts with dist=0
    i = dists.argmin()
    while dists[i] == 0:
        dists[i] = np.Inf
        i = dists.argmin()
    return i, dists[i]

def CaoNeighborRatio(ts,lagsize,dims=4,norm=Linf):
    # first make all the reconstructions and shift them so that the 
    # indices always refer to the same point
    manifolds = []
    shifts = [(d-1)*lagsize for d in range(1,dims+1)]
    corrections = [shifts[-1] - s for s in shifts]
    for d in range(1,dims+1):
        m = SSR.makeShadowManifold(ts,d,lagsize)
        manifolds.append(m[corrections[d-1]:,:])
    # now find nearest neighbors and calculate Cao ratio
    E = []
    for d in range(dims-1):
        m = manifolds[d]
        m1 = manifolds[d+1]
        rats = []
        for k in range(len(ts)-shifts[-1]):
            nk, distknk = findNearestNeighbor(m[k,:],m,norm)
            dist1 = norm(m1[k,:],m1[nk,:])
            rats.append(dist1/distknk)
        E.append(np.mean(rats))
    print([E[k+1] / E[k] for k in range(dims-2)])

if __name__ == '__main__':
    CaoNeighborRatio(ts,lagsize,10)
    CaoNeighborRatio(ts2,lagsize,10)