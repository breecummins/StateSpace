import numpy as np
from functools import partial
from scipy.special import gamma
import StateSpaceReconstruction as SSR

def lagFromFirstZeroAutocorrelation(ts,M=None):
    mu = np.mean(ts)
    s2 = np.var(ts,ddof=1) #unbiased estimator of variance
    N = len(ts)
    if M == None:
        M = np.floor(N/100.)
    autocc = []
    for k in range(1,M+1):
        autocc.append( ((ts[:N-k] - mu)*(ts[k:] - mu)).sum() / ( s2 *(N-k) ) )
    return findFirstZero(autocc)

def findFirstZero(arr):
    arr = np.array(arr)
    up = np.nonzero(arr >= 0)[0]
    down = np.nonzero(arr < 0)[0]
    for i in up:
        if i+1 in down:
            if np.abs(arr[i]) >= np.abs(arr[i+1]):
                return i+1
            else:
                return i
        if i-1 in down:
            if np.abs(arr[i]) >= np.abs(arr[i-1]):
                return i-1
            else:
                return i
    return None

def Ln(poi,pts,n=None):
    try:
        return (((np.abs(pts - poi))**n).sum(1))**(1./n)
    except:
        return (((np.abs(pts - poi))**n).sum())**(1./n)

def Linf(poi,pts):
    try:
        return (np.abs(pts - poi)).max(1)
    except:
        return (np.abs(pts - poi)).max()

def findNearestNeighbor(poi,pts,norm=None):
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
    times = np.arange(0,10*np.pi,0.05)
    ts1 = np.cos(times)
    ts2 = ts1 + 0.4*np.random.normal(ts1.shape)
    lagsize = 30 #found empirically after trying many smaller taus, used the fact that I *knew* dim=2 is the correct answer
    # CaoNeighborRatio(ts1,lagsize,10)
    # CaoNeighborRatio(ts2,lagsize,10)

    M=200
    lagsize1 = lagFromFirstZeroAutocorrelation(ts1,M)
    print(lagsize1)
    lagsize2 = lagFromFirstZeroAutocorrelation(ts2,M)
    print(lagsize2)
    print('Linf')
    CaoNeighborRatio(ts1,lagsize1,dims=10,norm=Linf)
    CaoNeighborRatio(ts2,lagsize2,dims=10,norm=Linf)
    print('L5')
    CaoNeighborRatio(ts1,lagsize1,dims=10,norm=partial(Ln,n=5))
    CaoNeighborRatio(ts2,lagsize2,dims=10,norm=partial(Ln,n=5))
    print('L2')
    CaoNeighborRatio(ts1,lagsize1,dims=10,norm=partial(Ln,n=2))
    CaoNeighborRatio(ts2,lagsize2,dims=10,norm=partial(Ln,n=2))
    print('L1')
    CaoNeighborRatio(ts1,lagsize1,dims=10,norm=partial(Ln,n=1))
    CaoNeighborRatio(ts2,lagsize2,dims=10,norm=partial(Ln,n=1))
    print('L1/2')
    CaoNeighborRatio(ts1,lagsize1,dims=10,norm=partial(Ln,n=0.5))
    CaoNeighborRatio(ts2,lagsize2,dims=10,norm=partial(Ln,n=0.5))
