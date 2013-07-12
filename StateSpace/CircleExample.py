import numpy as np
from scipy.special import gamma
import StateSpaceReconstruction as SSR

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

def lagFromFirstMinMutualInfo(ts,M=None):
    #FIXME: stub
    mi = None
    return findFirstMin(mi)

def findFirstMin(arr):
    # This method of finding minima is sensitive to noise.
    d = [arr[_k+1] - arr[_k] for _k in range(len(arr)-1)]
    inds = [_k for _k in range(1,len(d)-2) if -d[_k-1] >= 0 and -d[_k] >=0 and d[_k+1] >= 0 and d[_k+2] >= 0]
    return min(inds)

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
    times = np.arange(0,10*np.pi,0.05)
    ts1 = np.cos(times)
    ts2 = ts1 + 0.4*np.random.normal(ts1.shape)
    lagsize = 30 #found empirically after trying many smaller taus, used the fact that I *knew* dim=2 is the correct answer
    CaoNeighborRatio(ts1,lagsize,10)
    CaoNeighborRatio(ts2,lagsize,10)

    M=200
    lagsize1 = lagFromFirstZeroAutocorrelation(ts1,M)
    print(lagsize1)
    lagsize2 = lagFromFirstZeroAutocorrelation(ts2,M)
    print(lagsize2)
    CaoNeighborRatio(ts1,lagsize1,10)
    CaoNeighborRatio(ts2,lagsize2,10)
