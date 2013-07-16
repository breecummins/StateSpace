import numpy as np
from functools import partial
from scipy.special import gamma
import StateSpaceReconstruction as SSR

def lagFromFirstZeroAutocorrelation(ts,M=None):
    mu = np.mean(ts)
    s2 = np.var(ts,ddof=1) #unbiased estimator of variance
    N = len(ts)
    if M == None:
        M = int(np.floor(N/10.))
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
    # add extra dims that will get absorbed during the process
    # of making ratios
    dims = dims + 3
    # make all the reconstructions and shift them so that the 
    # indices always refer to the same point
    manifolds = []
    shifts = [(d-1)*lagsize for d in range(1,dims)]
    corrections = [shifts[-1] - s for s in shifts]
    for d in range(1,dims):
        m = SSR.makeShadowManifold(ts,d,lagsize)
        manifolds.append(m[corrections[d-1]:,:])
    # now find nearest neighbors and calculate Cao ratio
    E = []
    for d in range(len(manifolds)-1):
        m = manifolds[d]
        m1 = manifolds[d+1]
        rats = []
        for k in range(len(ts)-shifts[-1]):
            nk, distknk = findNearestNeighbor(m[k,:],m,norm)
            dist1 = norm(m1[k,:],m1[nk,:])
            rats.append(dist1/distknk)
        E.append(np.mean(rats))
    return [E[k+1] / E[k] for k in range(len(E)-1)]

def getLagDim(arr,cols=None,dims=10):
    '''
    Employs an arbitrary threshold to choose the embedding dim.

    '''
    thresh = 0.97
    if cols == None:
        cols = range(arr.shape[1])
    lg = []
    rats = []
    for c in cols:
        ts = np.squeeze(arr[:,c])
        lg.append(lagFromFirstZeroAutocorrelation(ts))
        rats.append(CaoNeighborRatio(ts,lg[-1],dims=dims))
    if len(lg)> 1:
        diffs = [abs(lg[k] - lg[j]) for k in range(len(lg)) for j in range(k+1,len(lg))]
        if max(diffs) > 2:
            print('Discrepancy in lagsizes: {0}'.format(lg))
    inds = [min([i+1 for i in range(len(rat)) if rat[i] > thresh]) for rat in rats]
    numlags = max(inds)
    lagsize = lg[inds.index(numlags)]
    return lagsize, numlags

if __name__ == '__main__':
    # print('Circle examples - perfect and noisy')
    # dt = 0.05
    # times = np.arange(0,10*np.pi,dt)
    # ts1 = np.cos(times)
    # ts2 = ts1 + 0.4*np.random.normal(ts1.shape)
    # print('dt = {0}'.format(dt))
    # lagsize = 10 #arbitrary
    # print('arbitrary lagsize = {0}'.format(lagsize))
    # print('Cao neighbor method for embedding dimension, perfect and noisy circle')
    # CaoNeighborRatio(ts1,lagsize,10)
    # CaoNeighborRatio(ts2,lagsize,10)

    # # M=200
    # lagsize1 = lagFromFirstZeroAutocorrelation(ts1)
    # print('autocorr lagsize = {0}'.format(lagsize1))
    # lagsize2 = lagFromFirstZeroAutocorrelation(ts2)
    # print('noisy autocorr lagsize = {0}'.format(lagsize2))
    # print('Cao neighbor method for embedding dimension, perfect and noisy circle')
    # CaoNeighborRatio(ts1,lagsize1,dims=10,norm=Linf)
    # CaoNeighborRatio(ts2,lagsize2,dims=10,norm=Linf)
    # # print('L5')
    # # CaoNeighborRatio(ts1,lagsize1,dims=10,norm=partial(Ln,n=5))
    # # CaoNeighborRatio(ts2,lagsize2,dims=10,norm=partial(Ln,n=5))
    # # print('L2')
    # # CaoNeighborRatio(ts1,lagsize1,dims=10,norm=partial(Ln,n=2))
    # # CaoNeighborRatio(ts2,lagsize2,dims=10,norm=partial(Ln,n=2))
    # # print('L1')
    # # CaoNeighborRatio(ts1,lagsize1,dims=10,norm=partial(Ln,n=1))
    # # CaoNeighborRatio(ts2,lagsize2,dims=10,norm=partial(Ln,n=1))
    # # print('L1/2')
    # # CaoNeighborRatio(ts1,lagsize1,dims=10,norm=partial(Ln,n=0.5))
    # # CaoNeighborRatio(ts2,lagsize2,dims=10,norm=partial(Ln,n=0.5))

    # dt = 0.01
    # times = np.arange(0,10*np.pi,dt)
    # ts1 = np.cos(times)
    # ts2 = ts1 + 0.4*np.random.normal(ts1.shape)
    # print('dt = {0}'.format(dt))
    # # M=1000
    # lagsize1 = lagFromFirstZeroAutocorrelation(ts1)
    # print('autocorr lagsize = {0}'.format(lagsize1))
    # lagsize2 = lagFromFirstZeroAutocorrelation(ts2)
    # print('noisy autocorr lagsize = {0}'.format(lagsize2))
    # print('Cao neighbor method for embedding dimension, perfect and noisy circle')
    # CaoNeighborRatio(ts1,lagsize1,dims=10,norm=Linf)
    # CaoNeighborRatio(ts2,lagsize2,dims=10,norm=Linf)

    # import LorenzEqns
    # print('Lorenz attractor')
    # dt = 0.01
    # finaltime = 80.0
    # timeseries = LorenzEqns.solveLorenz([1.0,0.5,0.5],finaltime,dt)
    # ts = np.squeeze(timeseries[:,1])
    # lagsize = int(0.08/dt) #because lagsize=8 is good with dt = 0.01
    # print('eyeball lagsize = {0}'.format(lagsize))
    # print('Cao neighbor method for embedding dimension')
    # CaoNeighborRatio(ts,lagsize,dims=10,norm=Linf)
    # M=2000
    # lagsize1 = lagFromFirstZeroAutocorrelation(ts,M)
    # print('autocorr lagsize = {0}'.format(lagsize1))
    # print('Cao neighbor method for embedding dimension, Lorenz attractor')
    # CaoNeighborRatio(ts,lagsize1,dims=10,norm=Linf)

    import DoublePendulum
    print('Double pendulum attractor')
    dt = 0.1
    finaltime = 1200.0
    timeseries = DoublePendulum.solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt)
    lagsize,numlags=getLagDim(timeseries,cols=[0,3],dims=15)
    print(lagsize)
    print(numlags)