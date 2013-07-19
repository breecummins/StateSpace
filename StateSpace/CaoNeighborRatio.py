import numpy as np
from functools import partial
from scipy.special import gamma
import StateSpaceReconstruction as SSR

def lagFromFirstZeroAutocorrelation(ts,M=None):
    '''
    Calculate autocorrelation of a 1D timeseries and 
    return the first zero crossing.

    '''
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
    '''
    arr must be a list or 1D numpy array

    '''
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
    '''
    Norm of order n>0 between poi and pts, where poi is a 1D array
    of length d, and pts is an mxd array or a 1D array of length d. 

    '''
    try:
        return (((np.abs(pts - poi))**n).sum(1))**(1./n)
    except:
        return (((np.abs(pts - poi))**n).sum())**(1./n)

def Linf(poi,pts):
    '''
    Infinity norm between poi and pts, where poi is a 1D array
    of length d, and pts is an mxd array or a 1D array of length d. 

    '''
    try:
        return (np.abs(pts - poi)).max(1)
    except:
        return (np.abs(pts - poi)).max()

def findNearestNeighbor(poi,pts,norm=None):
    '''
    Find the closest distinct neighbor in pts (numpy array) 
    to poi using some norm (Ln for n >0 and Linf implemented).

    '''
    dists = norm(poi,pts)
    # Throw out all pts with dist=0
    i = dists.argmin()
    while dists[i] == 0:
        dists[i] = np.Inf
        i = dists.argmin()
    return i, dists[i]

def CaoNeighborRatio(ts,lagsize,randinds,dims=4,norm=Linf):
    # add extra dims that will get absorbed during the process
    # of making ratios
    dims = dims + 3
    # make all the reconstructions and shift them so that the 
    # indices always refer to the same point
    manifolds = []
    shifts = [(_d-1)*lagsize for _d in range(1,dims)]
    corrections = [shifts[-1] - _s for _s in shifts]
    L = len(randinds)
    randinds = set(randinds).intersection(range(len(ts)-shifts[-1]))
    if len(randinds) != L:
        print('Number of random indices changed from {0} to {1}'.format(L,len(randinds)))
    for d in range(1,dims):
        m = SSR.makeShadowManifold(ts,d,lagsize)
        manifolds.append(m[corrections[d-1]:,:])
    # now find nearest neighbors and calculate Cao ratio
    E = []
    for d in range(len(manifolds)-1):
        m = manifolds[d]
        m1 = manifolds[d+1]
        rats = []
        for k in randinds:
            nk, distknk = findNearestNeighbor(m[k,:],m,norm)
            dist1 = norm(m1[k,:],m1[nk,:])
            rats.append(dist1/distknk)
        E.append(np.mean(rats))
    return [E[k+1] / E[k] for k in range(len(E)-1)]

def CaoNeighborRatio2(ts,lagsize,randinds,thresh=0.97,norm=Linf):
    '''
    Adaptive Cao neighbor ratio. Test embedding dimensions starting at 2 and 
    continue until stopping criterion is met.
    Stopping criterion: Get 2 above thresholds in a row, then return the dimension
    of the first. (This is a very crude test for convergence.)
    '''
    d0 = 2
    shifts = [(_d-1)*lagsize for _d in range(d0,d0+3)]
    corrections = [shifts[-1] - _s for _s in shifts]
    manifolds = []
    abovethresh = 0
    d = 1
    while abovethresh < 2:
        d += 1
        print('dim: {0}'.format(d))
        L = len(randinds)
        randinds = set(randinds).intersection(range(len(ts)-(d+1)*lagsize))
        if len(randinds) != L:
            print('Number of random indices changed from {0} to {1}'.format(L,len(randinds)))
        dif = d - len(manifolds)
        for k in range(dif,-1,-1):
            dim = d+d0-k
            m = SSR.makeShadowManifold(ts,dim,lagsize)
            manifolds.append(m)
        e=[]
        for j,i in enumerate(range(d-d0,d-d0+2)):
            m = manifolds[i][corrections[j]:,:]
            m1 = manifolds[i+1][corrections[j+1]:,:]
            rats = []
            for k in randinds:
                nk, distknk = findNearestNeighbor(m[k,:],m,norm)
                dist1 = norm(m1[k,:],m1[nk,:])
                rats.append(dist1/distknk)
            e.append(np.mean(rats))
        E = e[1]/e[0]
        print('Neighbor ratio: {0}'.format(E))
        if E > thresh:
            abovethresh += 1
        elif abovethresh == 1:
            abovethresh = 0
    return d-1



def getLagDim(timeseries,cols=None,dims=10,randinds=None):
    '''
    Employs an arbitrary threshold to choose the embedding dim.
    A fixed number of embedding dimensions are tested for each
    timeseries.

    '''
    thresh = 0.97
    if cols == None:
        cols = range(timeseries.shape[1])
    if randinds == None:
        randinds = range(timeseries.shape[0]) 
    lg = []
    rats = []
    for c in cols:
        print('Variable {0}'.format(c))
        ts = np.squeeze(timeseries[:,c])
        lg.append(lagFromFirstZeroAutocorrelation(ts))
        print('Lagsize: {0}'.format(lg[-1]))
        rats.append(CaoNeighborRatio(ts,lg[-1],randinds,dims=dims))
    if len(lg)> 1:
        diffs = [abs(lg[k] - lg[j]) for k in range(len(lg)) for j in range(k+1,len(lg))]
        if max(diffs) > 2:
            print('Discrepancy in lagsizes: {0}'.format(lg))
            print('Embedding dimensions: {0}'.format(nl))
        else:
            print('Lagsizes: {0}'.format(lg))
            print('Embedding dimensions: {0}'.format(nl))
    else:
        print('Lagsize: {0}'.format(lg))
        print('Embedding dimension: {0}'.format(nl))
    inds = [min([i+1 for i in range(len(rat)) if rat[i] > thresh]) for rat in rats]
    numlags = max(inds)
    lagsize = lg[inds.index(numlags)]
    return lagsize, numlags

def getLagDim2(timeseries,cols=None,thresh=0.97,randinds=None):
    '''
    Employs an arbitrary threshold to choose the embedding dim.

    '''
    if cols == None:
        cols = range(timeseries.shape[1])
    if randinds == None:
        randinds = range(timeseries.shape[0]) 
    lg = []
    nl = []
    for c in cols:
        print('Variable {0}'.format(c))
        ts = np.squeeze(timeseries[:,c])
        lg.append(lagFromFirstZeroAutocorrelation(ts))
        print('Lagsize: {0}'.format(lg[-1]))
        nl.append(CaoNeighborRatio2(ts,lg[-1],randinds,thresh=thresh))
    if len(lg)> 1:
        diffs = [abs(lg[k] - lg[j]) for k in range(len(lg)) for j in range(k+1,len(lg))]
        if max(diffs) > 2:
            print('Discrepancy in lagsizes: {0}'.format(lg))
            print('Embedding dimensions: {0}'.format(nl))
        else:
            print('Lagsizes: {0}'.format(lg))
            print('Embedding dimensions: {0}'.format(nl))
    else:
        print('Lagsize: {0}'.format(lg))
        print('Embedding dimension: {0}'.format(nl))
    numlags = max(nl)
    lagsize = lg[nl.index(numlags)]
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

    import DoublePendulum, random
    print('Double pendulum attractor')
    dt = 0.1
    finaltime = 1200.0
    timeseries = DoublePendulum.solvePendulum([1.0,2.0,3.0,2.0],finaltime,dt)
    startind = int(50/dt)#2000 #how much to cut off the front
    timeseries = timeseries[startind:,:]
    N = len(timeseries)
    randinds = random.sample(range(N-int(N/10.)),int(N/5.))
    lagsize,numlags=getLagDim2(timeseries,cols=[0,3],randinds=randinds)
    print('Lagsize: {0}'.format(lagsize))
    print('Embedding dimension: {0}'.format(numlags))