#import python modules
import numpy as np
import random
#import home-rolled modules
import StateSpaceReconstruction as SSR
import Weights
import Similarity


def estManifold(Mx,My,wgtfunc):
    '''
    Estimate My from Mx.

    '''
    Mest=np.zeros(My.shape)
    for k in range(Mx.shape[0]):
        poi = Mx[k,:]
        dists,inds = Weights.findClosest(poi,Mx,Mx.shape[1]+1)
        w = wgtfunc(np.array(dists))
        pts = [My[j,:] for j in inds]
        Mest[k,:] = np.array([w[j]*pts[j] for j in range(len(w))]).sum(0)
    return Mest

def crossMapModified1(M1,M2,wgtfunc):
    '''
    Estimate manifold 1 (M1) from manifold 2 and vice versa using a 
    cross-mapping technique.
    Find the nearest points to each point in M1.
    Construct weights based on their distances to the point of
    interest. 
    Use the time indices of these points and the associated weights to 
    make a weighted-sum estimate of the contemporaneous points in M2.

    '''
    Mest1 = estManifold(M2,M1,wgtfunc)
    Mest2 = estManifold(M1,M2,wgtfunc)
    return Mest1, Mest2

def testCausalityModified1(ts1,ts2,numlags,lagsize,listoflens,numiters,wgtfunc=Weights.makeExpWeights):
    '''
    Check for convergence (Sugihara) to infer causality between ts1 and ts2.
    ts1 and ts2 must have the same length.
    numlags is the dimension of the embedding space for the reconstruction.
    Use time lags of size lagsize * dt to construct shadow manifolds. lagsize
    is an integer representing the index of the time lag.
    listoflens contains the lengths to use to show convergence 
    Example: range(100,10000,100)
    Each length will be run numiters times from different random starting 
    locations in the time series. numiters must be <= len(ts1) - max(listoflens).
    The estimated time series will be constructed using the weighting function 
    handle given by wgtfunc.

    '''
    L = len(ts1)
    dt = ts1[1] - ts2[0] #assume uniform sampling in time
    if len(ts2) != L:
        raise(ValueError,"The lengths of the two time series must be the same.")
    listoflens.sort()
    lol = [l for l in listoflens if l < L]
    avgcc1=[]
    stdcc1=[]
    avgcc2=[]
    stdcc2=[]
    for l in lol:
        startinds = random.sample(range(L-l-numlags*lagsize),numiters)
        cc1=[]
        cc2=[]
        for s in startinds:
            M1 = np.array(list(SSR.makeShadowManifold(ts1[s:s+l],numlags,lagsize)))
            M2 = np.array(list(SSR.makeShadowManifold(ts2[s:s+l],numlags,lagsize)))
            Mest1,Mest2 = crossMapModified1(M1,M2,wgtfunc)
            cc1.append(Similarity.RootMeanSquaredError(Mest1,M1))
            cc2.append(Similarity.RootMeanSquaredError(Mest2,M2))
        avgcc1.append(np.mean(np.array(cc1)))
        avgcc2.append(np.mean(np.array(cc2)))
        stdcc1.append(np.std(np.array(cc1)))
        stdcc2.append(np.std(np.array(cc2)))
    return lol,avgcc1,avgcc2,stdcc1,stdcc2

def crossMapModified2(M1,M2,wgtfunc):
    '''
    Estimate manifold 1 (M1) from manifold 2 and vice versa using a 
    cross-mapping technique.
    Find the nearest points to each point in M1.
    Construct weights based on their distances to the point of
    interest. 
    Use the time indices of these points and the associated weights to 
    make a weighted-sum estimate of the contemporaneous points in M2.
    Average the columns of the estimated manifold to get an estimated 
    time series.

    '''
    Mest1 = estManifold(M2,M1,wgtfunc)
    Mest2 = estManifold(M1,M2,wgtfunc)
    def estSeries(Mest):
        n = Mest.shape[1]
        N = Mest.shape[0]
        est = np.zeros(N)
        for k in range(1,n+1):
            est[:-n+1] += Mest[n-k:N-k+1,k-1]/n
        for k in range(n-1,0,-1):
            q = 0
            c = n-1 -k +1
            while q < k:
                est[-k] += Mest[N-1-q,q+c] / k 
                q+=1
        return est
    est1 = estSeries(Mest1)
    est2 = estSeries(Mest2)
    return est1, est2

def crossMapModified3(M1,M2,proj,wgtfunc):
    '''
    Estimate manifold 1 (M1) from manifold 2 and vice versa using a 
    cross-mapping technique.
    Find the nearest points to each point in M1.
    Construct weights based on their distances to the point of
    interest. 
    Use the time indices of these points and the associated weights to 
    make a weighted-sum estimate of the contemporaneous points in M2.
    Take a projection of the estimated manifold on dimension proj.

    '''
    Mest1 = estManifold(M2,M1,wgtfunc)
    Mest2 = estManifold(M1,M2,wgtfunc)
    est1 = Mest1[:,proj]
    est2 = Mest2[:,proj]
    return est1, est2

