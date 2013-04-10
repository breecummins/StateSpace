import StateSpaceReconstruction as SSR
import numpy as np
import random

def findClosest(poi,pts,N):
    '''
    Find the closest N points in pts (numpy array) to poi.

    '''
    dists = np.sqrt(((pts - poi)**2).sum(1))
    out = []
    for _ in range(N+1):
        i = dists.argmin()
        out.append((dists[i],i))
        dists[i] = np.Inf
    return zip(*out[1:])

def makeExpWeights(dists):
    u = np.exp( -dists/dists[0])
    return u / u.sum()

def crossMap(ts1,ts2,numlags,lagsize,wgtfunc):
    '''
    Estimate timeseries 1 (ts1) from timeseries 2 and vice versa using a 
    cross-mapping technique between shadow manifolds.
    Construct a shadow manifold M from one time series.
    Find the nearest points to each point in M.
    Construct weights based on their distances to the point of
    interest. 
    Use the time indices of these points and the associated weights to 
    make a weighted-sum estimate of the contemporaneous points in the 
    other time series.

    '''
    M1 = np.array(list(SSR.makeShadowManifold(ts1,numlags,lagsize)))
    M2 = np.array(list(SSR.makeShadowManifold(ts2,numlags,lagsize)))
    def estSeries(M,ts):
        est=np.zeros(ts.shape)
        for k in range(M.shape[0]):
            poi = M[k,:]
            dists,inds = findClosest(poi,M,numlags+1)
            w = wgtfunc(np.array(dists))
            est[k] = (w*ts[list(inds)]).sum()
        return est
    est1 = estSeries(M2,M1[:,-1])
    est2 = estSeries(M1,M2[:,-1])
    return est1, est2

def corrCoeffPearson(ts1,ts2):
    shift1 = ts1 - np.mean(ts1)
    shift2 = ts2 - np.mean(ts2)
    s12 = ( shift1*shift2 ).sum()
    s11 = ( shift1*shift1 ).sum()
    s22 = ( shift2*shift2 ).sum()
    return s12 / np.sqrt(s11*s22)
    
def testCausality(ts1,ts2,numlags,lagsize,listoflens,numiters,wgtfunc=makeExpWeights):
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
    if len(ts2) != L:
        raise(ValueError,"The lengths of the two time series must be the same.")
    listoflens.sort()
    lol = [l for l in listoflens if l < L]
    avgcc1=[]
    stdcc1=[]
    avgcc2=[]
    stdcc2=[]
    for l in lol:
        startinds = random.sample(range(L-l),numiters)
        cc1=[]
        cc2=[]
        for s in startinds:
            est1,est2 = crossMap(ts1[s:s+l],ts2[s:s+l],numlags,lagsize,wgtfunc)
            #correct for the time points lost in shadow manifold construction
            s1 = s+(numlags-1)*lagsize 
            l1 = len(est1)              
            cc1.append(corrCoeffPearson(est1,ts1[s1:s1+l1]))
            cc2.append(corrCoeffPearson(est2,ts2[s1:s1+l1]))
        avgcc1.append(np.mean(np.array(cc1)))
        avgcc2.append(np.mean(np.array(cc2)))
        stdcc1.append(np.std(np.array(cc1)))
        stdcc2.append(np.std(np.array(cc2)))
    return lol,avgcc1,avgcc2,stdcc1,stdcc2
    
if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    from LorenzEqns import solveLorenz
    timeseries = solveLorenz([1.0,0.5,0.5],80.0)
    l,avg1,avg2,std1,std2 = testCausality(timeseries[:,0],timeseries[:,1],2,8,range(25,200,15),75) 
    print(np.array(l))
    print(np.array([avg1,avg2]))
    avgarr = np.zeros((len(avg1),2))
    avgarr[:,0] = avg1
    avgarr[:,1] = avg2
    SSRPlots.plots(np.array(l),avgarr,stylestr=['b-','r-'],leglabels=['x from My','y from Mx'],xstr='length of time interval',ystr='mean corr coeff')       
        




     