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
        dists,inds = Similarity.findClosestInclusive(poi,Mx,Mx.shape[1]+1)
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

def crossMapModified2(M1,M2,wgtfunc):
    '''
    Estimate manifold 1 (M1) from manifold 2 and vice versa using a 
    cross-mapping technique.
    Find the nearest points to each point in M1.
    Construct weights based on their distances to the point of
    interest. 
    Use the time indices of these points and the associated weights to 
    make a weighted-sum estimate of the contemporaneous points in M2.
    Average the shifted columns of the estimated manifold to get an estimated 
    time series.

    '''
    Mest1 = estManifold(M2,M1,wgtfunc)
    Mest2 = estManifold(M1,M2,wgtfunc)
    def estSeries(Mest):
        n = Mest.shape[1]
        N = Mest.shape[0]
        est = np.zeros(N)
        for k in range(n):
            est[:-n+1] += Mest[k:N-n+1+k,k]/n
        for k in range(N-n+1,N):
            q = 0
            while q < N-k:
                est[k] += Mest[k+q,q] / (N - k) 
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

def testCausalityModified(ts1,ts2,numlags,lagsize,listoflens,numiters,CM=crossMapModified1,wgtfunc=Weights.makeExpWeights,simMeasure=Similarity.RootMeanSquaredError):
    '''
    Check for convergence in root mean squared error to infer causality between ts1 and ts2.
    ts1 and ts2 must have the same length.
    numlags is the dimension of the embedding space for the reconstruction.
    Use time lags of size lagsize * dt to construct shadow manifolds. lagsize
    is an integer representing the index of the time lag.
    listoflens contains the lengths to use to show convergence 
    Example: range(100,10000,100)
    Each length will be run numiters times from different random starting 
    locations in the time series. numiters must be <= len(ts1) - max(listoflens).
    The estimated time series will be constructed using the weighting function 
    handle given by wgtfunc and the cross map function given by CM.

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
        startinds = random.sample(range(L-l),numiters)
        cc1=[]
        cc2=[]
        for s in startinds:
            M1 = SSR.makeShadowManifold(ts1[s:s+l],numlags,lagsize)
            M2 = SSR.makeShadowManifold(ts2[s:s+l],numlags,lagsize)
            Mest1,Mest2 = CM(M1,M2,wgtfunc)
            cc1.append(simMeasure(Mest1,M1))
            cc2.append(simMeasure(Mest2,M2))
        avgcc1.append(np.mean(np.array(cc1)))
        avgcc2.append(np.mean(np.array(cc2)))
        stdcc1.append(np.std(np.array(cc1)))
        stdcc2.append(np.std(np.array(cc2)))
    return lol,avgcc1,avgcc2,stdcc1,stdcc2

def testDiffeomorphism(ts1,ts2,numlags,lagsize,listoflens,numiters,simMeasure=Similarity.countingMeasure):
    '''
    Check for diffeomorphism between shadow manifolds constructed from ts1 and ts2.
    ts1 and ts2 must have the same length.
    numlags is the dimension of the embedding space for the reconstruction.
    Use time lags of size lagsize * dt to construct shadow manifolds. lagsize
    is an integer representing the index of the time lag.
    listoflens contains the lengths to use to show convergence 
    Example: range(100,10000,100)
    Each length will be run numiters times from different random starting 
    locations in the time series. numiters must be <= len(ts1) - max(listoflens).
    Neighborhoods of contemporaneous points will be assessed for similarity using
    simMeasure.

    '''
    L = len(ts1)
    dt = ts1[1] - ts2[0] #assume uniform sampling in time
    if len(ts2) != L:
        raise(ValueError,"The lengths of the two time series must be the same.")
    listoflens.sort()
    lol = [l for l in listoflens if l < L]
    avgcc1=[]
    stdcc1=[]
    for l in lol:
        startinds = random.sample(range(L-l),numiters)
        cc1=[]
        for s in startinds:
            M1 = SSR.makeShadowManifold(ts1[s:s+l],numlags,lagsize)
            M2 = SSR.makeShadowManifold(ts2[s:s+l],numlags,lagsize)
            cc1.append(simMeasure(M1,M2))
        avgcc1.append(np.mean(np.array(cc1)))
        stdcc1.append(np.std(np.array(cc1)))
    return lol,avgcc1,stdcc1

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    from LorenzEqns import solveLorenz
    timeseries = solveLorenz([1.0,0.5,0.5],80.0)
    # l,avg1,avg2,std1,std2 = testCausalityModified(timeseries[:,0],timeseries[:,1],2,8,range(20,3001,200),25,simMeasure=Similarity.HausdorffDistance) 
    # from differenceEqns import solve2Species
    # timeseries = solve2Species([0.4,0.2],8.0)
    # l,avg1,avg2,std1,std2 = testCausalityModified(timeseries[:,0],timeseries[:,1],2,8,range(20,320,40),25) 
    # ystr = "RMSE"
    # print(np.array(l))
    # print(np.array([avg1,avg2]))
    # avgarr = np.zeros((len(avg1),2))
    # avgarr[:,0] = avg1
    # avgarr[:,1] = avg2
    # SSRPlots.plots(np.array(l),avgarr,hold=0,show=1,stylestr=['b-','r-'],leglabels=['Mx from My','My from Mx'], legloc=0,xstr='length of time interval',ystr=ystr)   

    l,avg1,std1 = testDiffeomorphism(timeseries[:,0],timeseries[:,1],3,8,range(20,1001,200),25,Similarity.compareLocalDiams) 
    ystr='mean prod of diam ratios'
    print(np.array(l))
    print(np.array(avg1))
    SSRPlots.plots(np.array(l),np.array(avg1),hold=0,show=1,stylestr=['b-','r-'],leglabels=['Mx from My','My from Mx'], legloc=0,xstr='length of time interval',ystr=ystr)   
