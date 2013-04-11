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
    
def L2error(M1,M2):
    return np.sqrt(((M1-M2)**2).sum(1)).sum(0) / M1.shape[0]

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
    def estManifold(Mx,My):
        est=np.zeros(My.shape)
        for k in range(Mx.shape[0]):
            poi = Mx[k,:]
            dists,inds = findClosest(poi,Mx,Mx.shape[1]+1)
            w = wgtfunc(np.array(dists))
            pts = [My[j,:] for j in inds]
            est[k,:] = np.array([w[j]*pts[j] for j in range(len(w))]).sum(0)
        return est
    Mest1 = estManifold(M2,M1)
    Mest2 = estManifold(M1,M2)
    return Mest1, Mest2

def testCausalityModified1(ts1,ts2,numlags,lagsize,listoflens,numiters,wgtfunc=makeExpWeights):
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
            M1 = np.array(list(SSR.makeShadowManifold(ts1[s:s+l],numlags,lagsize)))
            M2 = np.array(list(SSR.makeShadowManifold(ts2[s:s+l],numlags,lagsize)))
            Mest1,Mest2 = crossMapModified(M1,M2,wgtfunc)
            cc1.append(L2error(Mest1,M1))
            cc2.append(L2error(Mest2,M2))
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

    '''
    def estManifold(Mx,My):
        est=np.zeros(My.shape)
        for k in range(Mx.shape[0]):
            poi = Mx[k,:]
            dists,inds = findClosest(poi,Mx,Mx.shape[1]+1)
            w = wgtfunc(np.array(dists))
            pts = [My[j,:] for j in inds]
            est[k,:] = np.array([w[j]*pts[j] for j in range(len(w))]).sum(0)
        return est
    Mest1 = estManifold(M2,M1)
    Mest2 = estManifold(M1,M2)
    #FIXME
    n = M1.shape[1]
    est1 = Mest1[(n-1):-(n-1),0]+Mest1[(n-2):-n,1]+Mest1[:-(n+1),n-1]) / n
    est1[0] = Mest1[0,0]
    est1[1] = (Mest1[0,1]+Mest1[1,0])/2.
    est1[2:-2] = 
    est1[-2] = (Mest1[-2,-1]+Mest1[-1,1])/2.
    est1[-1] = Mest1[-1,-1]
    est2 = np.zeros(Mest2.shape[0])
    est2[0] = Mest2[0,0]
    est2[1] = (Mest2[0,1]+Mest2[1,0])/2.
    est2[2:-2] = (Mest2[2:-2,0]+Mest2[1:-3,1]+Mest2[:-4,2]) / 3.
    est2[-2] = (Mest2[-2,-1]+Mest2[-1,1])/2.
    est2[-1] = Mest2[-1,-1]
    return est1, est2

def testCausalityModified1(ts1,ts2,numlags,lagsize,listoflens,numiters,wgtfunc=makeExpWeights):
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
            M1 = np.array(list(SSR.makeShadowManifold(ts1[s:s+l],numlags,lagsize)))
            M2 = np.array(list(SSR.makeShadowManifold(ts2[s:s+l],numlags,lagsize)))
            est1,est2 = crossMapModified(M1,M2,wgtfunc)
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
    # l,avg1,avg2,std1,std2 = testCausalityModified(timeseries[:,0],timeseries[:,1],2,8,range(100,2000,100),25) 
    # print(np.array(l))
    # print(np.array([avg1,avg2]))
    # avgarr = np.zeros((len(avg1),2))
    # avgarr[:,0] = avg1
    # avgarr[:,1] = avg2
    # SSRPlots.plots(np.array(l),avgarr,stylestr=['b-','r-'],leglabels=['x from My','y from Mx'],xstr='length of time interval',ystr='mean L2 error')   
    numlags=2
    lagsize=8   
    endind=2000 
    M1=np.array(list(SSR.makeShadowManifold(timeseries[:endind,0],numlags,lagsize)))
    M2=np.array(list(SSR.makeShadowManifold(timeseries[:endind,1],numlags,lagsize)))
    est1,est2=crossMap(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize,makeExpWeights)
    M1Sug=np.array(list(SSR.makeShadowManifold(est1,numlags,lagsize)))
    M2Sug=np.array(list(SSR.makeShadowManifold(est2,numlags,lagsize)))
    corr = (numlags-1)*lagsize
    l = M1Sug.shape[0]
    M1SugErr = L2error(M1[corr:corr+l,:],M1Sug)
    M2SugErr = L2error(M2[corr:corr+l,:],M2Sug)
    print("Sugihara method L2 error:")
    print("    L2 error between Mx and estimated Mx is " + str(M1SugErr))
    print("    L2 error between My and estimated My is " + str(M2SugErr))
    M1us1,M2us1=crossMapModified1(M1,M2,makeExpWeights)
    print("Our method 1 L2 error:")
    print("    L2 error between Mx and estimated Mx is " + str(L2error(M1,M1us1)))
    print("    L2 error between My and estimated My is " + str(L2error(M2,M2us1)))    
    est1,est2=crossMapModified2(M1,M2,makeExpWeights)
    M1us2=np.array(list(SSR.makeShadowManifold(est1,numlags,lagsize)))
    M2us2=np.array(list(SSR.makeShadowManifold(est2,numlags,lagsize)))
    corr = (numlags-1)*lagsize
    l = M1us2.shape[0]
    M1us2 = L2error(M1[corr:corr+l,:],M1us2)
    M2us2 = L2error(M2[corr:corr+l,:],M2us2)
    print("Our method 2 L2 error:")
    print("    L2 error between Mx and estimated Mx is " + str(L2error(M1,M1us2)))
    print("    L2 error between My and estimated My is " + str(L2error(M2,M2us2)))    


    SSRPlots.plotEstShadowManifoldSugihara(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize)        
    SSRPlots.plotEstShadowManifoldUs1(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize)        
    SSRPlots.plotEstShadowManifoldUs2(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize)        




     