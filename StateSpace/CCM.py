#import python modules
import numpy as np
import random
#import home-rolled modules
import StateSpaceReconstruction as SSR
import Weights
import Similarity

def crossMap(ts1,ts2,numlags,lagsize,wgtfunc):
    '''
    Estimate timeseries 1 (ts1) from timeseries 2 and vice versa using  
    Sugihara's cross-mapping technique between shadow manifolds.
    Construct a shadow manifold M from one time series.
    Find the nearest points to each point in M.
    Construct weights based on their distances to the point of
    interest. 
    Use the time indices of these points and the associated weights to 
    make a weighted-sum estimate of the contemporaneous points in the 
    other time series.

    Could also be calculated by estimating the manifold and taking a 
    projection:
    Mest1 = estManifold(M2,M1)
    Mest2 = estManifold(M1,M2)
    est1 = Mest1[:,-1]
    est2 = Mest2[:,-1]


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

def testCausality(ts1,ts2,numlags,lagsize,listoflens,numiters,wgtfunc=Weights.makeExpWeights):
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
            cc1.append(Similarity.corrCoeffPearson(est1,ts1[s1:s1+l1]))
            cc2.append(Similarity.corrCoeffPearson(est2,ts2[s1:s1+l1]))
        avgcc1.append(np.mean(np.array(cc1)))
        avgcc2.append(np.mean(np.array(cc2)))
        stdcc1.append(np.std(np.array(cc1)))
        stdcc2.append(np.std(np.array(cc2)))
    return lol,avgcc1,avgcc2,stdcc1,stdcc2
    
def estManifold(Mx,My,wgtfunc):
    '''
    Estimate My from Mx.

    '''
    Mest=np.zeros(My.shape)
    for k in range(Mx.shape[0]):
        poi = Mx[k,:]
        dists,inds = findClosest(poi,Mx,Mx.shape[1]+1)
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
            cc1.append(Similarity.L2error(Mest1,M1))
            cc2.append(Similarity.L2error(Mest2,M2))
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
    numlags=3
    lagsize=8 
    endind=2000 
    M1=np.array(list(SSR.makeShadowManifold(timeseries[:endind,0],numlags,lagsize)))
    M2=np.array(list(SSR.makeShadowManifold(timeseries[:endind,1],numlags,lagsize)))
    est1,est2=crossMap(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize,Weights.makeExpWeights)
    M1Sug=np.array(list(SSR.makeShadowManifold(est1,numlags,lagsize)))
    M2Sug=np.array(list(SSR.makeShadowManifold(est2,numlags,lagsize)))
    corr = (numlags-1)*lagsize
    l = M1Sug.shape[0]
    M1SugErr = Similarity.L2error(M1[corr:corr+l,:],M1Sug)
    M2SugErr = Similarity.L2error(M2[corr:corr+l,:],M2Sug)
    print("Sugihara method L2 error:")
    print("    L2 error between Mx and estimated Mx is " + str(M1SugErr))
    print("    L2 error between My and estimated My is " + str(M2SugErr))
    M1us1,M2us1=crossMapModified1(M1,M2,Weights.makeExpWeights)
    print("Our method 1 L2 error:")
    print("    L2 error between Mx and estimated Mx is " + str(Similarity.L2error(M1,M1us1)))
    print("    L2 error between My and estimated My is " + str(Similarity.L2error(M2,M2us1)))    
    est1,est2=crossMapModified2(M1,M2,Weights.makeExpWeights)
    M1us2=np.array(list(SSR.makeShadowManifold(est1,numlags,lagsize)))
    M2us2=np.array(list(SSR.makeShadowManifold(est2,numlags,lagsize)))
    corr = (numlags-1)*lagsize
    l = M1us2.shape[0]
    M1err2 = Similarity.L2error(M1[corr:corr+l,:],M1us2)
    M2err2 = Similarity.L2error(M2[corr:corr+l,:],M2us2)
    print("Our method 2 L2 error:")
    print("    L2 error between Mx and estimated Mx is " + str(M1err2))
    print("    L2 error between My and estimated My is " + str(M2err2))   
    proj = numlags - 3 
    est1,est2=crossMapModified3(M1,M2,proj,Weights.makeExpWeights)
    M1us3=np.array(list(SSR.makeShadowManifold(est1,numlags,lagsize)))
    M2us3=np.array(list(SSR.makeShadowManifold(est2,numlags,lagsize)))
    corr = (numlags-1)*lagsize
    l = M1us3.shape[0]
    M1err3 = Similarity.L2error(M1[corr-proj:corr+l-proj,:],M1us3)
    M2err3 = Similarity.L2error(M2[corr-proj:corr+l-proj,:],M2us3)
    print("Our method 3 L2 error:")
    print("    L2 error between Mx and estimated Mx is " + str(M1err3))
    print("    L2 error between My and estimated My is " + str(M2err3))    


    # SSRPlots.plotEstShadowManifoldSugihara(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize)        
    # SSRPlots.plotEstShadowManifoldUs1(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize)        
    # SSRPlots.plotEstShadowManifoldUs2(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize)        
    SSRPlots.plotEstShadowManifoldUs3(timeseries[:endind,0],timeseries[:endind,1],numlags,lagsize,proj)        




     