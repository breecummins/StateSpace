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
    avgcc2=[]
    stdcc2=[]
    for l in lol:
        startinds = random.sample(range(L-l),numiters)
        cc1=[]
        cc2=[]
        for s in startinds:
            M1 = SSR.makeShadowManifold(ts1[s:s+l],numlags,lagsize)
            M2 = SSR.makeShadowManifold(ts2[s:s+l],numlags,lagsize)
            cc1.append(simMeasure(M1,M2))
            cc2.append(simMeasure(M2,M1))
        avgcc1.append(np.mean(np.array(cc1)))
        stdcc1.append(np.std(np.array(cc1)))
        avgcc2.append(np.mean(np.array(cc2)))
        stdcc2.append(np.std(np.array(cc2)))
    return lol,avgcc1,stdcc1,avgcc2,stdcc2

def testDiffeomorphism2(ts1,ts2,numlags,lagsize,listoflens,numiters,simMeasure=Similarity.neighborDistance,N=None):
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
    N is an extra argument required by simMeasure and may vary between functions.

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
            c12,c21 = simMeasure(M1,M2,N)
            cc1.append(c12)
            cc2.append(c21)
        avgcc1.append(np.mean(np.array(cc1)))
        stdcc1.append(np.std(np.array(cc1)))
        avgcc2.append(np.mean(np.array(cc2)))
        stdcc2.append(np.std(np.array(cc2)))
    return lol,avgcc1,stdcc1,avgcc2,stdcc2

def callme(ts1,ts2,numlags,lagsize,listoflens,numiters,simMeasure,N,ystr,leglabels,fname,note):
        l,avg1,std1, avg2, std2 = testDiffeomorphism2(ts1,ts2,numlags,lagsize,listoflens,numiters,simMeasure,N) 
        cPickle.dump({'listoflens':l,'avg1':avg1,'avg2':avg2,'std1':std1,'std2':std2,'note':note,'numlags':numlags,'lagsize':lagsize,'timeseries':timeseries,'numneighbors':N},open(fname+'.pickle','w'))        
        print(np.array(l))
        print(np.array([avg1,avg2]))
        avgarr = np.zeros((len(avg1),2))
        avgarr[:,0] = avg1
        avgarr[:,1] = avg2
        SSRPlots.plots(np.array(l),avgarr,hold=0,show=0,stylestr=['b-','r-'],leglabels=leglabels, legloc=0,xstr='length of time interval',ystr=ystr,fname=fname+'.pdf')   


if __name__ == '__main__':
    import os
    import cPickle
    import StateSpaceReconstructionPlots as SSRPlots
    from LorenzEqns import solveLorenz
    timeseries = solveLorenz([1.0,0.5,0.5],80.0)
    numlags = 3
    lagsize = 8
    numiters = 25
    listoflens = range(200,2001,200)
    simMeasure=Similarity.neighborDistance
    leglabels1=['\phi: M_x \to M_z','\phi: M_z \to M_x']
    leglabels2=['\phi: M_x \to M_y','\phi: M_y \to M_x']
    note1 = "Make Mz from Mx in avg1 (z -> x?), make Mx from Mz in avg2 (x->z?), Lorenz eqns"
    note2 = "Make My from Mx in avg1 (y -> x?), make Mx from My in avg2 (x->y?), Lorenz eqns"
 
    for N in range(numlags+1,5*(numlags+1),numlags+1):
        ystr='mean ' + str(N) + ' neighbor dist'
        #xz
        ts1=timeseries[:,0]
        ts2=timeseries[:,2]
        fname= os.path.expanduser('~/temp/Lorenzxz'+str(N))
        callme(ts1,ts2,numlags,lagsize,listoflens,numiters,simMeasure,N,ystr,leglabels1,fname,note1)
        #xy
        ts1=timeseries[:,0]
        ts2=timeseries[:,1]
        fname= os.path.expanduser('~/temp/Lorenzxy'+str(N))
        callme(ts1,ts2,numlags,lagsize,listoflens,numiters,simMeasure,N,ystr,leglabels2,fname,note2)
 
