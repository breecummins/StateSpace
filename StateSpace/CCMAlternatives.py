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
    Estimate manifold 1 (M1) from manifold 2 (and vice versa) using a 
    cross-mapping technique.
    Find the nearest points to each point in M2.
    Construct weights based on their distances to the point of
    interest. 
    Use the time indices of these points and the associated weights to 
    make a weighted-sum estimate of the contemporaneous points in M1.

    '''
    Mest1 = estManifold(M2,M1,wgtfunc)
    Mest2 = estManifold(M1,M2,wgtfunc)
    return Mest1, Mest2

def crossMapModified2(M1,M2,wgtfunc):
    '''
    Estimate time series 1 (M1[:,0]) from manifold 2 (and vice versa) by
    estimating the manifolds from each other and averaging the different
    time series.
    Find the nearest points to each point in M2.
    Construct weights based on their distances to the point of
    interest. 
    Use the time indices of these points and the associated weights to 
    make a weighted-sum estimate of the contemporaneous points in M1.
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
    Estimate a projection of M1 from manifold 2 and vice versa using a 
    cross-mapping technique.
    Find the nearest points to each point in M2.
    Construct weights based on their distances to the point of
    interest. 
    Use the time indices of these points and the associated weights to 
    make a weighted-sum estimate of the contemporaneous points in M1.
    Take a projection of the estimated manifold on dimension proj.

    '''
    Mest1 = estManifold(M2,M1,wgtfunc)
    Mest2 = estManifold(M1,M2,wgtfunc)
    est1 = Mest1[:,proj]
    est2 = Mest2[:,proj]
    return est1, est2

def testCausalityReconstruction(ts1,ts2,numlags,lagsize,listoflens,numiters,CM=crossMapModified1,wgtfunc=Weights.makeExpWeights,simMeasure=Similarity.RootMeanSquaredErrorManifold):
    '''
    Check for convergence to infer causality between ts1 and ts2.
    ts1 and ts2 must have the same length.
    numlags is the dimension of the embedding space for the reconstruction.
    Use time lags of size lagsize * dt to construct shadow manifolds. lagsize
    is an integer representing the index of the time lag.
    listoflens contains the lengths to use to show convergence 
    Example: range(100,10000,100)
    Each length will be run numiters times from different random starting 
    locations in the time series. numiters must be <= len(ts1) - max(listoflens).
    The estimated time series (or manifold) will be constructed using the weighting function 
    handle given by wgtfunc and the cross map function given by CM.
    CM = crossMapModified1 will estimate manifolds.
    CM = crossMapModified2 or 3 will estimate time series.
    The similarity between a time series (or a manifold) and its estimate will be given by
    simMeasure.
    For manifolds, simMeasure = Similarity.RootMeanSquaredErrorManifold or 
    Similarity.HausdorffDistance.
    For time series, simMeasure = Similarity.RootMeanSquaredErrorTS or Similarity.corrCoeffPearson.

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

def testDiffeomorphism(ts1,ts2,numlags,lagsize,listoflens,numiters,startind=0,simMeasure=Similarity.neighborDistance,N=None,poi=None):
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
    simMeasure, which can be Similarity.neighborDistance, Similarity.maxNeighborDistMean, 
    or Similarity.countingMeasure.
    N is an argument required by simMeasure that indicates the number of points
    in a neighborhood. The default depends on the dimension of the embedding space
    (numlags), so is constructed in the body of the function.

    '''
    if N == None:
        N = numlags+1
    L = len(ts1)
    if len(ts2) != L:
        raise(ValueError,"The lengths of the two time series must be the same.")
    listoflens.sort()
    lol = [l for l in listoflens if l < L-startind]
    avgcc1=[]
    stdcc1=[]
    avgcc2=[]
    stdcc2=[]
    for l in lol:
        startinds = random.sample(range(startind,L-l),numiters)
        cc1=[]
        cc2=[]
        for s in startinds:
            M1 = SSR.makeShadowManifold(ts1[s:s+l],numlags,lagsize)
            M2 = SSR.makeShadowManifold(ts2[s:s+l],numlags,lagsize)
            if not poi:
               c12,c21 = simMeasure(M1,M2,N)
            else:
               c12,c21 = simMeasure(M1,M2,N,poi)            
            cc1.append(c12)
            cc2.append(c21)
        avgcc1.append(np.mean(np.array(cc1)))
        stdcc1.append(np.std(np.array(cc1)))
        avgcc2.append(np.mean(np.array(cc2)))
        stdcc2.append(np.std(np.array(cc2)))
    return lol,avgcc1,stdcc1,avgcc2,stdcc2

def testDiffeomorphismSamePoints(ts1,ts2,numlags,lagsize,listoflens,startind=0,simMeasure=Similarity.maxNeighborDistArray,N=None,poi=None):
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
    simMeasure, which can be Similarity.maxNeighborDistArray only at the moment.
    N is an argument required by simMeasure that indicates the number of points
    in a neighborhood. The default depends on the dimension of the embedding space
    (numlags), so is constructed in the body of the function.
    poi is an optional argument of simMeasure that indicates the indices of points
    to track as the length of the tested time series increases. The default is set in 
    simMeasure and is all the points in the time series. Note that the poi indices reference  
    the truncated time series: ts[startind:startind+l] for l in listoflens. 
    So max(poi) is required to be < min(listoflens).

    '''
    if N == None:
        N = numlags+1
    L = len(ts1)
    if len(ts2) != L:
        raise(ValueError,"The lengths of the two time series must be the same.")
    listoflens.sort()
    lol = [l for l in listoflens if l < L-startind]
    sm12=[]
    sm21=[]
    for l in lol:
        M1 = SSR.makeShadowManifold(ts1[startind:startind+l],numlags,lagsize)
        M2 = SSR.makeShadowManifold(ts2[startind:startind+l],numlags,lagsize)
        s12,s21 = simMeasure(M1,M2,N,poi)
        sm12.append(s12)
        sm21.append(s21)
    return lol,sm12,sm21

def callme(ts1,ts2,numlags,lagsize,listoflens,numiters,simMeasure,N,ystr,leglabels,fname,note,startind=0,poi=None):
        l,avg1,std1, avg2, std2 = testDiffeomorphism(ts1,ts2,numlags,lagsize,listoflens,numiters,startind,simMeasure,N,poi) 
        cPickle.dump({'listoflens':l,'avg1':avg1,'avg2':avg2,'std1':std1,'std2':std2,'note':note,'numlags':numlags,'lagsize':lagsize,'timeseries':timeseries,'numneighbors':N,'startind':startind,'poi':poi},open(fname+'.pickle','w'))        
        print(np.array(l))
        print(np.array([avg1,avg2]))
        avgarr = np.zeros((len(avg1),2))
        avgarr[:,0] = avg1
        avgarr[:,1] = avg2
        SSRPlots.plots(np.array(l),avgarr,hold=0,show=0,stylestr=['b-','r-'],leglabels=leglabels, legloc=0,xstr='length of time interval',ystr=ystr,fname=fname+'.pdf')  

def callmesamepts(ts1,ts2,numlags,lagsize,listoflens,startind,simMeasure,N,poi,ystr,leglabels,fname,note):
        lol,sm12,sm21 = testDiffeomorphismSamePoints(ts1,ts2,numlags,lagsize,listoflens,startind,simMeasure,N,poi) 
        sm12 = np.array(sm12)
        sm21 = np.array(sm21)
        lol = np.array(lol)
        cPickle.dump({'listoflens':lol,'sm12':sm12,'sm21':sm21,'note':note,'numlags':numlags,'lagsize':lagsize,'timeseries':timeseries,'numneighbors':N,'startind':startind,'poi':poi},open(fname+'.pickle','w'))
        # print(lol)
        # print(sm12)
        # print(sm21)
        stylestr=['b-']*sm12.shape[1]
        SSRPlots.plots(lol,sm12,hold=0,show=0,stylestr=stylestr,leglabels=None, legloc=0,xstr='length of time interval',ystr=ystr,fname=fname+'_0.pdf',titlestr=leglabels[0])  
        stylestr=['r-']*sm12.shape[1]
        SSRPlots.plots(lol,sm21,hold=0,show=0,stylestr=stylestr,leglabels=None, legloc=0,xstr='length of time interval',ystr=ystr,fname=fname+'_1.pdf',titlestr=leglabels[1])  

def callmesameptsscalar(ts1,ts2,numlags,lagsize,listoflens,startind,simMeasure,N,poi,ystr,leglabels,fname,note):
        lol,sm12,sm21 = testDiffeomorphismSamePoints(ts1,ts2,numlags,lagsize,listoflens,startind,simMeasure,N,poi) 
        arr = np.array([sm12,sm21])
        lol = np.array(lol)
        cPickle.dump({'listoflens':lol,'sm12':sm12,'sm21':sm21,'note':note,'numlags':numlags,'lagsize':lagsize,'timeseries':timeseries,'numneighbors':N,'startind':startind,'poi':poi},open(fname+'.pickle','w'))
        print(lol)
        print(arr)
        SSRPlots.plots(lol,arr.transpose(),hold=0,show=0,stylestr=['b-','r-'],leglabels=leglabels, legloc=0,xstr='length of time interval',ystr=ystr,fname=fname+'.pdf')  

if __name__ == '__main__':
    import os
    import cPickle
    import StateSpaceReconstructionPlots as SSRPlots
    from LorenzEqns import solveLorenz
    timeseries = solveLorenz([1.0,0.5,0.5],161.0)
    numlags = 3
    lagsize = 8
    numiters = 25
    listoflens = range(1000,8001,1000)

    def LorenzCall(simMeasure,ystr,fname):
        leglabels1=[r'$f$: $M_x$ -> $M_z$',r'$f$: $M_z$ -> $M_x$']
        leglabels2=[r'$f$: $M_x$ -> $M_y$',r'$f$: $M_y$ -> $M_x$']
        note1 = "Make Mz from Mx in avg1 (does z -> x?), make Mx from Mz in avg2 (does x->z?), Lorenz eqns"
        note2 = "Make My from Mx in avg1 (does y -> x?), make Mx from My in avg2 (does x->y?), Lorenz eqns"
        fname1 = fname + 'xz'
        fname2 = fname + 'xy'

        for N in range(numlags+1,5*(numlags+1),numlags+1):
            print('xz, '+str(N)+ ' neighbors')
            ts1=timeseries[:,0]
            ts2=timeseries[:,2]
            callme(ts1,ts2,numlags,lagsize,listoflens,numiters,simMeasure,N,ystr+' '+str(N),leglabels1,fname1+str(N),note1)
            print('xy, '+str(N)+ ' neighbors')
            ts1=timeseries[:,0]
            ts2=timeseries[:,1]
            callme(ts1,ts2,numlags,lagsize,listoflens,numiters,simMeasure,N,ystr+' '+str(N),leglabels2,fname2+str(N),note2)

    # LorenzCall(Similarity.neighborDistance,'mean neighbor dist',os.path.expanduser('~/temp/Lorenzneighbordist'))
    # LorenzCall(Similarity.countingMeasure,'mean counting measure',os.path.expanduser('~/temp/Lorenzcountingmeasure'))

    listoflens = range(500,16001,500)
    startind=2000
    # np.random.seed(43)
    # poi = sorted(list(set([int(r) for r in (min(listoflens)-2*numlags-1)*np.random.rand(25)])))#[0,27,100,157,226,250,321,366]#,512,601]
    # print(poi)
    poi = range(0,min(listoflens)-1 - (numlags-1)*lagsize,7)
    def LorenzCallSamePts(simMeasure,ystr,fname,whichcall=callmesamepts):
        leglabels1=[r'$f$: $M_x$ -> $M_z$',r'$f$: $M_z$ -> $M_x$']
        leglabels2=[r'$f$: $M_x$ -> $M_y$',r'$f$: $M_y$ -> $M_x$']
        note1 = "Make Mz from Mx in avg1 (does z -> x?), make Mx from Mz in avg2 (does x->z?), Lorenz eqns"
        note2 = "Make My from Mx in avg1 (does y -> x?), make Mx from My in avg2 (does x->y?), Lorenz eqns"
        fname1 = fname + 'xz'
        fname2 = fname + 'xy'

        for N in range(numlags+1,5*(numlags+1),numlags+1):
            print('xz, '+str(N)+ ' neighbors')
            ts1=timeseries[:,0]
            ts2=timeseries[:,2]
            whichcall(ts1,ts2,numlags,lagsize,listoflens,startind,simMeasure,N,poi,ystr+' '+str(N),leglabels1,fname1+str(N),note1)
            print('xy, '+str(N)+ ' neighbors')
            ts1=timeseries[:,0]
            ts2=timeseries[:,1]
            whichcall(ts1,ts2,numlags,lagsize,listoflens,startind,simMeasure,N,poi,ystr+' '+str(N),leglabels2,fname2+str(N),note2)


    # LorenzCallSamePts(Similarity.maxNeighborDist,'max neighbor dist',os.path.expanduser('~/temp/LorenzMaxNeighborDist'))
    LorenzCallSamePts(Similarity.maxNeighborDistMean,'mean max neighbor dist',os.path.expanduser('~/temp/LorenzMaxNeighborDistMeanEvery7_'),callmesameptsscalar)
