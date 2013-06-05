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
    est1 = Mest1[:,0]
    est2 = Mest2[:,0]

    '''
    M1 = SSR.makeShadowManifold(ts1,numlags,lagsize)
    M2 = SSR.makeShadowManifold(ts2,numlags,lagsize)
    def estSeries(M,ts):
        est=np.zeros(ts.shape)
        for k in range(M.shape[0]):
            poi = M[k,:]
            dists,inds = Similarity.findClosestInclusive(poi,M,numlags+1)
            w = wgtfunc(np.array(dists))
            est[k] = (w*ts[list(inds)]).sum()
        return est
    est1 = estSeries(M2,M1[:,0])
    est2 = estSeries(M1,M2[:,0])
    return est1, est2

def testCausality(ts1,ts2,startinds,l,numlags,lagsize,wgtfunc=Weights.makeExpWeights,simMeasure=[Similarity.corrCoeffPearson]):
    '''
    Must be called by causalityWrapper.

    startinds contains a number of starting indices to test subintervals of 
    length l of time series ts1 and ts2.
    numlags is the dimension of the embedding space for the reconstruction.
    Use time lags of size lagsize * dt to construct shadow manifolds. lagsize
    is an integer representing the index of the time lag.
    This function constructs the estimated time series using the weighting function 
    handle given by wgtfunc, which can be any function in the Weights module.
    The similarity between the time series and its estimate will be assessed by 
    each function in simMeasure, and may include Similarity.corrCoeffPearson, 
    Similarity.RootMeanSquaredErrorTS, and Similarity.MeanAbsoluteErrorTS.

    '''
    try:
        sL = len(simMeasure)
    except:
        simMeasure=[simMeasure]
        sL = 1
    cc1=[]
    cc2=[]
    for s in startinds:
        est1,est2 = crossMap(ts1[s:s+l],ts2[s:s+l],numlags,lagsize,wgtfunc)
        #correct for the time points lost in shadow manifold construction
        shift = (numlags-1)*lagsize 
        for simfunc in simMeasure:
            cc1.append(simfunc(est1,ts1[s+shift:s+l]))
            cc2.append(simfunc(est2,ts2[s+shift:s+l]))
    return cc1,cc2,sL
    
def crossMapManifold(M1,M2,numlags,lagsize,wgtfunc):
    '''
    Estimate timeseries 1 from timeseries 2 and vice versa using  
    Sugihara's cross-mapping technique between shadow manifolds.
    In this version, the shadow manifolds are supplied as arguments.
    Find the nearest points to each point in M1.
    Construct weights based on their distances to the point of
    interest. 
    Use the time indices of these points and the associated weights to 
    make a weighted-sum estimate of the contemporaneous points in the 
    second time series. 
    Repeat starting with points in M2 and casting into time series 1.

    Could also be calculated by estimating the manifold and taking a 
    projection:
    Mest1 = estManifold(M2,M1)
    Mest2 = estManifold(M1,M2)
    est1 = Mest1[:,0]
    est2 = Mest2[:,0]


    '''
    def estSeries(M,ts):
        est=np.zeros(ts.shape)
        for k in range(M.shape[0]):
            poi = M[k,:]
            dists,inds = Similarity.findClosestInclusive(poi,M,numlags+1)
            w = wgtfunc(np.array(dists))
            est[k] = (w*ts[list(inds)]).sum()
        return est
    est1 = estSeries(M2,M1[:,0])
    est2 = estSeries(M1,M2[:,0])
    return est1, est2

def testCausalityReconstruction(ts1,ts2,startinds,l,numlags,lagsize,wgtfunc=Weights.makeExpWeights,simMeasure=[Similarity.RootMeanSquaredErrorManifold]):
    '''
    Must be called by causalityWrapper.
    
    startinds contains a number of starting indices to test subintervals of 
    length l of time series ts1 and ts2.
    numlags is the dimension of the embedding space for the reconstruction.
    Use time lags of size lagsize * dt to construct shadow manifolds. lagsize
    is an integer representing the index of the time lag.
    This function constructs the estimated time series directly from the 
    manifolds using the weighting function handle given by wgtfunc, which can be 
    any function in the Weights module.
    Instead of comparing a time series to its estimate, we compare a shadow manifold
    to an estimated shadow manifold constructed from the time series estimates.
    The similarity between the manifold and its estimate will be assessed by 
    each function in simMeasure, and may include Similarity.RootMeanSquaredErrorManifold, 
    Similarity.MeanErrorManifold, and Similarity.HausdorffDistance.

    '''
    try:
        sL = len(simMeasure)
    except:
        simMeasure=[simMeasure]
        sL = 1
    cc1=[]
    cc2=[]
    for s in startinds:
        M1orig=SSR.makeShadowManifold(ts1[s:s+l],numlags,lagsize)
        M2orig=SSR.makeShadowManifold(ts2[s:s+l],numlags,lagsize)
        est1,est2=crossMapManifold(M1orig,M2orig,numlags,lagsize,wgtfunc)
        M1est=SSR.makeShadowManifold(est1,numlags,lagsize)
        M2est=SSR.makeShadowManifold(est2,numlags,lagsize)
        #correct for the time points lost in shadow manifold construction
        shift = (numlags-1)*lagsize 
        for simfunc in simMeasure:
            cc1.append(simfunc(M1est,M1orig[shift:,:]))
            cc2.append(simfunc(M2est,M2orig[shift:,:]))
    return cc1, cc2, sL
    
def causalityWrapper(ts1,ts2,numlags,lagsize,listoflens,numiters,allstartinds=None,growtraj=0,causalitytester=testCausality,morefunctions=None):
    '''
    Check for convergence to infer causality between the time series ts1 and ts2, 
    where len(ts1) == len(ts2).
    
    numlags is the dimension of the embedding space for the reconstruction.
    Time lags of size lagsize * dt (dt = uniform time step in series ts1 and ts2) 
    are used to construct shadow manifolds. lagsize is an integer representing the 
    index of the time lag.
    
    listoflens contains the lengths to use to show convergence.
    Example: range(100,10000,100)
    Each length will be run numiters times from different random starting 
    locations in the time series, where numiters must be <= len(ts1) - max(listoflens).
    The optional argument allstartinds allows specific starting locations to be
    specified instead of randomly generating the starting indices.
    The optional argument growtraj=0 (default) or 1 says whether or not to keep the same 
    starting indices across all lengths in listoflens, so that the examples are growing 
    in the same locations instead of at random locations.
    
    The details of the specific causality test is in the function causalitytester.
    The optional dictionary morefunctions can contain keyword entries for 'CM', 'wgtfunc', 
    and 'simMeasure'. Default CM, wgtfunc, and simMeasure methods are supplied by each
    causalitytester function, and allowable functions vary between causalitytesters. 
    See the notes and defaults for each function.

    causalitytester may be testCausality (Sugihara original), 
    testCausalityReconstruction (Sugihara with reconstruction), or CCMAlternatives.
    testCausalityReconstruction (our reconstruction methods). All three functions
    may specify wgtfunc and simMeasure, but only the third can specify 
    CM = CCMAlternatives.crossMapModified*, where * = 1, 2, or 3.
    
    The estimated time series or manifold will be constructed using the weighting function 
    handle given by wgtfunc. wgtfunc may be any function in the module Weights.
    The similarity between the time series or manifold and its estimate will be assessed by 
    each function in simMeasure. See the notes for each causalitytester for allowable
    simMeasure functions in the Similarity module.
    
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
    if allstartinds == None:
        allstartinds = []
        for l in lol:
            allstartinds.append(random.sample(range(L-l),numiters))
    if morefunctions == None:
        morefunctions = {}
    for k,l in enumerate(lol):
        if growtraj:
            startinds = allstartinds[-1]
        else:
            startinds = allstartinds[k]
        cc1, cc2, sL = causalitytester(ts1,ts2,startinds,l,numlags,lagsize,**morefunctions)
        avgcc1.append([np.mean(cc1[_k::sL]) for _k in range(sL)])
        avgcc2.append([np.mean(cc2[_k::sL]) for _k in range(sL)])
        stdcc1.append([np.std(cc1[_k::sL]) for _k in range(sL)])
        stdcc2.append([np.std(cc2[_k::sL]) for _k in range(sL)])
    return lol,avgcc1,avgcc2,stdcc1,stdcc2
    
if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    # from LorenzEqns import solveLorenz
    # timeseries = solveLorenz([1.0,0.5,0.5],80.0)
    # l,avg1,avg2,std1,std2 = causalityWrapper(timeseries[:4001,0],timeseries[:4001,1],2,8,range(20,2000,200),25,causalitytester=testCausality) 
    # from differenceEqns import solve2Species
    # timeseries = solve2Species([0.4,0.2],8.0)
    # l,avg1,avg2,std1,std2 = causalityWrapper(timeseries[:,0],timeseries[:,1],2,8,range(20,320,40),25,causalitytester=testCausality) 
    # print(np.array(l))
    # print(np.array([avg1,avg2]))
    # avgarr = np.zeros((len(avg1),2))
    # avgarr[:,0] = avg1
    # avgarr[:,1] = avg2
    # SSRPlots.plots(np.array(l),avgarr,hold=0,show=1,stylestr=['b-','r-'],leglabels=['x from My','y from Mx'], legloc=0,xstr='length of time interval',ystr='mean corr coeff')   


    from DoublePendulum import solvePendulum
    timeseries = solvePendulum([1.0,2.0,3.0,2.0],300.0)
    names = ['x','y','z']
    styles = ['b-','r-','g-','k-']
    hold = 0
    show = 0
    for k in range(3):
        l,avg1,avg2,std1,std2 = causalityWrapper(timeseries[200:,k],timeseries[200:,-1],4,8,range(500,3000,500),25,causalitytester=testCausality)
        print(np.array(l))
        print(np.array([avg1,avg2]))
        avgarr = np.zeros((len(avg1),2))
        avgarr[:,0] = avg1
        avgarr[:,1] = avg2
        if k == 0: 
            xw = avgarr[:,1]
        elif k==1:
            yw = avgarr[:,1]
        legloc=0
        SSRPlots.plots(np.array(l),avgarr[:,0],hold=hold,show=show,stylestr=[styles[k]],leglabels=[names[k] + ' from Mw'], legloc=legloc,xstr='length of time interval',ystr='mean corr coeff')   
        if not hold:
            hold = 1
        if k == 2:
            show = 1
            SSRPlots.plots(np.array(l),avgarr[:,1],hold=hold,show=show,stylestr=[styles[-1]],leglabels=['w from M'+names[k]], legloc=legloc,xstr='length of time interval',ystr='mean corr coeff')   
    SSRPlots.plots(np.array(l),xw,hold=0,show=0,stylestr=['b-'],leglabels=['w from Mx'], legloc=legloc,xstr='length of time interval',ystr='mean corr coeff')   
    SSRPlots.plots(np.array(l),yw,hold=1,show=1,stylestr=['r-'],leglabels=['w from My'], legloc=legloc,xstr='length of time interval',ystr='mean corr coeff')   
           
