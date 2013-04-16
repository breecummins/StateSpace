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
            dists,inds = Weights.findClosest(poi,M,numlags+1)
            w = wgtfunc(np.array(dists))
            est[k] = (w*ts[list(inds)]).sum()
        return est
    est1 = estSeries(M2,M1[:,0])
    est2 = estSeries(M1,M2[:,0])
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
            shift = (numlags-1)*lagsize 
            cc1.append(Similarity.corrCoeffPearson(est1,ts1[s+shift:s+l]))
            cc2.append(Similarity.corrCoeffPearson(est2,ts2[s+shift:s+l]))
        avgcc1.append(np.mean(np.array(cc1)))
        avgcc2.append(np.mean(np.array(cc2)))
        stdcc1.append(np.std(np.array(cc1)))
        stdcc2.append(np.std(np.array(cc2)))
    return lol,avgcc1,avgcc2,stdcc1,stdcc2
    

if __name__ == '__main__':
    import StateSpaceReconstructionPlots as SSRPlots
    # from LorenzEqns import solveLorenz
    # timeseries = solveLorenz([1.0,0.5,0.5],80.0)
    # l,avg1,avg2,std1,std2 = testCausality(timeseries[:4001,0],timeseries[:4001,1],2,8,range(20,2000,200),25) 
    # from differenceEqns import solve2Species
    # timeseries = solve2Species([0.4,0.2],8.0)
    # l,avg1,avg2,std1,std2 = testCausality(timeseries[:,0],timeseries[:,1],2,8,range(20,320,40),25) 
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
        l,avg1,avg2,std1,std2 = testCausality(timeseries[:,k],timeseries[:,-1],9,16,range(500,3100,500),25)
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
            hold += 1
        if k == 2:
            show += 1
            SSRPlots.plots(np.array(l),avgarr[:,1],hold=hold,show=show,stylestr=[styles[-1]],leglabels=['w from M'+names[k]], legloc=legloc,xstr='length of time interval',ystr='mean corr coeff')   
    SSRPlots.plots(np.array(l),xw,hold=0,show=0,stylestr=['b-'],leglabels=['w from Mx'], legloc=legloc,xstr='length of time interval',ystr='mean corr coeff')   
    SSRPlots.plots(np.array(l),yw,hold=1,show=1,stylestr=['r-'],leglabels=['w from My'], legloc=legloc,xstr='length of time interval',ystr='mean corr coeff')   
           
