import numpy as np
from scipy.misc import comb
import random
import sys
import StateSpaceReconstruction as SSR

def getBinomialMax(n,p):
    '''
    Return the maximum value of the binomial distribution 
    parameterized by n observations with p probability of 
    success.

    '''
    maxloc = np.floor( (n+1)*p )
    return comb(n,maxloc)*(p**maxloc)*((1-p)**(n-maxloc))

def getContinuityConfidence(neps,ndel,numpts):
    '''
    Calculate a confidence that points are _not_ randomly distributed
    over the manifolds M1 and M2. 

    We observed that the ndel nearest points to x mapped to within 
    epsilon of the image point f(x). We also observed that a total of 
    neps points were within epsilon of f(x). Under the null hypothesis 
    that the points in M1 and M2 are randomly distributed, the 
    probability of our observation of ndel successes is given by
    (neps / numpts)**ndel, where numpts is the number of points in M1
    (and likewise in M2). If this probability is small compared to 1, and 
    also small compared to the maximum probability (p_max) in the binomial 
    distribution B(ndel, neps/numpts), then we are confident that the 
    points are not randomly distributed. We define the confidence level
    to be 1 - (neps / numpts)**ndel / p_max.

    Since we made our observations to be consistent with the definition
    of continuity, we say we are relatively confident (output near 1) or 
    not (output near 0) of the presence of a continuous function between
    M1 and M2. 

    '''
    p = float(neps) / numpts
    pmax = getBinomialMax(ndel,p)
    return 1 - (p**ndel)/pmax

def countPtsWithinEps(dists,eps):
    '''
    Count the distances less than eps and return the count less 1 to 
    remove the zero distance to the point itself.

    '''
    return (dists < eps).sum() - 1 

def countDeltaPtsMappedToEps(dists1,dists2,delta,eps):
    '''
    Find all points within delta of M1[ind,:] using cached distances, 
    then check to see if their images fall within eps of M2[ind,:], also
    using cached distances. If all delta pts fall within eps, we have a
    success and return the number of points within delta less the point 
    itself. If we fail, then return False (delta is too big). 

    '''
    deltainds = np.nonzero(dists1 < delta)[0]
    for ind in deltainds:
        if dists2[ind] >= eps:
            return False
    return len(deltainds)-1

def continuityTest(dists1,dists2,ptinds,eps,startdelta):
    '''
    Do Pecora continuity test on the reconstructions M1 and M2 using 
    the points with indices ptinds and continuity parameter eps. dists1
    and dists2 contain the distances to the points of interest (ptinds), 
    which are all that is needed from M1 and M2. 
    Guess delta values beginning with startdelta, which ideally has the
    relation startdelta/|M1| approximates eps/|M2|.

    '''
    contstat = np.zeros(len(ptinds))
    # record individual probabilities - not directly needed for the method, needed for interpretation
    probs = np.zeros(len(ptinds))
    for k in range(len(ptinds)):
        neps = countPtsWithinEps(dists2[k],eps)
        probs[k] = float(neps) / len(dists1[k])
        # print("Probability of 1 correct mapping: {0}".format(np.round(float(neps) / len(dists1[k]),3)))
        if neps > 0: # if eps big enough, continue; else leave 0 in place
            delta = 2.0*startdelta
            ndel = False
            while ndel is False:
                delta = delta*0.5
                ndel = countDeltaPtsMappedToEps(dists1[k],dists2[k],delta,eps) 
            if ndel: #ndel can be 0, in which case we leave 0 confidence in place
                contstat[k] = getContinuityConfidence(neps,ndel,len(dists1[k]))
                while np.isnan(contstat[k]):
                    # Need handling of overflow for success under very improbable conditions. 
                    # Check lower numbers. Confidence may be underestimated. 
                    ndel = ndel/2
                    contstat[k] = getContinuityConfidence(neps,ndel,len(dists1[k]))
            # print("Confidence: {0}".format(contstat[k]))
    return np.mean(contstat), np.mean(probs)

def cacheDistances(M,ptinds):
    dists = []
    for ind in ptinds:
        dists.append(np.sqrt(((M - M[ind,:])**2).sum(axis=1)))
    return dists
 
def chooseEpsilons(M,epsprops):
    '''
    Estimate the standard deviation of the manifold M (sensu Pecora)
    and take fractions of it for different epsilons. The proportions
    to take are given in epsprops, a numpy array.

    '''
    meanM = np.mean(M,axis=0)
    dists = np.sqrt(((M - meanM)**2).sum(axis=1))
    stdM = np.std(dists)
    return stdM*epsprops

def makeReconstructions(ts1,ts2,numlags,lag1,lag2):
    M1 = SSR.makeShadowManifold(ts1,numlags,lag1)
    M2 = SSR.makeShadowManifold(ts2,numlags,lag2)
    # if the lags are different, remove noncontemporaneous points from the front of the M with more points (constructed with the smaller lag)
    if M2.shape[0] > M1.shape[0]:
        M2 = M2[(lag1-lag2)*(numlags-1):,:]
    elif M1.shape[0] > M2.shape[0]:
        M1 = M1[(lag2-lag1)*(numlags-1):,:]
    return M1,M2

def convergenceWithContinuityTestMultipleLags(ts1,ts2,numlags,lags,tsprops=np.arange(0.4,1.1,0.2),epsprops=np.array([0.005,0.01,0.02,0.05,0.1,0.2])):
    '''
    Perform Pecora's continuity tests between reconstructions M1 and M2 built from
    time series ts1 and ts2 with parameters numlags (embedding dimension) and lags 
    (lag sizes). The continuity test will be performed in the forward and reverse 
    directions for increasing proportions of the time series (proportions given in 
    tsprops) and for a collection of epsilons. The epsilons are given as proportions 
    (in epsprops) of the standard deviation of each M1 and M2 for each length.

    ts1 and ts2 are 1D numpy arrays, numlags is an integer, lags is a list of lists, 
    and tsprops and epsprops are numpy arrays with values between 0 and 1 with default 
    values in the function argument list. lags will be a list containing n 2-element 
    lists where n = len(tsprops). Each sublist is of the form [lag1N, lag2N], where
    lag1N is the appropriate lag for ts1[:N] and likewise for lag2N. It is an error
    if the lengths of lags and tsprops do not match.

    The output is two numpy arrays, one containing the results of the forward 
    continuity test and the other of the inverse test. The rows correspond to different
    time series lengths, and the columns to different epsilons.

    If we have high confidence for some small epsilons and that confidence increases
    with increasing time series length, then we conclude that we have circumstantial
    evidence for a continuous function. 

    '''
    if len(lags) != len(tsprops):
        raise ValueError("The lengths of lags and tsprops must be equal.")
    Mlens = (np.round(len(ts1)*tsprops)).astype(int)
    forwardconf = np.zeros((len(Mlens),len(epsprops)))
    inverseconf = np.zeros((len(Mlens),len(epsprops)))
    forwardprobs = np.zeros((len(Mlens),len(epsprops)))
    inverseprobs = np.zeros((len(Mlens),len(epsprops)))
    epsvalsM1 = []
    epsvalsM2 = []
    for j,L in enumerate(Mlens):
        print('-----------------------')
        print('{0} of {1} lengths'.format(j+1,len(Mlens)))
        print('-----------------------')
        if lags[j][0]*(numlags-1) >= L or lags[j][1]*(numlags-1) >= L:
                print("Lag {0} is too big compared to timeseries length {1}. Skipping it.".format(max(lags[j]),L))
                continue
        M1L,M2L = makeReconstructions(ts1[:L],ts2[:L],numlags,lags[j][0],lags[j][1])
        # choose 10% of points randomly in the reconstructions to test
        N = int(np.round(0.1*M1L.shape[0]))
        ptinds = random.sample(range(M1L.shape[0]),N) 
        dists1 = cacheDistances(M1L,ptinds)
        dists2 = cacheDistances(M2L,ptinds)
        epslist1 = chooseEpsilons(M2L,epsprops) # M2 is range in forward continuity 
        epslist2 = chooseEpsilons(M1L,epsprops) # M1 is range in inverse continuity
        epsvalsM1.append(epslist2)
        epsvalsM2.append(epslist1)
        for k,eps1 in enumerate(epslist1):
            print('{0} of {1} epsilons'.format(k+1,len(epslist1)))
            forwardconf[j,k],forwardprobs[j,k] = continuityTest(dists1,dists2,ptinds,eps1,epslist2[k])
            inverseconf[j,k],inverseprobs[j,k] = continuityTest(dists2,dists1,ptinds,epslist2[k],eps1)
    return forwardconf, inverseconf,epsvalsM1,epsvalsM2,forwardprobs,inverseprobs

def convergenceWithContinuityTestFixedLags(ts1,ts2,numlags,lag1,lag2,tsprops=np.arange(0.4,1.1,0.2),epsprops=np.array([0.005,0.01,0.02,0.05,0.1,0.2])):
    '''
    Perform Pecora's continuity tests between reconstructions M1 and M2 built from
    time series ts1 and ts2 with parameters numlags (embedding dimension) and lag1 
    and lag2 (lag sizes). The continuity test will be performed in the forward and 
    reverse directions for increasing proportions of the time series (proportions 
    given in tsprops) and for a collection of epsilons. The epsilons are given as 
    proportions (in epsprops) of the standard deviation of each M1 and M2 for each 
    length.

    ts1 and ts2 are 1D numpy arrays, numlags, lag1, and lag2 are all integers, and 
    tsprops and epsprops are numpy arrays with values between 0 and 1 with default 
    values in the function argument list. 

    The output is two numpy arrays, one containing the results of the forward 
    continuity test and the other of the inverse test. The rows correspond to different
    time series lengths, and the columns to different epsilons.

    If we have high confidence for some small epsilons and that confidence increases
    with increasing time series length, then we conclude that we have circumstantial
    evidence for a continuous function. 

    '''
    Mlens = (np.round(len(ts1)*tsprops)).astype(int)
    M1,M2 = makeReconstructions(ts1,ts2,numlags,lag1,lag2) 
    reflag = max(lag1,lag2)
    badL = np.nonzero(Mlens < reflag*(numlags-1) )[0]
    if np.any( badL ):
        for n in badL:
            print("Lag {0} is too big compared to timeseries length {1}. Skipping it.".format(reflag,Mlens[n]))
    forwardconf = np.zeros((len(Mlens),len(epsprops)))
    inverseconf = np.zeros((len(Mlens),len(epsprops)))
    forwardprobs = np.zeros((len(Mlens),len(epsprops)))
    inverseprobs = np.zeros((len(Mlens),len(epsprops)))
    epsvalsM1 = []
    epsvalsM2 = []
    for j,L in enumerate(Mlens):
        print('-----------------------')
        print('{0} of {1} lengths'.format(j+1,len(Mlens)))
        print('-----------------------')
        if j in badL:
            continue
        M1L = M1[:L-reflag*(numlags-1),:]
        M2L = M2[:L-reflag*(numlags-1),:]
        # choose 10% of points randomly in the reconstructions to test
        N = int(np.round(0.1*M1L.shape[0]))
        ptinds = random.sample(range(M1L.shape[0]),N) 
        dists1 = cacheDistances(M1L,ptinds)
        dists2 = cacheDistances(M2L,ptinds)
        epslist1 = chooseEpsilons(M2L,epsprops) # M2 is range in forward continuity 
        epslist2 = chooseEpsilons(M1L,epsprops) # M1 is range in inverse continuity
        epsvalsM1.append(epslist2)
        epsvalsM2.append(epslist1)
        for k,eps1 in enumerate(epslist1):
            print('{0} of {1} epsilons'.format(k+1,len(epslist1)))
            forwardconf[j,k],forwardprobs[j,k] = continuityTest(dists1,dists2,ptinds,eps1,epslist2[k])
            inverseconf[j,k],inverseprobs[j,k] = continuityTest(dists2,dists1,ptinds,epslist2[k],eps1)
    return forwardconf, inverseconf,epsvalsM1,epsvalsM2,forwardprobs,inverseprobs

def convergenceWithContinuityTestMultipleLagsFixedEps(ts1,ts2,numlags,lags,tsprops=np.arange(0.4,1.1,0.2),epsprops=np.array([0.005,0.01,0.02,0.05,0.1,0.2])):
    '''
    Perform Pecora's continuity tests between reconstructions M1 and M2 built from
    time series ts1 and ts2 with parameters numlags (embedding dimension) and lags 
    (lag sizes). The continuity test will be performed in the forward and reverse 
    directions for increasing proportions of the time series (proportions given in 
    tsprops) and for a collection of epsilons. The epsilons are given as proportions 
    (in epsprops) of the standard deviation of each M1 and M2 for each length.

    ts1 and ts2 are 1D numpy arrays, numlags is an integer, lags is a list of lists, 
    and tsprops and epsprops are numpy arrays with values between 0 and 1 with default 
    values in the function argument list. lags will be a list containing n 2-element 
    lists where n = len(tsprops). Each sublist is of the form [lag1N, lag2N], where
    lag1N is the appropriate lag for ts1[:N] and likewise for lag2N. It is an error
    if the lengths of lags and tsprops do not match.

    The output is two numpy arrays, one containing the results of the forward 
    continuity test and the other of the inverse test. The rows correspond to different
    time series lengths, and the columns to different epsilons.

    If we have high confidence for some small epsilons and that confidence increases
    with increasing time series length, then we conclude that we have circumstantial
    evidence for a continuous function. 

    '''
    if len(lags) != len(tsprops):
        raise ValueError("The lengths of lags and tsprops must be equal.")
    Mlens = (np.round(len(ts1)*tsprops)).astype(int)
    M1,M2 = makeReconstructions(ts1,ts2,numlags,lags[-1][0],lags[-1][1]) 
    epslist1 = chooseEpsilons(M2,epsprops) # M2 is range in forward continuity 
    epslist2 = chooseEpsilons(M1,epsprops) # M1 is range in inverse continuity
    forwardconf = np.zeros((len(Mlens),len(epsprops)))
    inverseconf = np.zeros((len(Mlens),len(epsprops)))
    forwardprobs = np.zeros((len(Mlens),len(epsprops)))
    inverseprobs = np.zeros((len(Mlens),len(epsprops)))
    for j,L in enumerate(Mlens):
        print('-----------------------')
        print('{0} of {1} lengths'.format(j+1,len(Mlens)))
        print('-----------------------')
        if lags[j][0]*(numlags-1) >= L or lags[j][1]*(numlags-1) >= L:
                print("Lag {0} is too big compared to timeseries length {1}. Skipping it.".format(max(lags[j]),L))
                continue
        M1L,M2L = makeReconstructions(ts1[:L],ts2[:L],numlags,lags[j][0],lags[j][1])
        # choose 10% of points randomly in the reconstructions to test
        N = int(np.round(0.1*M1L.shape[0]))
        ptinds = random.sample(range(M1L.shape[0]),N) 
        dists1 = cacheDistances(M1L,ptinds)
        dists2 = cacheDistances(M2L,ptinds)
        for k,eps1 in enumerate(epslist1):
            print('{0} of {1} epsilons'.format(k+1,len(epslist1)))
            forwardconf[j,k],forwardprobs[j,k] = continuityTest(dists1,dists2,ptinds,eps1,epslist2[k])
            inverseconf[j,k],inverseprobs[j,k] = continuityTest(dists2,dists1,ptinds,epslist2[k],eps1)
    return forwardconf, inverseconf,epslist1,epslist2,forwardprobs,inverseprobs

def convergenceWithContinuityTestFixedLagsFixedEps(ts1,ts2,numlags,lag1,lag2,tsprops=np.arange(0.4,1.1,0.2),epsprops=np.array([0.005,0.01,0.02,0.05,0.1,0.2])):
    '''
    Perform Pecora's continuity tests between reconstructions M1 and M2 built from
    time series ts1 and ts2 with parameters numlags (embedding dimension) and lag1 
    and lag2 (lag sizes). The continuity test will be performed in the forward and 
    reverse directions for increasing proportions of the time series (proportions 
    given in tsprops) and for a collection of epsilons. The epsilons are given as 
    proportions (in epsprops) of the standard deviation of each M1 and M2 for each 
    length.

    ts1 and ts2 are 1D numpy arrays, numlags, lag1, and lag2 are all integers, and 
    tsprops and epsprops are numpy arrays with values between 0 and 1 with default 
    values in the function argument list. 

    The output is two numpy arrays, one containing the results of the forward 
    continuity test and the other of the inverse test. The rows correspond to different
    time series lengths, and the columns to different epsilons.

    If we have high confidence for some small epsilons and that confidence increases
    with increasing time series length, then we conclude that we have circumstantial
    evidence for a continuous function. 

    '''
    Mlens = (np.round(len(ts1)*tsprops)).astype(int)
    M1,M2 = makeReconstructions(ts1,ts2,numlags,lag1,lag2) 
    epslist1 = chooseEpsilons(M2,epsprops) # M2 is range in forward continuity 
    epslist2 = chooseEpsilons(M1,epsprops) # M1 is range in inverse continuity
    reflag = max(lag1,lag2)
    badL = np.nonzero(Mlens < reflag*(numlags-1) )[0]
    if np.any( badL ):
        for n in badL:
            print("Lag {0} is too big compared to timeseries length {1}. Skipping it.".format(reflag,Mlens[n]))
    forwardconf = np.zeros((len(Mlens),len(epsprops)))
    inverseconf = np.zeros((len(Mlens),len(epsprops)))
    forwardprobs = np.zeros((len(Mlens),len(epsprops)))
    inverseprobs = np.zeros((len(Mlens),len(epsprops)))
    for j,L in enumerate(Mlens):
        print('-----------------------')
        print('{0} of {1} lengths'.format(j+1,len(Mlens)))
        print('-----------------------')
        if j in badL:
            continue
        # shave off the portion of the reconstructions that we're testing this iterate
        M1L = M1[:L-reflag*(numlags-1),:]
        M2L = M2[:L-reflag*(numlags-1),:]
        # choose 10% of points randomly in the reconstructions for testing
        N = int(np.round(0.1*M1L.shape[0]))
        ptinds = random.sample(range(M1L.shape[0]),N) 
        # cache distances to the random points
        dists1 = cacheDistances(M1L,ptinds)
        dists2 = cacheDistances(M2L,ptinds)
        for k,eps1 in enumerate(epslist1):
            print('{0} of {1} epsilons'.format(k+1,len(epslist1)))
            forwardconf[j,k],forwardprobs[j,k] = continuityTest(dists1,dists2,ptinds,eps1,epslist2[k])
            inverseconf[j,k],inverseprobs[j,k] = continuityTest(dists2,dists1,ptinds,epslist2[k],eps1)
    return forwardconf, inverseconf,epslist1,epslist2,forwardprobs,inverseprobs

