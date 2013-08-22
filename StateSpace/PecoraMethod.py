import numpy as np
from scipy.misc import comb
import random
import sys

def getBinomialMax(n,p):
    '''
    Return the maximum value of the binomial distribution 
    parameterized by n observations with p probability of 
    success.

    '''
    maxloc = np.floor( (n+1)*p )
    return comb(n,maxloc)*(p**maxloc)*((1-p)**(n-maxloc)) 

def getContinuityConfidence(neps,ndelta,numpts):
    '''
    Compare the probability of observing that all ndelta points
    map to the epsilon set with the null hypthesis that the
    points are randomly distributed. Random distribution 
    corresponds to neps/numpts probability per point, where neps
    is the number of points observed within eps of the image
    point and numpts is the total number of points in the 
    reconstruction. Successes (falling within eps of the image) should
    follow a binomial distribution for an underlyning random distribution.
    If ndelta is large enough to be unlikely, then we can reject
    the null hypothesis of random distribution. Since the points
    are consistent with the definition of continuity, we say we 
    are relatively confident (or not) of the presence of a 
    continuous function.

    '''
    p = float(neps) / numpts
    pmax = getBinomialMax(ndelta,p)
    return 1 - (p**ndelta)/pmax

def countPtsWithinEps(M,ind,eps):
    '''
    M is an mxn numpy array with one n-dimensional point on each
    row. Count the number of points within eps of M[ind,:]. Exclude 
    the M[ind,:] point itself by subtracting 1 from the count.

    '''
    dists = np.sqrt(((M - M[ind,:])**2).sum(1))
    return (dists < eps).sum() - 1 

def findPtsWithinDelta(M,ind,delta):
    '''
    M is an mxn numpy array with one n-dimensional point on each
    row. Find the indices of the points within delta of M[ind,:]. 
    Exclude ind from the returned indices.

    '''
    dists = np.sqrt(((M - M[ind,:])**2).sum(1))
    inds = list(np.nonzero(dists < delta)[0])
    inds.remove(ind)
    return inds

def countDeltaPtsMappedToEps(M1,M2,delta,eps,ind):
    '''
    Find all points within delta of M1[ind,:], then check to see
    if their images fall within eps of M2[ind,:]. If they all do,
    we have a success and return the number of points within delta.
    If we fail, then return False (delta is too big). If there are 
    no points within delta, then return None (delta is too small).

    '''
    deltainds = findPtsWithinDelta(M1,ind,delta)
    if len(deltainds) == 0:
        return None
    for k in deltainds:
        if np.sqrt(((M2[k,:] - M2[ind,:])**2).sum()) >= eps:
            return False
    return len(deltainds)

def continuityTest(M1,M2,ptinds,eps,startdelta):
    '''
    Do Pecora continuity test on the reconstructions M1 and M2 using 
    the points with indices ptinds and continuity parameter eps. 
    Guess delta values beginning with startdelta, which ideally has the
    relation startdelta/|M1| is approximately eps/|M2|.

    '''
    contstat = np.zeros(len(ptinds))
    for k,ind in enumerate(ptinds):
        neps = countPtsWithinEps(M2,ind,eps)
        if neps > 0: # if eps big enough, continue; else leave 0 in place
            delta = 2*startdelta
            out = False
            while out is False:
                delta = delta*0.5
                out = countDeltaPtsMappedToEps(M1,M2,delta,eps,ind) 
            if out: #out can be None, in which case we want to leave zero in contstat
                contstat[k] = getContinuityConfidence(neps,out,M1.shape[0])
    return np.mean(contstat)

def chooseEpsilons(M,mastereps):
    '''
    Estimate the standard deviation of the manifold M (sensu Pecora)
    and take fractions of it for different epsilons. The proportions
    to take are given in mastereps, a numpy array.

    '''
    meanM = np.mean(M,axis=0)
    dists = np.sqrt(((M - meanM)**2).sum(axis=1))
    stdM = np.std(dists)
    return stdM*mastereps

def convergenceWithContinuityTest(M1,M2,N,masterts=np.arange(0.2,1.1,0.2),mastereps=np.array([0.005,0.01,0.02,0.05,0.1,0.2])):
    '''
    Do Pecora method (continuity and inverse continuity) on the 
    reconstructions M1 and M2, which are mxn numpy arrays of m 
    points in n dimensions. 
    The method is performed on N random points for increasing length
    reconstructions. The proportional lengths of the full reconstructions
    to be used are in masterts.
    The method is also performed for varying continuity parameter epsilon,
    since a priori an appropriate epsilon is unknown. The proportions in
    mastereps are multiplied by the standard deviation of the distances of 
    points in M1 and M2 from their respective mean values to give the 
    epsilon parameters that will be tested. 

    We are checking for convergence patterns in masterts and in mastereps
    to establish a confidence level for continuity and inverse continuity
    between M1 and M2. 

    '''
    epslist1 = chooseEpsilons(M2,mastereps) # M2 is range in forward continuity 
    epslist2 = chooseEpsilons(M1,mastereps) # M1 is range in inverse continuity
    Mlens = (masterts*M1.shape[0]).astype(int)
    forwardconf = np.zeros((len(Mlens),len(epslist1)))
    inverseconf = np.zeros((len(Mlens),len(epslist2)))
    for j,L in enumerate(Mlens):
        print('-----------------------')
        print('{0} of {1} lengths'.format(j+1,len(Mlens)))
        print('-----------------------')
        ptinds = random.sample(range(L),N) # different points for each different reconstruction len
        M1L = M1[:L,:]
        M2L = M2[:L,:]
        for k,eps1 in enumerate(epslist1):
            print('{0} of {1} epsilons'.format(k+1,len(epslist1)))
            forwardconf[j,k] = continuityTest(M1L,M2L,ptinds,eps1,epslist2[k])
            inverseconf[j,k] = continuityTest(M2L,M1L,ptinds,epslist2[k],eps1)
    return forwardconf, inverseconf


