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
    Calculate a confidence that points are _not_ randomly distributed
    over the manifolds M1 and M2. 

    We observed that the ndelta nearest points to x mapped to within 
    epsilon of the image point f(x). We also observed that a total of 
    neps points were within epsilon of f(x). Under the null hypothesis 
    that the points in M1 and M2 are randomly distributed, the 
    probability of our observation of ndelta successes is given by
    (neps / numpts)**ndelta, where numpts is the number of points in M1
    (and likewise in M2). If this probability is small compared to 1, and 
    also small compared to the maximum probability (p_max) in the binomial 
    distribution B(ndelta, neps/numpts), then we are confident that the 
    points are not randomly distributed. We define the confidence level
    to be 1 - (neps / numpts)**ndelta / p_max.

    Since we made our observations to be consistent with the definition
    of continuity, we say we are relatively confident (output near 1) or 
    not (output near 0) of the presence of a continuous function between
    M1 and M2. 

    '''
    p = float(neps) / numpts
    pmax = getBinomialMax(ndelta,p)
    return 1 - (p**ndelta)/pmax

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
    for k in range(len(ptinds)):
        neps = countPtsWithinEps(dists2[k],eps)
        if neps > 0: # if eps big enough, continue; else leave 0 in place
            delta = 2*startdelta
            out = False
            while out is False:
                delta = delta*0.5
                out = countDeltaPtsMappedToEps(dists1[k],dists2[k],delta,eps) 
            if out: #out can be 0, in which case we want to report 0 confidence
                contstat[k] = getContinuityConfidence(neps,out,len(dists1[k]))
    return np.mean(contstat)

def cacheDistances(M,ptinds):
    dists = []
    for ind in ptinds:
        dists.append(np.sqrt(((M - M[ind,:])**2).sum(axis=1)))
    return dists
 
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
    reconstructions, where the lengths are given by the proportions in 
    masterts multiplied by the number of points in M1 (or M2). 
    The method is also performed for varying continuity parameter epsilon,
    since a priori an appropriate epsilon is unknown. The epsilon parameters
    to test are given by the proportions in mastereps multiplied by the standard 
    deviation of the distances of points in M1 and M2 from their respective 
    mean values.

    We are checking for convergence patterns in masterts and in mastereps
    to establish a confidence level for continuity and inverse continuity
    between M1 and M2. 

    '''
    Mlens = (masterts*M1.shape[0]).astype(int)
    epslist1 = chooseEpsilons(M2,mastereps) # M2 is range in forward continuity 
    epslist2 = chooseEpsilons(M1,mastereps) # M1 is range in inverse continuity
    forwardconf = np.zeros((len(Mlens),len(epslist1)))
    inverseconf = np.zeros((len(Mlens),len(epslist2)))
    for j,L in enumerate(Mlens):
        print('-----------------------')
        print('{0} of {1} lengths'.format(j+1,len(Mlens)))
        print('-----------------------')
        ptinds = random.sample(range(L),N) # different points for each different reconstruction len
        dists1 = cacheDistances(M1[:L,:],ptinds)
        dists2 = cacheDistances(M2[:L,:],ptinds)
        for k,eps1 in enumerate(epslist1):
            print('{0} of {1} epsilons'.format(k+1,len(epslist1)))
            forwardconf[j,k] = continuityTest(dists1,dists2,ptinds,eps1,epslist2[k])
            inverseconf[j,k] = continuityTest(dists2,dists1,ptinds,epslist2[k],eps1)
    return forwardconf, inverseconf


